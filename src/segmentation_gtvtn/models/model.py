import tensorflow as tf

from src.models.layers import ResidualLayer


class UpBlock(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 upsampling_factor=1,
                 filters_output=24,
                 n_conv=2
                 ):
        super().__init__()
        self.upsampling_factor = upsampling_factor
        self.conv = tf.keras.Sequential()
        for k in range(n_conv):
            self.conv.add(
                tf.keras.layers.Conv3D(filters,
                                       3,
                                       padding='SAME',
                                       activation='relu'), )
        self.trans_conv = tf.keras.layers.Conv3DTranspose(filters,
                                                          3,
                                                          strides=(2, 2, 2),
                                                          padding='SAME',
                                                          activation='relu')
        self.concat = tf.keras.layers.Concatenate()
        if upsampling_factor != 1:
            self.upsampling = tf.keras.Sequential([
                tf.keras.layers.Conv3D(filters_output,
                                       1,
                                       padding='SAME',
                                       activation='relu'),
                tf.keras.layers.UpSampling3D(size=(upsampling_factor,
                                                   upsampling_factor,
                                                   upsampling_factor)),
            ])
        else:
            self.upsampling = None

    def call(self, inputs):
        x, skip = inputs
        x = self.trans_conv(x)
        x = self.concat([x, skip])
        x = self.conv(x)
        if self.upsampling:
            return x, self.upsampling(x)
        else:
            return x


def get_first_block(input_layer, filters):
    x = get_residual_layer(input_layer,
                           filters,
                           7,
                           padding="SAME",
                           projection=True)
    x = get_residual_layer(x, filters, 3, padding="SAME", projection=False)
    return x


def get_residual_layer(input_layer, *args, projection=True, **kwargs):
    x = tf.keras.layers.Conv3D(*args, **kwargs)(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)

    if projection:
        input_proj = tf.keras.layers.Conv3D(args[0], 1, **kwargs)(input_layer)
        input_proj = tf.keras.layers.BatchNormalization()(input_proj)
        return x + input_proj
    else:
        return x + input_layer


def get_down_block(input_layer, filters):
    x = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2),
                                  padding='SAME')(input_layer)
    x = get_residual_layer(x, filters, 3, padding="SAME", projection=True)
    x = get_residual_layer(x, filters, 3, padding="SAME", projection=False)
    x = get_residual_layer(x, filters, 3, padding="SAME", projection=False)
    return x


class UnetGTVtn(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.down_stack = [
            self.get_first_block(8),
            self.get_down_block(16),
            self.get_down_block(32),
            self.get_down_block(64),
            self.get_down_block(128),
        ]
        self.up_stack = [
            UpBlock(64, upsampling_factor=8, filters_output=8),
            UpBlock(32, upsampling_factor=4, filters_output=8),
            UpBlock(16, upsampling_factor=2, filters_output=8),
            UpBlock(8, n_conv=1),
        ]
        self.last_gtvt = tf.keras.Sequential([
            tf.keras.layers.Conv3D(24, 3, activation='relu', padding='SAME'),
            tf.keras.layers.Conv3D(
                1, 1, activation='sigmoid', padding='SAME', name="segmentation_output_gtvt"),
        ])
        self.last_gtvn = tf.keras.Sequential([
            tf.keras.layers.Conv3D(24, 3, activation='relu', padding='SAME'),
            tf.keras.layers.Conv3D(
                1, 1, activation='sigmoid', padding='SAME', name="segmentation_output_gtvn"),
        ])

    def get_submodel(self, inputs):
        x1 = self.down_stack[0](inputs)
        x2 = self.down_stack[1](x1)
        x3 = self.down_stack[2](x2)
        x4 = self.down_stack[3](x3)
        x5 = self.down_stack[4](x4)

        return [x1, x2, x3, x4, x5]

    def get_first_block(self, filters):
        return tf.keras.Sequential([
            ResidualLayer(filters, 7, padding='SAME'),
            ResidualLayer(filters, 3, padding='SAME'),
        ])

    def get_down_block(self, filters):
        return tf.keras.Sequential([
            tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), padding='SAME'),
            ResidualLayer(filters, 3, padding='SAME'),
            ResidualLayer(filters, 3, padding='SAME'),
        ])

    def call(self, inputs):
        nmod = inputs.shape[-1]
        x = tf.keras.layers.InputLayer((144, 144, 144, nmod))(inputs)
        skips = []
        xs_downsampled = []
        for block in self.down_stack:
            x = block(x)
            skips.append(x)
            xs_downsampled.append(tf.reduce_mean(x, axis=[1, 2, 3]))
        skips = reversed(skips[:-1])
        xs_upsampled = []

        for block, skip in zip(self.up_stack, skips):
            x = block((x, skip))
            if type(x) is tuple:
                x, x_upsampled = x
                xs_upsampled.append(x_upsampled)

        x += tf.add_n(xs_upsampled)
        return self.last_gtvt(x), self.last_gtvn(x)


class UnetGTVtn2(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.down_stack = [
            self.get_first_block(8),
            self.get_down_block(16),
            self.get_down_block(32),
            self.get_down_block(64),
            self.get_down_block(128),
        ]
        self.up_stack = [
            UpBlock(64, upsampling_factor=8, filters_output=8),
            UpBlock(32, upsampling_factor=4, filters_output=8),
            UpBlock(16, upsampling_factor=2, filters_output=8),
            UpBlock(8, n_conv=1),
        ]

        self.last_gtvtn = tf.keras.Sequential([
            tf.keras.layers.Conv3D(24, 3, activation='relu', padding='SAME'),
            tf.keras.layers.Conv3D(
                3, 1, activation='softmax', padding='SAME', name="segmentation_output_gtvtn"),
        ])

    def get_submodel(self, inputs):
        x1 = self.down_stack[0](inputs)
        x2 = self.down_stack[1](x1)
        x3 = self.down_stack[2](x2)
        x4 = self.down_stack[3](x3)
        x5 = self.down_stack[4](x4)
        return [x1, x2, x3, x4, x5]

    def get_first_block(self, filters):
        return tf.keras.Sequential([
            ResidualLayer(filters, 7, padding='SAME'),
            ResidualLayer(filters, 3, padding='SAME'),
        ])

    def get_down_block(self, filters):
        return tf.keras.Sequential([
            tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), padding='SAME'),
            ResidualLayer(filters, 3, padding='SAME'),
            ResidualLayer(filters, 3, padding='SAME'),
        ])

    def call(self, inputs):
        x = tf.keras.layers.InputLayer((144, 144, 144, 2))(inputs)
        skips = []
        xs_downsampled = []
        for block in self.down_stack:
            x = block(x)
            skips.append(x)
            xs_downsampled.append(tf.reduce_mean(x, axis=[1, 2, 3]))
        skips = reversed(skips[:-1])
        xs_upsampled = []

        for block, skip in zip(self.up_stack, skips):
            x = block((x, skip))
            if type(x) is tuple:
                x, x_upsampled = x
                xs_upsampled.append(x_upsampled)

        x += tf.add_n(xs_upsampled)
        return self.last_gtvtn(x)
