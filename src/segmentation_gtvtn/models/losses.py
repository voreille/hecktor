import tensorflow as tf


def dice_loss_multiclass_indiv_agg(y_true, y_pred):
    # Aggregated unions and intersection over the batch for GTVn
    return 2-(dice_coef_multiclass_indiv0(y_true, y_pred)+dice_coef_multiclass_indiv1_agg(y_true, y_pred))


def dice_loss(y_true, y_pred, loss_type='sorensen', smooth=1e-7):
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_f = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=(1, 2, 3, 4))
    if loss_type == 'jaccard':
        union = tf.reduce_sum(tf.square(y_pred_f), axis=(
            1, 2, 3, 4)) + tf.reduce_sum(tf.square(y_true_f), axis=(1, 2, 3, 4))
    elif loss_type == 'sorensen':
        union = tf.reduce_sum(y_pred_f, axis=(1, 2, 3, 4)) + \
            tf.reduce_sum(y_true_f, axis=(1, 2, 3, 4))
    return 1 - tf.reduce_mean((2. * intersection + smooth) / (union + smooth))


def dice_loss_agg(y_true, y_pred, loss_type='sorensen', smooth=1e-7):
    # aggregate unions and intersection on the batch
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_f = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=(0, 1, 2, 3, 4))
    if loss_type == 'jaccard':
        union = tf.reduce_sum(tf.square(y_pred_f), axis=(
            0, 1, 2, 3, 4)) + tf.reduce_sum(tf.square(y_true_f), axis=(0, 1, 2, 3, 4))
    elif loss_type == 'sorensen':
        union = tf.reduce_sum(y_pred_f, axis=(0, 1, 2, 3, 4)) + \
            tf.reduce_sum(y_true_f, axis=(0, 1, 2, 3, 4))
    return 1 - (2. * intersection + smooth) / (union + smooth)


def dice_coef_multiclass(y_true, y_pred, smooth=1e-7):
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_f = tf.cast(y_pred[..., 1:], tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=(1, 2, 3, 4))
    union = tf.reduce_sum(y_pred_f+y_true_f, axis=(1, 2, 3, 4))
    return tf.reduce_mean((2. * intersection) / (union + smooth))


def dice_coef_multiclass_indiv0(y_true, y_pred, smooth=1e-7):
    y_true_f = tf.cast(y_true[..., 0], tf.float32)
    y_pred_f = tf.cast(y_pred[..., 1], tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=(1, 2, 3))
    union = tf.reduce_sum(y_pred_f, axis=(1, 2, 3)) + \
        tf.reduce_sum(y_true_f, axis=(1, 2, 3))
    return tf.reduce_mean((2. * intersection + smooth) / (union + smooth))


def dice_coef_multiclass_indiv1(y_true, y_pred, smooth=1e-7):
    y_true_f = tf.cast(y_true[..., 1], tf.float32)
    y_pred_f = tf.cast(y_pred[..., 2], tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=(1, 2, 3))
    union = tf.reduce_sum(y_pred_f, axis=(1, 2, 3)) + \
        tf.reduce_sum(y_true_f, axis=(1, 2, 3))
    return tf.reduce_mean((2. * intersection + smooth) / (union + smooth))


def dice_coef_multiclass_indiv1_agg(y_true, y_pred, smooth=1e-7):
    y_true_f = tf.cast(y_true[..., 1], tf.float32)
    y_pred_f = tf.cast(y_pred[..., 2], tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=(0, 1, 2, 3))
    union = tf.reduce_sum(y_pred_f, axis=(0, 1, 2, 3)) + \
        tf.reduce_sum(y_true_f, axis=(0, 1, 2, 3))
    return (2. * intersection + smooth) / (union + smooth)


def dice_coe_hard(y_true, y_pred, loss_type='sorensen', smooth=1.):
    return dice_coe(y_true, tf.cast(y_pred > 0.5, tf.float32), loss_type=loss_type, smooth=smooth)


def dice_coe(y_true, y_pred, loss_type='jaccard', smooth=1.):
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3, 4))
    if loss_type == 'jaccard':
        union = tf.reduce_sum(tf.square(y_pred), axis=(1, 2, 3, 4)) + tf.reduce_sum(
            tf.square(y_true), axis=(1, 2, 3, 4))
    elif loss_type == 'sorensen':
        union = tf.reduce_sum(y_pred, axis=(1, 2, 3, 4)) + \
            tf.reduce_sum(y_true, axis=(1, 2, 3, 4))
    else:
        raise ValueError("Unknown `loss_type`: %s" % loss_type)
    return tf.reduce_mean((2. * intersection + smooth) / (union + smooth))
