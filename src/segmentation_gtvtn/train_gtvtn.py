import os
import warnings
import time
import datetime
import logging
import random

import numpy as np
import click
import setproctitle
import tensorflow as tf

from src.segmentation_gtvtn.models.model import UnetGTVtn2, UnetGTVtn
from src.segmentation_gtvtn.models.fetch_data_segmentation_gtvtn import get_tf_dataset_gtvtn, get_ds
from src.segmentation_gtvtn.models.losses import dice_loss_multiclass_indiv_agg, dice_loss, dice_loss_agg, dice_coef_multiclass, dice_coef_multiclass_indiv0, dice_coef_multiclass_indiv1, dice_coe_hard
from src.segmentation_gtvtn.evaluation.utils import dice

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")


def class_func(images, outs):
    return tf.cast(outs[-1][0][0], dtype=tf.int32)


all_cases_folder = '/mnt/nas4/datasets/ToReadme/HECKTOR/HECKTOR2021_radiomics'
bbox_path = '/mnt/nas4/datasets/ToReadme/HECKTOR/HECKTOR2021_radiomics/bbox.csv'

augment_shift = 40
isAugmentMirror = True
augment_angles = 15
isTrain = False

##############


@click.command()
@click.option('--nepochs',
              type=click.INT,
              default=1)  # 200
@click.option('--bs',
              type=click.INT,
              default=1)  # 4
@click.option('--task',
              type=click.STRING,
              default='GTVtn')
@click.option('--modalities',
              type=click.STRING,
              default='ptct')
def main(nepochs, bs, task, modalities):
    logger = logging.getLogger(__name__)
    logger.info('Training UNet segmentation GTVt and GTVn HECKTOR')

    setproctitle.setproctitle('trainGTVtn_'+modalities)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,",
                  len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    # Set seeds
    os.environ['PYTHONHASHSEED'] = '0'
    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)

    path_train = os.path.join(all_cases_folder, 'hecktor_train/')
    path_val = os.path.join(all_cases_folder, 'hecktor_val/')
    path_test = os.path.join(all_cases_folder, 'hecktor_test/')

    if task == 'GTVtn':
        losses = {
            "output_1": dice_loss_multiclass_indiv_agg
        }
        metrics = {
            "output_1": [dice_coef_multiclass, dice_coef_multiclass_indiv0, dice_coef_multiclass_indiv1],
        }
        model = UnetGTVtn2()

    elif task in ['GTVn', 'GTVt']:
        losses = {
            "output_1": dice_loss,  # segmentation_output GTVt
            "output_2": dice_loss_agg,  # segmentation_output GTVn
        }
        metrics = {
            "output_1": dice_coe_hard,
            "output_2": dice_coe_hard,
        }
        model = UnetGTVtn()
    else:
        print('task '+task+' unknown')

    # Test set data:
    hecktor_ds_test = get_tf_dataset_gtvtn(
        path_test,
        bbox_path,
        modalities=modalities,
        num_parallel_calls=None,
        isReturnPatientName=False,
        task=task,
        isTrain=isTrain)
    hecktor_ds_test = hecktor_ds_test.batch(
        1).prefetch(tf.data.experimental.AUTOTUNE)
    test_images_ds, test_masks_ds, test_masks_gtvn_ds = get_ds(
        hecktor_ds_test)
    # Validation set data:
    hecktor_ds_val = get_tf_dataset_gtvtn(
        path_val,
        bbox_path,
        modalities=modalities,
        num_parallel_calls=None,
        isReturnPatientName=False,
        task=task,
        isTrain=isTrain)
    hecktor_ds_val = hecktor_ds_val.batch(
        int(bs*2)).prefetch(tf.data.experimental.AUTOTUNE)
    #  Training set data:
    hecktor_ds_train_in = get_tf_dataset_gtvtn(
        path_train,
        bbox_path,
        modalities=modalities,
        augment_shift=augment_shift,
        isAugmentMirror=isAugmentMirror,
        augment_angles=augment_angles,
        num_parallel_calls=None,
        isReturnPatientName=False,
        task=task,
        isTrain=isTrain)
    hecktor_ds_train = hecktor_ds_train_in.batch(
        bs).prefetch(tf.data.experimental.AUTOTUNE)

    checkpoint_filepath = './weights/train_gtvtn/checkpoint_nep' + \
        str(nepochs)+modalities+'_'+task

    if isTrain:
        # Define model and hyper-params
        t_mul = 1.1
        m_mul = 0.95
        alpha = 1e-3
        lr_init = 1e-3
        # warm 10 epochs
        first_decay_steps = 10
        lr = (tf.keras.experimental.CosineDecayRestarts(
            initial_learning_rate=lr_init,
            first_decay_steps=first_decay_steps,
            t_mul=t_mul,
            m_mul=m_mul,
            alpha=alpha
        ))
        monitor = 'val_loss'

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor=monitor, patience=20)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        if task == 'GTVtn':
            lossWeights = {"output_1": 1.}
        elif task == 'GTVn':
            lossWeights = {"output_1": 0.,
                           "output_2": 1.}
        elif task == 'GTVt':
            lossWeights = {"output_1": 1.,
                           "output_2": 0.}
        model.compile(optimizer,
                      loss=losses,
                      loss_weights=lossWeights,
                      metrics=metrics,
                      run_eagerly=False)  # True to debug
        # Callbacks
        log_dir = "logs/fit/train_gtvtn_"+modalities+'_'+task+"_nep" + \
            str(nepochs) + \
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        print('log_dir:', log_dir)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir)
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor=monitor,
            mode='min',
            verbose=1,
            save_best_only=True)

        start_time = time.time()
        model.fit(
            hecktor_ds_train,
            epochs=nepochs,
            validation_data=hecktor_ds_val,
            validation_freq=1,
            verbose=1,
            callbacks=[
                model_checkpoint_callback, early_stopping_callback,
                tensorboard_callback
            ]
        )
        print('model fit running time = ', (time.time() - start_time))
    # Load the best model based on val loss
    try:
        model.load_weights(checkpoint_filepath)
        print('best model (based on val loss) loaded')
    except:
        print('model', checkpoint_filepath,
              'not found / or failed loading')

    # Evaluate segmentation on test set
    preds = model.predict(test_images_ds, batch_size=1, verbose=0)
    if task == 'GTVtn':
        preds_gtvt = preds[..., 1:2]
        preds_gtvn = preds[..., 2:3]
    else:
        preds_gtvt = preds[0]
        if len(preds) > 1:
            preds_gtvn = preds[1]
        else:
            preds_gtvn = None

    dice_gtvt_fold = [dice(mask.squeeze(), preds_gtvt[i].squeeze() > 0.5)
                      for i, mask in enumerate(test_masks_ds)]
    if preds_gtvn is not None:
        # For dice on GTVn, remove test cases without GTVn
        ind_gtvn = np.sum(test_masks_gtvn_ds, axis=(1, 2, 3, 4)) > 0
        dice_gtvn_fold = [dice(mask.squeeze(), preds_gtvn[ind_gtvn][i].squeeze(
        ) > 0.5) for i, mask in enumerate(test_masks_gtvn_ds[ind_gtvn])]
    dice_test_gtvt = np.mean(dice_gtvt_fold)
    dice_test_gtvt_std = np.std(dice_gtvt_fold)
    dice_test_gtvn = np.mean(dice_gtvn_fold)
    dice_test_gtvn_std = np.std(dice_gtvn_fold)
    # Aggregated dice GTVn:
    dice_test_gtvt_agg = dice(test_masks_ds, preds_gtvt)
    dice_test_gtvn_agg = dice(test_masks_gtvn_ds, preds_gtvn)

    # Write results to text
    path_results = 'results/results_'+modalities+'_'+task+'_nep'+str(nepochs) + \
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.txt'
    with open(path_results, 'w') as f:
        f.write('Dice test gtvt: '+str(dice_test_gtvt) +
                ' +-'+str(dice_test_gtvt_std)+'\n')
        f.write('Dice test gtvt agg: '+str(dice_test_gtvt_agg)+'\n')
        f.write('Dice test gtvn: '+str(dice_test_gtvn) +
                ' +-'+str(dice_test_gtvn_std)+'\n')
        f.write('Dice test gtvn agg: '+str(dice_test_gtvn_agg)+'\n')

    print('Dice test GTVt', dice_test_gtvt)
    print('Dice test GTVt aggregated', dice_test_gtvt_agg)
    print('Dice test GTVn', dice_test_gtvn)
    print('Dice test GTVn aggregated', dice_test_gtvn_agg)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logging.captureWarnings(True)

    main()
