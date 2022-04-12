import os
import numpy as np
import tensorflow as tf
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import SimpleITK as sitk
from skimage import segmentation
import cc3d

from src.resampling.utils import get_np_volume_from_sitk
from src.resampling.resampling import resample_np_volume as resample_spline
from src.models.losses import dice_coef_multiclass_indiv1_agg


def dice(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_pred) + np.sum(y_true)
    return (2. * intersection + 0.000001) / (union + 0.000001)


def count(counts, batch):
    _, outs = batch
    labels = outs[-1][0]
    class_1 = labels == 1
    class_1 = tf.cast(class_1, tf.int32)

    class_0 = labels == 0
    class_0 = tf.cast(class_0, tf.int32)

    counts['class_0'] += tf.reduce_sum(class_0)
    counts['class_1'] += tf.reduce_sum(class_1)

    return counts


def compute_fraction(ds):
    counts = ds.reduce(
        initial_state={'class_0': 0, 'class_1': 0}, reduce_func=count)
    counts = np.array([counts['class_0'].numpy(),
                       counts['class_1'].numpy()]).astype(np.float32)
    fractions = counts/counts.sum()
    print('fraction:', fractions, 'counts:', counts)
    return fractions, counts


def confusion_matrix(pred, gt, normalization='true'):
    pred = pred.reshape([-1, 2])
    pred[pred <= 0.5] = 0
    pred[pred > 0.5] = 1
    pred = pred[:, 0]+2*pred[:, 1]
    gt = gt.reshape([-1, 2])
    gt = gt[:, 0]+2*gt[:, 1]
    cm = sklearn.metrics.confusion_matrix(pred, gt, normalize=normalization)

    df_cm = pd.DataFrame(cm, index=[i for i in ['background', 'GTVt', 'GTVn']], columns=[
                         i for i in ['background', 'GTVt', 'GTVn']])
    plt.figure(figsize=(10, 7))
    sn.set(font_scale=2)
    sn.heatmap(df_cm, annot=True, cmap="Blues")
    plt.savefig("cm2.png")
    return cm


def select_slice(contour, sagittal=False):
    positions = np.where(contour != 0)
    if sagittal:
        axis = 1
    else:
        axis = 2
    return (np.max(positions[axis]) + np.min(positions[axis])) // 2


def plot_contours(i_im, np_gt, np_pred):
    np_gt_gtvt = np.squeeze(np_gt[0])
    np_gt_gtvn = np.squeeze(np_gt[1])
    np_pred_gtvt = np.squeeze(np_pred[..., 1])
    np_pred_gtvn = np.squeeze(np_pred[..., 2])
    # reload the image
    path_im = '/mnt/nas4/datasets/ToReadme/HECKTOR/HECKTOR2021_radiomics/hecktor_test/'
    path_bb = '/mnt/nas4/datasets/ToReadme/HECKTOR/HECKTOR2021_radiomics/bbox.csv'
    list_patients = [x[1] for x in os.walk(path_im)][0]
    patient = list_patients[i_im]
    print(i_im, patient)

    bb_df = pd.read_csv(path_bb).set_index('PatientID')
    sitk_CT = sitk.ReadImage(os.path.join(
        path_im, patient, patient+'_ct.nii.gz'))
    sitk_PT = sitk.ReadImage(os.path.join(
        path_im, patient, patient+'_pt.nii.gz'))
    np_CT, pixel_spacing_ct, origin_ct = get_np_volume_from_sitk(sitk_CT)
    np_PT, pixel_spacing_pt, origin_pt = get_np_volume_from_sitk(sitk_PT)
    # resample to CT resolution
    pixel_spacing_resampling = (1, 1, 1)
    bb = (bb_df.loc[patient, "x1"], bb_df.loc[patient, "y1"], bb_df.loc[patient, "z1"],
          bb_df.loc[patient, "x2"], bb_df.loc[patient, "y2"], bb_df.loc[patient, "z2"])
    np_PT = resample_spline(
        np_PT, origin_pt, pixel_spacing_pt, pixel_spacing_resampling, bb)
    np_CT = resample_spline(
        np_CT, origin_ct, pixel_spacing_ct, pixel_spacing_resampling, bb)

    # Select slice
    if np.sum(np_gt_gtvn) == 0:
        slicePlot = select_slice(np_gt_gtvt)
    else:
        slicePlot = int((select_slice(np_gt_gtvn)+select_slice(np_gt_gtvt))/2)
    pred_slice_gtvn = np_pred_gtvn[:, :, slicePlot].T
    pred_slice_gtvt = np_pred_gtvt[:, :, slicePlot].T
    gt_slice_gtvn = np_gt_gtvn[:, :, slicePlot].T
    CT_slice = np_CT[:, :, slicePlot].T
    CT_slice[CT_slice < -160] = -160
    CT_slice[CT_slice > 240] = 240
    CT_slice = (CT_slice-np.min(CT_slice))/(np.max(CT_slice)-np.min(CT_slice))
    PT_slice = np_PT[:, :, slicePlot].T
    SUVcut = 7
    PT_slice[PT_slice > SUVcut] = SUVcut
    PT_slice = (PT_slice-np.min(PT_slice))/(np.max(PT_slice)-np.min(PT_slice))
    # Draw the contours and plot
    contour_pred_gtvn = segmentation.clear_border(
        pred_slice_gtvn).astype(np.int)
    contour_gt = segmentation.clear_border(gt_slice_gtvn).astype(np.int)
    '''
    CT_plot_gtvn = segmentation.mark_boundaries(CT_slice, contour_pred_gtvn,color=(0, 1, 0))
    CT_plot_gtvn = segmentation.mark_boundaries(CT_plot_gtvn, contour_gt,color=(1, 0, 0))
    PT_plot_gtvn = segmentation.mark_boundaries(1 - PT_slice, contour_pred_gtvn,color=(0, 1, 0))
    PT_plot_gtvn = segmentation.mark_boundaries(PT_plot_gtvn, contour_gt,color=(1, 0, 0))
    '''
    red1 = (239/255, 154/255, 154/255)
    red2 = (183/255, 28/255, 28/255)
    blue1 = (144/255, 202/255, 249/255)
    blue2 = (14/255, 71/255, 161/255)
    CT_plot_gtvtn = segmentation.mark_boundaries(
        CT_slice, contour_gt, color=red1)
    CT_plot_gtvtn = segmentation.mark_boundaries(
        CT_plot_gtvtn, contour_pred_gtvn, color=red2)
    PT_plot_gtvtn = segmentation.mark_boundaries(
        1 - PT_slice, contour_gt, color=red1)
    PT_plot_gtvtn = segmentation.mark_boundaries(
        PT_plot_gtvtn, contour_pred_gtvn, color=red2)

    gt_slice_gtvt = np_gt_gtvt[:, :, slicePlot].T
    contour_pred_gtvt = segmentation.clear_border(
        pred_slice_gtvt).astype(np.int)
    contour_gt = segmentation.clear_border(gt_slice_gtvt).astype(np.int)
    '''
    CT_plot_gtvt = segmentation.mark_boundaries(CT_slice, contour_pred_gtvt,color=(0, 1, 0))
    CT_plot_gtvt = segmentation.mark_boundaries(CT_plot_gtvt, contour_gt,color=(1, 0, 0))
    PT_plot_gtvt = segmentation.mark_boundaries(1 - PT_slice, contour_pred_gtvt,color=(0, 1, 0))
    PT_plot_gtvt = segmentation.mark_boundaries(PT_plot_gtvt, contour_gt,color=(1, 0, 0))
    '''
    CT_plot_gtvtn = segmentation.mark_boundaries(
        CT_plot_gtvtn, contour_gt, color=blue1)
    CT_plot_gtvtn = segmentation.mark_boundaries(
        CT_plot_gtvtn, contour_pred_gtvt, color=blue2)
    PT_plot_gtvtn = segmentation.mark_boundaries(
        PT_plot_gtvtn, contour_gt, color=blue1)
    PT_plot_gtvtn = segmentation.mark_boundaries(
        PT_plot_gtvtn, contour_pred_gtvt, color=blue2)
    '''
    plt.figure(figsize=(9,9))
    plt.axis('off')
    plt.imshow(CT_plot_gtvn)
    plt.savefig(f"plots/{patient}_CT_gtvn.png", bbox_inches='tight')
    plt.figure(figsize=(9,9))
    plt.axis('off')
    plt.imshow(PT_plot_gtvn)
    plt.savefig(f"plots/{patient}_PT_gtvn.png", bbox_inches='tight')
    plt.figure(figsize=(9,9))
    plt.axis('off')
    plt.imshow(CT_plot_gtvt)
    plt.savefig(f"plots/{patient}_CT_gtvt.png", bbox_inches='tight')
    plt.figure(figsize=(9,9))
    plt.axis('off')
    plt.imshow(PT_plot_gtvt)
    plt.savefig(f"plots/{patient}_PT_gtvt.png", bbox_inches='tight')
    '''
    plt.figure(figsize=(9, 9))
    plt.axis('off')
    plt.imshow(CT_plot_gtvtn)
    plt.savefig(f"plots/{patient}_CT_gtvtn.png", bbox_inches='tight')
    plt.figure(figsize=(9, 9))
    plt.axis('off')
    plt.imshow(PT_plot_gtvtn)
    plt.savefig(f"plots/{patient}_PT_gtvtn.png", bbox_inches='tight')
    return None
 
def compute_volumes(gt):
    print('number of cases:',gt.shape[0])
    # GTVt
    vols_t = np.sum(gt[...,0],axis=(1,2,3))
    print('median volume GTVt:',np.median(vols_t))
    #connected components for the GTVn
    list_N=[]
    list_vols=[]
    for i in range(gt.shape[0]):
        ccs, N = cc3d.connected_components(gt[i,...,1].astype(int), return_N=True)
        print('N=',N)
        for j in range(N):
            print('i,j:',i,j)
            vol = np.sum(ccs==j+1)
            list_vols.append(vol)
        list_N.append(N)
    list_N = np.asarray(list_N)
    vols_n = np.asarray(list_vols)
    print('median volume GTVn:', np.median(vols_n))
    print('quantiles volume GTVn:', np.quantile(vols_n,0.25), np.quantile(vols_n,0.75))
    print('average number of GTVn:', np.mean(list_N))
    print('number of cases with 0 GTVn:',np.sum(list_N==0))
    print('number of cases with 1 GTVn:',np.sum(list_N==1))
    print('number of cases with >1 GTVn:',np.sum(list_N>1))

    # histogram of sizes GTVt and GTVn
    bins = np.linspace(0, 60000, 30)
    plt.figure()
    plt.hist(vols_t, bins, alpha=0.5, label='GTVt volumes')
    plt.hist(vols_n, bins, alpha=0.5, label='GTVn volumes')
    plt.legend(loc='upper right')
    plt.title("Histogram of GTVt and GTVn volumes") 
    plt.xlabel("Voxels")
    plt.savefig("hist_volumes")
    #import pdb;pdb.set_trace()
    return None

def compute_thresh_dsc(gt,pred):
    gt=gt.squeeze()
    pred=pred.squeeze()
    list_th_q0=[]
    list_th_q1=[]
    list_th_q2=[]
    list_th_q3=[]
    list_th_all=[]
    for i in range(gt.shape[0]):
        ccs, N = cc3d.connected_components(gt[i].astype(int), return_N=True)
        #print('N=',N)
        max_vol=0
        for j in range(N):
            #print('i,j:',i,j)
            vol = np.sum(ccs==j+1)
            max_vol=np.maximum(max_vol,vol)
        # thresh at Q3
        if max_vol>12937.25:
            list_th_q3.append(i)
        # thresh at Q2
        elif max_vol>3195.5:
            list_th_q2.append(i)
        # thresh at Q1
        elif max_vol>1422.75:
            list_th_q1.append(i)
        # thresh 0:
        elif max_vol>0:
            list_th_q0.append(i)
        # no thresh:
        else:
            list_th_all.append(i)
    print(len(list_th_q0),len(list_th_q1),len(list_th_q2),len(list_th_q3))
    # all
    pred_th_all = np.repeat(pred[list_th_all][...,np.newaxis], 3, axis=-1)
    gt_th_all = np.repeat(gt[list_th_all][...,np.newaxis], 2, axis=-1)
    dice_all_gtvn_agg = dice_coef_multiclass_indiv1_agg(gt_th_all,pred_th_all).numpy()
    print('aggDice at all:',dice_all_gtvn_agg)
    # Q0
    pred_th_q0 = np.repeat(pred[list_th_q0][...,np.newaxis], 3, axis=-1)
    gt_th_q0 = np.repeat(gt[list_th_q0][...,np.newaxis], 2, axis=-1)
    dice_q0_gtvn_agg = dice_coef_multiclass_indiv1_agg(gt_th_q0,pred_th_q0).numpy()
    print('aggDice at Q0:',dice_q0_gtvn_agg)
    # Q1
    pred_th_q1 = np.repeat(pred[list_th_q1][...,np.newaxis], 3, axis=-1)
    gt_th_q1 = np.repeat(gt[list_th_q1][...,np.newaxis], 2, axis=-1)
    dice_q1_gtvn_agg = dice_coef_multiclass_indiv1_agg(gt_th_q1,pred_th_q1).numpy()
    print('aggDice at q1:',dice_q1_gtvn_agg)

    # q2
    pred_th_q2 = np.repeat(pred[list_th_q2][...,np.newaxis], 3, axis=-1)
    gt_th_q2 = np.repeat(gt[list_th_q2][...,np.newaxis], 2, axis=-1)
    dice_q2_gtvn_agg = dice_coef_multiclass_indiv1_agg(gt_th_q2,pred_th_q2).numpy()
    print('aggDice at q2:',dice_q2_gtvn_agg)
    # q3
    pred_th_q3 = np.repeat(pred[list_th_q3][...,np.newaxis], 3, axis=-1)
    gt_th_q3 = np.repeat(gt[list_th_q3][...,np.newaxis], 2, axis=-1)
    dice_q3_gtvn_agg = dice_coef_multiclass_indiv1_agg(gt_th_q3,pred_th_q3).numpy()
    print('aggDice at q3:',dice_q3_gtvn_agg)
    import pdb;pdb.set_trace()
    return None

def compute_detection_cm(gt,preds):
    preds=preds>0.5
    # Connected components (gt and preds)-> detection confusion matrix
    list_gt_nn_iou = []
    list_gt_nt_iou = []
    list_pred_nn_iou = []
    list_pred_nt_iou = []
    list_gt_tn_iou = []
    list_gt_tt_iou = []
    list_pred_tn_iou = []
    list_pred_tt_iou = []
    list_N_pred_t = []
    list_N_pred_n = []
    for i_patient in range(gt.shape[0]):
        ccs_gt_t, N_gt_t = cc3d.connected_components(gt[i_patient,...,0].astype(int), return_N=True, connectivity=26)
        #ccs_gt_t = gt[i_patient,...,0].astype(int)
        #N_gt_t = 1
        ccs_gt_n, N_gt_n = cc3d.connected_components(gt[i_patient,...,1].astype(int), return_N=True, connectivity=26)
        ccs_pred_t, N_pred_t = cc3d.connected_components(preds[i_patient,...,0].astype(int), return_N=True, connectivity=26)
        #ccs_pred_t = preds[i_patient,...,0].astype(int)
        #N_pred_t = 1
        ccs_pred_n, N_pred_n = cc3d.connected_components(preds[i_patient,...,1].astype(int), return_N=True, connectivity=26)
        list_N_pred_t.append(N_pred_t)
        list_N_pred_n.append(N_pred_n)
        # loop across gt (gtvn) ccs
        for i_gt_n in range(N_gt_n):
            cc_gt_n = ccs_gt_n==(i_gt_n+1)
            # Is there a pred cc (gtvn) that is IOU<0.5?
            iou=0.
            for i_pred_n in range(N_pred_n):
                cc_pred_n = ccs_pred_n==(i_pred_n+1)
                iou = np.maximum(iou,dice(cc_gt_n,cc_pred_n))
            list_gt_nn_iou.append(iou)
            # Is there a pred cc (gtvt) that is IOU<0.5?
            iou=0.
            for i_pred_t in range(N_pred_t):
                cc_pred_t = ccs_pred_t==(i_pred_t+1)
                iou = np.maximum(iou,dice(cc_gt_n,cc_pred_t))
            list_gt_nt_iou.append(iou)
        # loop across gt (gtvt) ccs
        for i_gt_t in range(N_gt_t):
            cc_gt_t = ccs_gt_t==(i_gt_t+1)
            # Is there a pred cc (gtvn) that is IOU<0.5?iou=0.
            iou=0.
            for i_pred_n in range(N_pred_n):
                cc_pred_n = ccs_pred_n==(i_pred_n+1)
                iou = np.maximum(iou,dice(cc_gt_t,cc_pred_n))
            list_gt_tn_iou.append(iou)
            # Is there a pred cc (gtvn) that is IOU<0.5?
            iou=0.
            for i_pred_t in range(N_pred_t):
                cc_pred_t = ccs_pred_t==(i_pred_t+1)
                iou = np.maximum(iou,dice(cc_gt_t,cc_pred_t))
            list_gt_tt_iou.append(iou)
        # loop across pred (gtvn) ccs
        for i_pred_n in range(N_pred_n):
            cc_pred_n = ccs_pred_n==(i_pred_n+1)
            # Is there a gt cc (gtvn) that is IOU<0.5?
            iou=0.
            for i_gt_n in range(N_gt_n):
                cc_gt_n = ccs_gt_n==(i_gt_n+1)
                iou = np.maximum(iou,dice(cc_pred_n,cc_gt_n))
            list_pred_nn_iou.append(iou)
            # Is there a gt cc (gtvt) that is IOU<0.5?
            iou=0.
            for i_gt_t in range(N_gt_t):
                cc_gt_t = ccs_gt_t==(i_gt_t+1)
                iou = np.maximum(iou,dice(cc_pred_n,cc_gt_t))
            list_pred_nt_iou.append(iou)
        # loop across pred (gtvt) ccs
        for i_pred_t in range(N_pred_t):
            cc_pred_t = ccs_pred_t==(i_pred_t+1)
            # Is there a gt cc (gtvn) that is IOU<0.5?iou=0.
            iou=0.
            for i_gt_n in range(N_gt_n):
                cc_gt_n = ccs_gt_n==(i_gt_n+1)
                iou = np.maximum(iou,dice(cc_pred_t,cc_gt_n))
            list_pred_tn_iou.append(iou)
            # Is there a pred cc (gtvn) that is IOU<0.5?
            iou=0.
            for i_gt_t in range(N_gt_t):
                cc_gt_t = ccs_gt_t==(i_gt_t+1)
                iou = np.maximum(iou,dice(cc_pred_t,cc_gt_t))
            list_pred_tt_iou.append(iou)
    print('GT gtvn, predicted as gtvn:',np.sum(np.asarray(list_gt_nn_iou)>=0.5),' out of ', len(list_gt_nn_iou))
    print('GT gtvn, predicted as gtvt:',np.sum(np.asarray(list_gt_nt_iou)>=0.5),' out of ', len(list_gt_nt_iou))
    print('GT gtvt, predicted as gtvt:',np.sum(np.asarray(list_gt_tt_iou)>=0.5),' out of ', len(list_gt_tt_iou))
    print('GT gtvt, predicted as gtvn:',np.sum(np.asarray(list_gt_tn_iou)>=0.5),' out of ', len(list_gt_tn_iou))

    print('pred gtvn, gt as gtvn:',np.sum(np.asarray(list_pred_nn_iou)>=0.5),' out of ', len(list_pred_nn_iou))
    print('pred gtvn, gt as gtvt:',np.sum(np.asarray(list_pred_nt_iou)>=0.5),' out of ', len(list_pred_nt_iou))
    print('pred gtvt, gt as gtvt:',np.sum(np.asarray(list_pred_tt_iou)>=0.5),' out of ', len(list_pred_tt_iou))
    print('pred gtvt, gt as gtvn:',np.sum(np.asarray(list_pred_tn_iou)>=0.5),' out of ', len(list_pred_tn_iou))

    #print('list of dscs: ',list_gt_tt_iou, 'mean: ',np.mean(np.asarray(list_gt_tt_iou)))
    print('number of cc pred gtvn: ',list_N_pred_n)
    print('number of cc pred gtvt: ',list_N_pred_t)
    import pdb; pdb.set_trace()

    return None