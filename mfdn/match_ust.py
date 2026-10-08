import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import cv2
import os
import torch
from PIL import Image
from tqdm import tqdm
from skimage.feature import match_descriptors
import scipy.io as scio
import argparse

blacklist_NIR = ['89.png', '87.png', '105.png', '129.png']

parser = argparse.ArgumentParser()
parser.add_argument("--feature_name", type=str, default='MFDN', help='Name of feature')
parser.add_argument("--subsets", type=str, default='UST_Campus',
                    help="UST_Campus, Tower_D, Facade_18")
parser.add_argument("--split", type=str, default='test', help='Dataset split to evaluate')
parser.add_argument("--vis-dir", type=str, default='VIS', help='Visible-image folder inside the split')
parser.add_argument("--ir-dir", type=str, default='IR', help='Infrared-image folder inside the split')
parser.add_argument("--identity-gt", action='store_true',
                    help='Use identity homography for already aligned image pairs')
parser.add_argument("--redfeat-gt", action='store_true',
                    help='Read ReDFeat .12/.21 transforms and normalize them to VIS-to-IR')
parser.add_argument("--nums_kp", type=int, default=-1, help="Number of feature for evluation")
parser.add_argument("--vis_flag", type=bool, default=True, help="Visualization flag")
args = parser.parse_args()

import argparse

bf = cv2.BFMatcher(crossCheck=True)

lm_counter = 0
MIN_MATCH_COUNT = 5
num_black_list = 0

subsets = [args.subsets]

if args.nums_kp < 0:
    nums_kp = [1024, 2048, 4096]
else:
    nums_kp = [args.nums_kp]
feature_name = args.feature_name
vis_flag = args.vis_flag

for subset in subsets:
    subset_path = os.path.join(SCRIPT_DIR, subset)
    dirlist = os.listdir(subset_path)
    if args.split in dirlist:
        imgs = os.listdir(os.path.join(subset_path, args.split, args.vis_dir))
    else:
        continue
    print(subset)
    filepath1 = os.path.join(subset_path, args.split, args.vis_dir)
    filepath2 = os.path.join(subset_path, args.split, args.ir_dir)
    for num in nums_kp:
        # for num in [2048]:
        N_k1 = []
        N_k2 = []
        N_corr = []
        N_corretmatches = []
        N_in_corretmatches = []
        N_k1_ol = []
        N_k2_ol = []
        N_corr_thres = []
        N_corretmatches_thres = []
        image_list = sorted(name for name in os.listdir(filepath1) if name.endswith('.png'))
        img_list_whitelist = []
        progress_bar = tqdm(range(len(image_list)))
        for id in progress_bar:
            # i=id+1

            imgpath1 = os.path.join(filepath1, image_list[id])
            imgpath2 = os.path.join(filepath2, image_list[id])
            image1 = np.array(Image.open(imgpath1).convert('RGB'))
            image2 = np.array(Image.open(imgpath2).convert('RGB'))
            ff = image_list[id].replace('.png', '.features.mat')
            feats = scio.loadmat(os.path.join(SCRIPT_DIR, 'features', subset, feature_name, ff))
            desc1 = feats['desc1']
            desc2 = feats['desc2']
            kp1 = feats['kp1'][:, 0:2]
            kp2 = feats['kp2'][:, 0:2]

            if args.identity_gt and args.redfeat_gt:
                raise ValueError("--identity-gt and --redfeat-gt are mutually exclusive")
            if args.identity_gt:
                H = np.eye(3, dtype=np.float64)
            elif args.redfeat_gt:
                transform_dir = os.path.join(subset_path, args.split, 'transforms')
                stem = os.path.splitext(image_list[id])[0]
                transform_21 = os.path.join(transform_dir, stem + '.21.mat')
                transform_12 = os.path.join(transform_dir, stem + '.12.mat')
                if os.path.exists(transform_21):
                    H = scio.loadmat(transform_21)['H']
                elif os.path.exists(transform_12):
                    # .12 warps VIS away from the aligned pair; evaluation needs
                    # the mapping from the current VIS image back to current IR.
                    H = np.linalg.inv(scio.loadmat(transform_12)['H'])
                else:
                    raise FileNotFoundError(
                        f"No ReDFeat transform found for {image_list[id]}")
            else:
                H = scio.loadmat(os.path.join(
                    subset_path, args.split, 'homo',
                    image_list[id].replace('.png', '.mat')))['H']
            ones = np.ones_like(image1)
            mask = cv2.warpPerspective(ones, H, [ones.shape[1], ones.shape[0]])
            mask_2 = mask > 0.5
            mask_2 = mask_2 * 1.0
            mask = cv2.warpPerspective(mask_2, np.linalg.inv(H), [mask.shape[1], mask.shape[0]])
            mask_1 = mask > 0.5
            ones = np.ones([np.size(kp1, 0), 1])
            kp_1_warped = np.hstack([kp1, ones])
            kp_1_warped = H @ kp_1_warped.transpose()
            kp_1_warped = kp_1_warped / kp_1_warped[2, :]
            kp_1_warped = kp_1_warped[0:2, :].transpose()
            kp_2_warped = kp2
            N_k1.append(kp1[0:num].shape[0])
            N_k2.append(kp2[0:num].shape[0])

            overlap1 = 0
            for kp in kp1[0:num]:
                x = int(kp[0] + 0.5)
                y = int(kp[1] + 0.5)
                if mask_1[(y, x)].sum(axis=-1) > 0.5:
                    overlap1 += 1
            N_k1_ol.append(overlap1)

            overlap2 = 0
            for kp in kp2[0:num]:
                x = int(kp[0] + 0.5)
                y = int(kp[1] + 0.5)
                if mask_2[((y, x))].sum(axis=-1) > 0.5:
                    overlap2 += 1
            N_k2_ol.append(overlap2)

            kp_1_warped_ = kp_1_warped[0:num][:, :2].reshape(-1, 1, 2)
            kp_2_warped_ = kp_2_warped[0:num][:, :2].reshape(1, -1, 2)
            dist_k = ((kp_1_warped_ - kp_2_warped_) ** 2).sum(axis=2)

            matches = match_descriptors(desc1[0:num], desc2[0:num], cross_check=True)
            keypoints_left = kp_1_warped[0:num][matches[:, 0], : 2]
            keypoints_left_raw = kp1[0:num][matches[:, 0], : 2]
            keypoints_right = kp_2_warped[0:num][matches[:, 1], : 2]
            keypoints_right_raw = kp2[0:num][matches[:, 1], : 2]

            dif = (keypoints_left - keypoints_right)
            dist_m = dif[:, 0] ** 2 + dif[:, 1] ** 2
            for thres in range(1, 11):
                #################################
                n_corr = ((dist_k <= thres ** 2).sum(axis=1) > 0.9).sum()
                N_corr_thres.append(n_corr.item())
                if thres == 3:
                    N_corr.append(n_corr.item())
                #################################

                inds = dist_m <= thres ** 2
                # print(inds.sum())
                N_corretmatches_thres.append(inds.sum())
                if thres == 3:
                    N_corretmatches.append(inds.sum())
        N_corr_thres = np.array(N_corr_thres)
        N_corr = np.array(N_corr) * 1.0
        N_k1 = np.array(N_k1) * 1.0
        N_k2 = np.array(N_k2) * 1.0
        N_k1_ol = np.array(N_k1_ol)
        N_k2_ol = np.array(N_k2_ol)
        N_corretmatches = np.array(N_corretmatches) * 1.0
        N_corretmatches_thres = np.array(N_corretmatches_thres)
        # N_in_corretmatches = np.array(N_in_corretmatches)
        RR = N_corr * 1.0 / np.array([N_k1, N_k2]).min(axis=0)
        mask_zero = N_k1_ol < 0.1
        N_k1_ol_temp = N_k1_ol
        N_k1_ol_temp[mask_zero] = 1
        mask_zero = N_k2_ol < 0.1
        N_k2_ol_temp = N_k2_ol
        N_k2_ol_temp[mask_zero] = 1
        MS = (N_corretmatches * 1.0 / N_k1_ol + N_corretmatches * 1.0 / N_k2_ol) / 2
        from scipy.io import savemat

        os.makedirs(os.path.join(SCRIPT_DIR, 'results', subset, feature_name), exist_ok=True)
        savemat(
            os.path.join(os.path.join(SCRIPT_DIR, 'results', subset, feature_name, 'match_result_{}.mat'.format(num))),
            {'N_corr': N_corr, 'N_k1': N_k1, 'N_k2': N_k2, 'N_correctmatches': N_corretmatches,
             'N_k1_ol': N_k1_ol, 'N_k2_ol': N_k2_ol, 'N_correctmatches_thres': N_corretmatches_thres,
             'N_corr_thres': N_corr_thres})
        print('Number of infrared keypoints: %f.' % np.mean(N_k1))
        print('Number of visible keypoints: %f.' % np.mean(N_k2))
        print('Number of correspondence: %f.' % N_corr.mean())
        print('Number of correct matches: %f.' % np.mean(N_corretmatches))
        # print('Number of inlier correct matches: %d.' % np.mean(N_in_corretmatches))
        print('RR: {}.'.format(RR.mean()))
        print('MS: {}.'.format(MS.mean()))

        log_file_path = os.path.join(SCRIPT_DIR, 'results', subset, feature_name, 'match_log.txt')
        if os.path.exists(log_file_path):
            print('[Warning] Log file already exists.')
        log_file = open(log_file_path, 'a+')
        log_file.write('Number of infrared keypoints: %f.\n' % np.mean(N_k1))
        log_file.write('Number of visible keypoints: %f.\n' % np.mean(N_k2))
        log_file.write('Number of correspondence: %f.\n' % N_corr.mean())
        log_file.write('Number of correct matches: %f.\n' % np.mean(N_corretmatches))
        # log_file.write('Number of inlier correct matches: %d.\n' % np.mean(N_in_corretmatches))
        log_file.write('RR: {}.\n'.format(RR.mean()))
        log_file.write('MS: {}.\n'.format(MS.mean()))
        log_file.close()
