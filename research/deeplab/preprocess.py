import os
import shutil
import argparse
import glob
from random import randrange
import cv2
import numpy as np
from skimage import morphology
import pandas as pd
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Face Recognition parameters')
parser.add_argument(
    '--fullpath_origin',
    metavar='SRC',
    help='Full path of original images')

parser.add_argument(
    '--fullpath_JPEGDestination',
    metavar='JPG',
    help='Full path of original images')

parser.add_argument(
    '--fullpath_destination',
    metavar='DEST',
    help='Full path of original images')

parser.add_argument(
    '--fullpath_train_val_list',
    metavar='SPLITS',
    help='Full path of train and val lists')

def save_train_val_list(files_list):
    df = pd.DataFrame(files_list)
    train, val = train_test_split(df, test_size=0.30, random_state=42)
    df.to_csv(f'{args.fullpath_train_val_list}/trainval.txt', header=None, index=False)
    train.to_csv(f'{args.fullpath_train_val_list}/train.txt', header=None, index=False)
    val.to_csv(f'{args.fullpath_train_val_list}/val.txt', header=None, index=False)

if __name__ == '__main__':
    args = parser.parse_args()

    shutil.rmtree(args.fullpath_destination, ignore_errors=True, onerror=None)
    os.makedirs(args.fullpath_destination)

    shutil.rmtree(args.fullpath_JPEGDestination, ignore_errors=True, onerror=None)
    os.makedirs(args.fullpath_JPEGDestination)

    shutil.rmtree(args.fullpath_train_val_list, ignore_errors=True, onerror=None)
    os.makedirs(args.fullpath_train_val_list)

    files = [f for f in glob.glob(f'{args.fullpath_origin}/*prime.tif', recursive=True)]

    for idx, f in enumerate(files):
        img_original = cv2.imread(f)
        target_filename = f.replace('prime', '-' + str(randrange(1, 6)))
        img_target = cv2.imread(target_filename)

        # Image Difference
        img_diff = img_target - img_original

        img_pb = img_diff.sum(2) # 3 channel to 1

        img_pb[img_pb > 0] = 255

        img_mask = img_pb == 255

        # Fill Area
        bg_pb = img_original.sum(2) # 3 channel to 1
        bg_mask = bg_pb < 50

        disk_mask = morphology.remove_small_holes(img_mask, img_mask.shape[0] * img_mask.shape[1] * 0.2) #Fill hole smaller than 20% os image

        cup_mask = morphology.remove_small_holes(img_mask, np.sum(disk_mask) * 0.45)
        cup_mask = morphology.remove_small_objects(cup_mask, np.sum(disk_mask) * 0.1)

        rim_mask = disk_mask ^ cup_mask #Disk less Cup using XOR operator

        eye_mask = np.invert(bg_mask) ^ disk_mask #Eye less Disk using XOR operator

        '''
        Stack RGB channels of each label follow this map:
        Background: (0,0,0)
        Eye: (128,0,0)
        Rim: (0,128,0)
        Cup: (128,128,0)
        '''

        img_segmentation = np.zeros(img_original.shape) # All is Background so far
        img_segmentation[:,:,0] = eye_mask * 128 # Fill image with eye segmentation RGB=(128,0,0)
        img_segmentation[:,:,1] = rim_mask * 128 # Fill image with rim segmentation RGB=(0,128,0)
        img_segmentation[:,:,0] = cup_mask * 128 # Fill image with cup segmentation RGB=(128,128,0)
        img_segmentation[:,:,1] = cup_mask * 128 # Fill image with cup segmentation RGB=(128,128,0)

        #img_segmentation = np.stack((eye, rim, cup), axis=2)
        #img_segmentation = (eye + rim + cup)

        ori_filename = target_filename.split('/')[-1].replace('tif', 'jpg')
        seg_filename = target_filename.split('/')[-1].replace('tif', 'png')

        if (np.sum(cup_mask) > 0) & (np.sum(rim_mask) > 0):
            cv2.imwrite(f'{args.fullpath_JPEGDestination}/{ori_filename}', img_original)
            cv2.imwrite(f'{args.fullpath_destination}/{seg_filename}', img_segmentation)
        else:
            print('Error on ' + f)

        if idx % 50 == 0:
            print('converted', idx)

    # Save train and val lists
    files = [f for f in glob.glob(f'{args.fullpath_destination}/*.png')]
    files_names = lambda files: files.split('/')[-1].split('.')[0]
    save_train_val_list(list(map(files_names, files)))
