import scipy.io
import numpy as np
import matplotlib.pyplot as plt

import estimator
import network
import pose_estimator
import os
from PIL import Image
import tqdm
import cv2

vid_path = './pose_dataset/mpii/videos/'

train_joints_path = 'pose_dataset/mpii/train_joints.csv'
data = np.genfromtxt(train_joints_path,delimiter=',', dtype=None)
# make dict: pic_name -> frame
vid_pic_mapping = scipy.io.loadmat('/Users/ewern/Downloads/mpii_human_pose_v1_sequences_keyframes.mat')
annot_frame_paths = vid_pic_mapping['annolist_keyframes'][0]  # 24987 frames
#print(f"annot_frame_paths[0]: {annot_frame_paths[0]}")
mappings = {}
for map in annot_frame_paths:
    map = map[0][0][0][0][0]  # format: '037454012/00000053.jpg'
    [pic_name, annot_frame] = map.split('/')
    mappings[pic_name] = annot_frame
# make array of joint coords from tuple
def make_array_of_joint_coords(tup):
    coords = np.zeros(32)
    for i in range(32):
        coords[i] = tup[i+1]
    return coords
# calc avg error in pixel prediction
def calc_error(a, a_true):
    difference = np.absolute(a - a_true)
    return np.average(difference)
# add vid images to set
vids_path = './pose_dataset/mpii/videos/'
vid_dirs = set(os.listdir(vid_path))
# check how many of train_joints are in annot_frames
cnt=0
annotated_vids = set()
n_data = len(data)
#for i in tqdm(range(n_data)):
betterness = 0
for i in range(n_data):  # 18000
    pic_num = data[i][0].decode('UTF-8').split('.')[0]  # pic_num of video in train data
    if pic_num in vid_dirs:  # if pic_num of train data has a video
        cnt+=1
        #annotated_vids.add(data[i][0].decode('UTF-8').split('.')[0])
        annot_i = make_array_of_joint_coords(data[i]).reshape((16,2))  # I think this is the shape of annotations
        frame_from_vid = mappings[pic_num]
        # get folder for pic_num
        vid_path = vids_path + pic_num
        # get sorted list of images in folder
        images = sorted(os.listdir(vid_path))
        # find index of frame_from_vid in images
        frame_idx = images.index(frame_from_vid)
        num_frames_to_ekf = min(5, frame_idx)
        ekf = pose_estimator.PoseEstimator()
        for j in range(frame_idx-num_frames_to_ekf, frame_idx):
            img = cv2.imread(vid_path + '/' + images[j])
            # predict ekf
            _ = ekf.update(img)  # run it 5 times
        # predict ekf & non-ekf on annotated frame
        img_annot = cv2.imread(vid_path + '/' + frame_from_vid)
        ekf_pose = ekf.update(img_annot)
        non_ekf_pose = pose_estimator.PoseEstimator().update(img_annot, use_ekf=False)
        # compare accuracy
        # print(f"\n---\nekf_pose:\n{ekf_pose}\n---\nnon_ekf_pose:\n{non_ekf_pose}\n---\nannot_i:\n{annot_i}\n---\n")
        ekf_err = calc_error(ekf_pose, annot_i)
        non_ekf_err = calc_error(non_ekf_pose, annot_i)
        print(f"ekf error: {ekf_err}")
        print(f"non_ekf error: {non_ekf_err}")
        betterness += non_ekf_err - ekf_err
        print(f"current betterness: {betterness}")

        ### CAN MY PREDICTOR GIVE ME CONFIDENCE LEVELS? ###

        
    if i>1000: break

print(f"\n\n COUNT: {cnt}")

# a shape: (16,2)
# a_true shape: (16,2)






