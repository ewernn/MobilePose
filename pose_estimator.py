'''
File: run_webcam.py
Project: MobilePose-PyTorch
File Created: Monday, 11th March 2019 12:47:30 am
Author: Yuliang Xiu (yuliangxiu@sjtu.edu.cn)
-----
Last Modified: Monday, 11th March 2019 12:48:49 am
Modified By: Yuliang Xiu (yuliangxiu@sjtu.edu.cn>)
-----
Copyright 2018 - 2019 Shanghai Jiao Tong University, Machine Vision and Intelligence Group
'''

import logging
import time

import os
import numpy as np

import torch
import torch.nn as nn
from torchvision import models

from estimator import ResEstimator
from network import CoordRegressionNetwork

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

class PoseEstimator(object):
    def __init__(self):
        model = 'mobilenetv2'
        inp_dim = 224

        # load the model 
        model_path = os.path.join("./models", model+"_%d.t7"%inp_dim)
        net = CoordRegressionNetwork(n_locations=16, backbone=model).to("cpu")
        self.e = ResEstimator(model_path, net, inp_dim)

        ### EKF ###
        n = 32  # number of coordinates to predict  (16 joints * 2 coordinates)
        self.A = np.hstack( [np.eye(n)*-1, np.eye(n)*2] )  # (n,2n)
        std_prediction_in_pixels = 2
        self.Q = np.eye(n) * std_prediction_in_pixels  # (n,n)
        std_measurement_in_pixels = 5
        self.R = np.eye(n) * std_measurement_in_pixels  # (n,n)
        self.x_last = np.zeros(n)
        self.mu = np.zeros(2*n, dtype=np.float32)  # mu_0
        self.sigma = np.eye(2*n)*0.01  # sigma_0

    @staticmethod
    def observation(state):
        return state

    @staticmethod
    def observation_state_jacobian(x):
        H = np.eye(32)
        return H
    
    # input: ob             shape=(36,1)
    # out:   ekfiltered ob  shape=(36,1)
    def EKF(self, ob):
            
        ## PREDICTION ##
        mu_bar_next = self.A @ self.mu  # (n,)
        sigma_bar_next = self.A @ self.sigma @ self.A.T + self.Q # (n, n)

        ## UPDATE ##
        H = self.observation_state_jacobian(mu_bar_next)  # (n, n)

        kalman_gain_numerator = sigma_bar_next @ H.T  # (n, n)
        kalman_gain_denominator = H @ sigma_bar_next @ H.T + self.R  # (n, n)
        kalman_gain = kalman_gain_numerator @ np.linalg.inv(kalman_gain_denominator)  # (n, n)
        
        expected_observation = self.observation(mu_bar_next)  # (n,)
        mu_next = mu_bar_next + kalman_gain @ (ob - expected_observation)  # (n,)
        sigma_next = sigma_bar_next - kalman_gain @ H @ sigma_bar_next  # (n, n)
        self.mu = np.concatenate((self.mu[32:], mu_next))  # (2n,)
        self.sigma[:32,:32] = self.sigma[32:,32:]     # (2n, 2n)
        self.sigma[32:,32:] = sigma_next
        
        return mu_next
    
    def crop_camera(image, ratio=0.15):
        #ratio=1.0
        height = image.shape[0]
        width = image.shape[1]
        mid_width = width / 2.0
        width_20 = width * ratio
        crop_img = image[0:int(height), int(mid_width - width_20):int(mid_width + width_20)]
        return crop_img

    def update(self, image):

        image = self.crop_camera(image)
        pose_guess = self.e.inference(image).flatten()  # (32,)
        pose = self.EKF(pose_guess).reshape((16,2)).astype(np.int64)

        return pose
