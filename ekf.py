import numpy as np
from numpy.random import multivariate_normal as multivariate_normal

class Q1_solution(object):

  """ 
  R: (n,2)covariance matrix R for observation noise.
  A: (n,2n) system matrix
  Q: (n,2n) covariance matrix Q for process noise.
  """
  def __init__(self):
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

  def EKF(self, observation):
    ob = observation
        
    ## PREDICTION ##
    #print(f"self.A shape: {self.A.shape}, self.mu shape: {self.mu.shape}")
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
