import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


list_angle = []
acorr_data = [1]

#角度計算の関する
def angle(a,b,c):
  vec_a = a - b
  vec_c = c - b
  length_vec_a = np.linalg.norm(vec_a)
  length_vec_c = np.linalg.norm(vec_c)
  inner_product = np.inner(vec_a, vec_c)
  cos = inner_product / (length_vec_a * length_vec_c)
  
  rad = np.arccos(cos)    # 角度（ラジアン）の計算
  # 弧度法から度数法（rad ➔ 度）への変換
  degree = np.rad2deg(rad)
  return degree

#自己相関の求め方
def autocorrelation(data,k):
  #yの平均
  y_avg = np.mean(data
  #分子の計算
  sum_of_covariance = 0
  for i in range(k+1,len(data)):
    covariance = (data[i] - y_avg) * (data[i-(k+1)] - y_avg)
    sum_of_covariance += covariance
  #分母の計算
  sum_of_denominator = 0
  for u in range(len(data)):
    denominator = ( data[u] - y_avg )**2
    sum_of_denominator += denominator

  return sum_of_covariance / sum_of_denominator



if __name__ == '__main__':
  
  #1個目の波形--------------------------------------
  for i in range(40):
    filename = 'test_' + str(i).zfill(12) + '_keypoints.json'
    file = pd.read_json(filename)
    
    #右関節の座標
    x1 = file['people'][0]['pose_keypoints_2d'][15]
    y1 = file['people'][0]['pose_keypoints_2d'][16]
    x2 = file['people'][0]['pose_keypoints_2d'][18]
    y2 = file['people'][0]['pose_keypoints_2d'][19]
    x3 = file['people'][0]['pose_keypoints_2d'][21]
    y3 = file['people'][0]['pose_keypoints_2d'][22]
  
    #角度計算---------------------------------------
    a = np.array([x1,y1])
    b = np.array([x2,y2])
    c = np.array([x3,y3])
    list_angle.append(angle(a,b,c))
  
  #自己相関---------------------------------------
  for i in range(len(list_angle)-1):
    acorr_data.append(autocorrelation(list_angle,i))

  #グラフの表示-------------------------------------
  acorr_data = np.asarray(acorr_data)
  plt.stem(np.arange(len(list_angle)), acorr_data, use_line_collection=True)
  plt.show()
  
  

