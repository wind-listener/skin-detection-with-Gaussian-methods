# -*- coding: utf-8 -*-
import os
import cv2
import matplotlib.pyplot as plt
from gmm import *
from plt_multiGaussian import pltMG

# 设置调试模式
DEBUG = True

DATASET_DIR = 'resized_image/mask'
fileList = os.listdir(DATASET_DIR)
fileList.sort(key=lambda x: int(x[5:-5]))
print("总共有 %d 张图片" % len(fileList))
# create trian and test set
trainSet = []
for i in range(70):
    randIndex = int(np.random.uniform(0, len(fileList)))
    trainSet.append(fileList[randIndex])
    del (fileList[randIndex])
testSet = fileList
print('Number of skin training data = %d' % (len(trainSet)))
print('Number of skin Testing data = %d' % (len(testSet)))

SkinTrain = np.zeros((0, 2))
for trainSample in trainSet:
    img = cv2.imread('resized_image/mask/' + trainSample)
    U_Matrix = img[:, :, 1].flatten()
    V_Matrix = img[:, :, 2].flatten()
    ColorArray = np.vstack((U_Matrix, V_Matrix)).transpose()
    SkinTrain = np.vstack((SkinTrain, ColorArray))
# 剔除零元素
mask = SkinTrain != np.array(0)
SkinTrain = SkinTrain[mask[:, 0], :]
# 测试集
SkinTest = np.zeros((0, 2))
for testSample in testSet:
    img = cv2.imread('resized_image/img/' + testSample)
    U_Matrix = img[:, :, 1].flatten()
    V_Matrix = img[:, :, 2].flatten()
    ColorArray = np.vstack((U_Matrix, V_Matrix)).transpose()
    SkinTest = np.vstack((SkinTest, ColorArray))
# 测试集的真实分类
SkinTestMask = np.zeros((0, 2))
for testSample in testSet:
    img = cv2.imread('resized_image/mask/' + testSample)
    U_Matrix = img[:, :, 1].flatten()
    V_Matrix = img[:, :, 2].flatten()
    ColorArray = np.vstack((U_Matrix, V_Matrix)).transpose()
    SkinTestMask = np.vstack((SkinTestMask, ColorArray))
TrueCategory = SkinTestMask != np.array(0)
TrueCategory = TrueCategory[:, 0]

# 载入数据
Y = SkinTrain
matY = np.matrix(Y, copy=True)
matTest = np.matrix(SkinTest, copy=True)
# 模型个数，即聚类的类别个数
K = 5

# 计算 GMM 模型参数
mu, cov, alpha = GMM_EM(matY, K, 10)
# 绘制 GMM 模型等高图##########################
pltMG(mu, cov, alpha)

# # 根据 GMM 模型，对测试数据进行判断
# N = matTest.shape[0]
# # 求当前模型参数下，各模型对样本的响应度矩阵
# gamma = getExpectation(matTest, mu, cov, alpha)
# # 对每个样本，求响应度最大的模型下标，作为其类别标识
# category = np.array(gamma.argmax(axis=1).flatten().tolist()[0])
# # 根据category就可以计算准确率了！
# P = np.sum(category[TrueCategory]) / len(category[TrueCategory])
# R = np.sum(category[TrueCategory]) / sum(category)+0.001
# print("查准率：%f  查全率：%f"%(P, R))
# class1 = np.array([matTest[i] for i in range(N) if category[i] == 0])
# class2 = np.array([matTest[i] for i in range(N) if category[i] == 1])

# # 绘制聚类结果
# plt.plot(class1[:, 0], class1[:, 1], 'rs', label="class1")
# plt.plot(class2[:, 0], class2[:, 1], 'bo', label="class2")
# plt.legend(loc="best")
# plt.title("GMM Clustering By EM Algorithm")
# plt.show()
