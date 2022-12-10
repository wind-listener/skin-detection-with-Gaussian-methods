"""
Time:2022/11/15
Author:Zhang Zhiming
Student ID:2022141133
"""
from my_utils import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import multivariate_normal
from tqdm import tqdm

TP = 0
TN = 0
FP = 0
FN = 0
sk = []
nonsk = []

DATASET_DIR = 'resized_image/mask'
if __name__ == '__main__':
    imageProcess('original_dataset/img', 'resized_image/img')
    imageProcess('original_dataset/mask', 'resized_image/mask')

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
print('Number of skin Testing data = %d' % (len(trainSet)))

SkinTrain = np.zeros((0, 2))
for trainSample in trainSet:
    img = cv2.imread('resized_image/mask/' + trainSample)
    U_Matrix = img[:, :, 1].flatten()
    V_Matrix = img[:, :, 2].flatten()
    ColorArray = np.vstack((U_Matrix, V_Matrix)).transpose()
    SkinTrain = np.vstack((SkinTrain, ColorArray))

SkinTest = np.zeros((0, 2))
for testSample in testSet:
    img = cv2.imread('resized_image/mask/' + testSample)
    U_Matrix = img[:, :, 1].flatten()
    V_Matrix = img[:, :, 2].flatten()
    ColorArray = np.vstack((U_Matrix, V_Matrix)).transpose()
    SkinTest = np.vstack((SkinTest, ColorArray))


# Calculate the mean along all U,V
Cmean = SkinTrain.mean(axis=0)
print('Mean = ' + str(Cmean))
x = SkinTrain - Cmean
[M, N] = x.shape
CovMatrix = (1 / M) * np.dot(x.T, x)
print(CovMatrix)
x, y = np.mgrid[60:150:1, 135:180:1]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x
pos[:, :, 1] = y
rv = multivariate_normal(Cmean, CovMatrix)
allout = rv.pdf(pos)
plt.contourf(x, y, allout)

## Test
threshold = 0.0001
for i in SkinTest:
    if (rv.pdf(i) > threshold):
        TP += 1
    else:
        FN += 1

print('TP = %d' % (TP))
print('TP Rate = %f %% ' % ((TP / SkinTest.shape[0]) * 100))
print('FN Rate = %f %% ' % ((FN / SkinTest.shape[0]) * 100))

# NonSkinTest = np.array(nonsk)  # convert the list "a" to numpy array "c"
#
# for i in NonSkinTest:
#     if (rv.pdf(i) > threshold):
#         FP += 1
#     else:
#         TN += 1
#
# print('FP = %d' % (FP))
# print('FP Rate = %f %% ' % ((FP / NonSkinTest.shape[0]) * 100))
# print('TN Rate = %f %% ' % ((TN / NonSkinTest.shape[0]) * 100))

plt.show()
