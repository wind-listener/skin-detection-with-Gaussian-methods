import os
import cv2
from tqdm import tqdm

def imageProcess(PATH_TO_ORIGINAL_IMAGES_DIR,PATH_TO_PROCESSED_IMAGES_DIR):
    fileList = os.listdir(PATH_TO_ORIGINAL_IMAGES_DIR)
    fileList.sort(key=lambda x:int(x[5:-5]))
    print("总共有 %d 张图片" % len(fileList))
    print("开始修改尺寸！")
    for image in tqdm(fileList, desc='图片尺寸修改中：'):
        img = cv2.imread(PATH_TO_ORIGINAL_IMAGES_DIR + "/" + image)
        newImage = cv2.resize(img, (140, 140), interpolation=cv2.INTER_AREA)
        cv2.imwrite(PATH_TO_PROCESSED_IMAGES_DIR + "/" + image, newImage)
