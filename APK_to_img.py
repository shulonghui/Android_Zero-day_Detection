# coding: utf-8
import os
import time
import numpy as np
from PIL import Image
import binascii
import time
import math
from tqdm import tqdm
from multiprocessing import Pool



MALWARE_APK_TRAIN1 = 'E:\\CICDataset\\APKS\\Adware'
MALWARE_IMG_TRAIN1 = 'F:\\malwareimg\\Adware'

MALWARE_APK_TRAIN2 = 'E:\\CICDataset\\APKS\\Banking\\Banking'
MALWARE_IMG_TRAIN2 = 'F:\\malwareimg\\Banking'

MALWARE_APK_TRAIN3 = 'E:\\CICDataset\\APKS\\Riskware\\Riskware'
MALWARE_IMG_TRAIN3 = 'F:\\malwareimg\\Riskware'

MALWARE_APK_TRAIN4 = 'E:\\CICDataset\\APKS\\SMS\\SMS'
MALWARE_IMG_TRAIN4 = 'F:\\malwareimg\\SMS'

#def getMatrixfrom_bin(filename, width):
def getMatrixfrom_bin(filename):
    with open(filename, 'rb') as f:#以二进制格式打开一个文件用于只读。文件指针将会放在文件开头。
        content = f.read()
    hexst = binascii.hexlify(content)  # 将二进制文件转换为十六进制字符串
    fh = np.array([int(hexst[i:i+2], 16) for i in range(0, len(hexst), 2)])  # 按字节分割
    All = len(fh)
    wid = math.pow(All,1/2)
    wid = math.ceil(wid)
    rn = len(fh) // wid
    fh = np.reshape(fh[:rn * wid], (-1, wid))
    fh = np.uint8(fh)
    return fh

def APK_to_img_task(file, APK_path, img_path):
    APK_file = os.path.join(APK_path, file)
    img_file = os.path.join(img_path, file + '.jpg')
    print(img_file)
    if not os.path.exists(img_file):  # 判断是否已经生成过该 apk 文件对应的图片
        # 判断文件大小
        file_size = os.path.getsize(APK_file)
        file_size = file_size / 1024  # 转为 KB
        if file_size == 0:  # 有的文件可能为空
            return

        # 文件大小在合理范围内，执行以下语句
        fh = getMatrixfrom_bin(APK_file)
        im = Image.fromarray(fh)  # 转换为图像
        im.save(img_file)

def get_img_files(APK_path, img_path):

    files = os.listdir(APK_path)
    print(len(files))

    p = Pool(11)
    for file in files:
        p.apply_async(APK_to_img_task, args=(file, APK_path, img_path,))
    p.close()
    p.join()

if __name__ == '__main__':

    start_time = time.time()

    get_img_files(MALWARE_APK_TRAIN1, MALWARE_IMG_TRAIN1)
    get_img_files(MALWARE_APK_TRAIN2, MALWARE_IMG_TRAIN2)
    get_img_files(MALWARE_APK_TRAIN3, MALWARE_IMG_TRAIN3)
    get_img_files(MALWARE_APK_TRAIN4, MALWARE_IMG_TRAIN4)


    end_time = time.time()
    duration = end_time - start_time
    print(f"函数执行时间为：{duration:.2f}秒")