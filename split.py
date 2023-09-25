'''
import os
import csv

apk_folder = r'E:\Androzoo\Benign1'
metadata_file = r'E:\Androzoo\metadata1.csv'

# 读取metadata.csv文件，将第一列和第二列分别存储到两个列表中
apk_names = []
apk_ids = []
with open(metadata_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        apk_names.append(row[0])
        apk_ids.append(row[1])

# 遍历apk_folder目录下的所有.apk文件
for file_name in os.listdir(apk_folder):
    if file_name.endswith('.apk'):
        # 切分文件名
        apk_name = file_name[:-4]
        # 如果切分后的文件名和metadata.csv表中第二列的值相同，则重命名文件
        if apk_name in apk_ids:
            index = apk_ids.index(apk_name)
            new_file_name = apk_names[index] + '.apk'
            old_file_path = os.path.join(apk_folder, file_name)
            new_file_path = os.path.join(apk_folder, new_file_name)
            os.rename(old_file_path, new_file_path)


import os
import csv
import shutil

# 读取metadata.csv文件，获取第一列的值
name_list = []
with open('E:\Androzoo\metadata1.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        name_list.append(row[0])

# 遍历E:\Androzoo\Benign目录下的所有文件
apk_dir = 'E:\Androzoo\Benign1'
target_dir = 'E:\Androzoo\Benig'
for filename in os.listdir(apk_dir):
    # 判断文件后缀名是否为.apk
    if filename.endswith('.apk'):
        # 切分文件名，获取前缀
        apk_name = filename.split('.apk')[0]
        # 判断前缀是否存在于name_list中
        if apk_name in name_list:
            # 获取对应的索引，并使用name_list中对应的值来对文件进行复制和重命名
            index = name_list.index(apk_name)
            new_name = name_list[index] + '.apk'
            shutil.copy(os.path.join(apk_dir, filename), os.path.join(target_dir, new_name))


import os
import shutil

apk_dir = 'E:\\Androzoo\\4440'
jpg_dir = 'D:\\new\\dataset\\train\\normal'
target_dir = 'E:\\Androzoo\\8'

# 获取apk_dir目录下的所有apk文件的前缀
apk_prefixes = set()
for filename in os.listdir(apk_dir):
    if filename.endswith('.apk'):
        prefix = filename.split('.apk')[0]
        apk_prefixes.add(prefix)

# 获取jpg_dir目录下的所有jpg文件的前缀
jpg_prefixes = set()
for filename in os.listdir(jpg_dir):
    if filename.endswith('.jpg'):
        prefix = filename.split('.jpg')[0]
        jpg_prefixes.add(prefix)

# 查找不在jpg_prefixes中的apk前缀，并将对应的apk文件复制到target_dir目录下
for filename in os.listdir(apk_dir):
    if filename.endswith('.apk'):
        prefix = filename.split('.apk')[0]
        if prefix not in jpg_prefixes:
            shutil.copy(os.path.join(apk_dir, filename), os.path.join(target_dir, filename))

import os
import random
import shutil

# 设置原始文件夹和目标文件夹
source_dir = r'D:\new\dataset\5560'
train_dir = r'D:\new\dataset\train\normal'
val_dir = r'D:\new\dataset\val\normal'

# 获取原始文件夹中所有文件的路径
file_paths = [os.path.join(source_dir, f) for f in os.listdir(source_dir)]

# 设置要选择的文件数量
num_train_files = 4448
num_val_files = 1112

# 随机选择要用于训练和验证的文件
train_files = random.sample(file_paths, num_train_files)
val_files = random.sample(set(file_paths) - set(train_files), num_val_files)

# 将选择的文件剪切到相应的目标文件夹中
for file_path in train_files:
    shutil.move(file_path, train_dir)

for file_path in val_files:
    shutil.move(file_path, val_dir)


import os
import shutil

tra_dir = r'E:\CICDataset\Images\normaltra'
val_dir = r'E:\CICDataset\Images\normalval'
dst_dir = r'E:\CICDataset\Images\4000'

tra_files = os.listdir(tra_dir)
val_files = os.listdir(val_dir)

for file in tra_files:
    if file not in val_files:
        src_path = os.path.join(tra_dir, file)
        dst_path = os.path.join(dst_dir, file)
        shutil.copy(src_path, dst_path)


import os

#dir_path = "D:/new/dataset/train/malware"
dir_path = "D:/new/dataset/val/normal"
#dir_path = "D:/new/dataset/val/malware"

# 获取目录下所有文件名
file_names = os.listdir(dir_path)

# 初始化最大和最小文件大小
max_size = 0
min_size = float('inf')

# 遍历所有文件，统计最大和最小文件大小
for file_name in file_names:
    file_path = os.path.join(dir_path, file_name)
    if os.path.isfile(file_path):
        file_size = os.path.getsize(file_path) / (1024 * 1024) # 将文件大小转换为兆字节（MB）
        if file_size > max_size:
            max_size = file_size
        if file_size < min_size:
            min_size = file_size

print("最大文件大小为：{:.2f} MB".format(max_size))
print("最小文件大小为：{:.2f} MB".format(min_size))


import os
from PIL import Image

folder_path = 'D:/new/dataset/train/malware'
image_extensions = ['.jpg', '.jpeg', '.png']  # 可能的图片文件扩展名

# 初始化最大和最小分辨率
min_width = float('inf')
min_height = float('inf')
max_width = 0
max_height = 0

# 遍历文件夹中的所有图片文件
for filename in os.listdir(folder_path):
    if any(ext in filename.lower() for ext in image_extensions):
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path)
        width, height = image.size

        # 更新最大和最小分辨率
        if width < min_width:
            min_width = width
        if height < min_height:
            min_height = height
        if width > max_width:
            max_width = width
        if height > max_height:
            max_height = height

# 打印结果
print(f"最小分辨率: {min_width} x {min_height}")
print(f"最大分辨率: {max_width} x {max_height}")



import os
from PIL import Image

folder_path = 'D:/new/dataset'
image_extensions = ['.jpg', '.jpeg', '.png']  # 可能的图片文件扩展名

total_width = 0
total_height = 0
num_images = 0

# 递归遍历文件夹及其子文件夹
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if any(ext in file.lower() for ext in image_extensions):
            image_path = os.path.join(root, file)
            image = Image.open(image_path)
            width, height = image.size

            total_width += width
            total_height += height
            num_images += 1

# 计算平均分辨率
average_width = total_width / num_images
average_height = total_height / num_images

# 打印结果
print(f"平均分辨率: {average_width} x {average_height}")

import os
import numpy as np
import cv2
import torchvision.transforms as transforms

def load_and_transform_images(image_paths, target_size):
    images = []
    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, target_size)
        images.append(image)

    images = np.stack(images, axis=0)  # 将图像堆叠成形状为(batch_size, height, width)的张量
    images = images.astype(np.float32) / 255.0  # 将像素值缩放到0到1的范围内
    return images

def compute_mean_std(images):
    mean = np.mean(images)
    std = np.std(images)
    return mean, std

#data_dir = "D:/new/dataset/train"  # 数据存储的目录
data_dir = "./dataset/train"  # 数据存储的目录
class1_dir = os.path.join(data_dir, "malware")  # 类别1图像存储的目录
class2_dir = os.path.join(data_dir, "normal")  # 类别2图像存储的目录
target_size = (224, 224)  # 调整图像的目标尺寸
batch_size = 128  # 每批加载的图像数量

# 获取类别1和类别2的图像路径
class1_image_paths = [os.path.join(class1_dir, filename) for filename in os.listdir(class1_dir)]
class2_image_paths = [os.path.join(class2_dir, filename) for filename in os.listdir(class2_dir)]

# 合并类别1和类别2的图像路径
image_paths = class1_image_paths + class2_image_paths

mean_list = []
std_list = []

# 使用批处理方式加载图像并计算均值和标准差
for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i:i+batch_size]
    batch_images = load_and_transform_images(batch_paths, target_size)
    mean, std = compute_mean_std(batch_images)
    mean_list.append(mean)
    std_list.append(std)

mean = np.mean(mean_list)
std = np.mean(std_list)

print("Mean:", mean)
print("Std:", std)

import os
import csv

# 设置文件路径
image_directory = r"D:\new\dataset\val\malware"
csv_file = r"F:\sha256_family.csv"

# 读取CSV文件中的内容
families = {}
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        filename = row[0]
        family = row[1]
        families[filename] = family

# 提取文件名和对应值
values = set()
for filename in os.listdir(image_directory):
    if filename.endswith(".jpg"):
        file_name = os.path.splitext(filename)[0]
        if file_name in families:
            value = families[file_name]
            values.add(value)

# 输出结果和统计信息
for value in values:
    print(value)

print("共有{}种不重复的值".format(len(values)))




import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# 定义真实标签和预测标签
y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
y_pred = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)

# 定义标签名称
labels = ['class 0', 'class 1', 'class 2']

# 绘制混淆矩阵
fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=labels, yticklabels=labels,
       title='Confusion matrix',
       ylabel='True label',
       xlabel='Predicted label')

# 添加数字标签
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='white')

fig.colorbar(im)
plt.show()

import os
import random
import shutil

# 定义原始数据目录和目标数据目录
source_dir = "D:/new/dataset/train/malware"
target_dir = "D:/new/data/train/malware"

# 定义要移动的文件比例
move_percentage = 0.01  # 移动 1% 的文件

# 获取原始数据目录中的所有文件列表
file_list = os.listdir(source_dir)

# 计算要移动的文件数量
num_files_to_move = int(len(file_list) * move_percentage)

# 随机选择要移动的文件
files_to_move = random.sample(file_list, num_files_to_move)

# 移动文件
for file_name in files_to_move:
    source_file_path = os.path.join(source_dir, file_name)
    target_file_path = os.path.join(target_dir, file_name)
    shutil.move(source_file_path, target_file_path)

print("文件移动完成！")

'''
import os
import random
import shutil

# 定义原始数据目录和目标数据目录
source_dir = "D:/new/dataset/val/malware"
target_dir = "D:/new/data/val/malware"

#source_dir = "D:/new/dataset/val/normal"
#target_dir = "D:/new/data/val/normal"

# 定义要复制的文件比例
copy_percentage = 0.01  # 复制 1% 的文件

# 获取原始数据目录中的所有文件列表
file_list = os.listdir(source_dir)

# 计算要复制的文件数量
num_files_to_copy = int(len(file_list) * copy_percentage)

# 随机选择要复制的文件
files_to_copy = random.sample(file_list, num_files_to_copy)

# 复制文件
for file_name in files_to_copy:
    source_file_path = os.path.join(source_dir, file_name)
    target_file_path = os.path.join(target_dir, file_name)
    shutil.copy(source_file_path, target_file_path)

print("文件复制完成！")





