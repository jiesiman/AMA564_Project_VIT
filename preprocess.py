import pandas as pd
import numpy as np
import cv2
import os
import random
from PIL import Image
from sklearn.model_selection import train_test_split

random.seed(456)


# 定义图像增强函数
def random_rotation(img):
    angle = random.randint(-30, 30)
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    img = cv2.warpAffine(img, M, (cols, rows))
    return img


def random_scale(img):
    scale = random.uniform(0.8, 1.2)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    return img


def gaussian_blur(img):
    return cv2.GaussianBlur(img, (5, 5), 0)


def color_perturbation(img):
    img = img.astype(np.float32)
    img += np.random.normal(0, 10, img.shape)
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def augment_image(img):
    img = random_rotation(img)
    img = random_scale(img)
    img = gaussian_blur(img)
    img = color_perturbation(img)
    # if random.random() < 0.3:  # 30%概率应用, 高龄特征增强
    #     # 添加皱纹噪声
    #     noise = np.random.randint(0, 50, img.shape[:2], dtype=np.uint8)
    #     mask = cv2.merge([noise, noise, noise])
    #     img = cv2.addWeighted(img, 0.8, mask, 0.2, 0)
    #
    #     # 局部模糊（模拟皮肤松弛）
    #     x, y = random.randint(0, 100), random.randint(0, 100)
    #     img[y:y + 50, x:x + 50] = cv2.GaussianBlur(img[y:y + 50, x:x + 50], (15, 15), 0)
    return img


def load_data_to_dataframe(data_dir):
    data = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".jpg"):
            try:
                # 文件名
                parts = filename.split('_')
                if len(parts) < 4:  # 如果分割后的部分不足4个，跳过该文件
                    print(f"跳过文件（命名不规范）: {filename}")
                    continue

                # 提取年龄、性别等信息
                age = int(parts[0])
                gender = int(parts[1])
                # ethnicity = int(parts[2])  # 如果需要种族信息，可以取消注释

                # 读取图像
                img_path = os.path.join(data_dir, filename)
                img = Image.open(img_path).convert('RGB')
                img = img.resize((224, 224))  # 调整图像尺寸
                img = np.array(img)  # 转换为numpy数组

                # 添加到数据列表
                data.append({
                    'age': age,
                    # 'ethnicity': ethnicity,  # 如果需要种族信息，可以取消注释
                    'gender': gender,
                    'img_name': filename,
                    'pixels': img
                })
            except (ValueError, IndexError) as e:
                # 如果解析失败，跳过该文件
                print(f"跳过文件（解析失败）: {filename}, 错误: {e}")
                continue

    # 创建DataFrame
    df = pd.DataFrame(data)
    return df


def augment_dataframe(df):
    augmented_data = []
    for idx, row in df.iterrows():
        img = row['pixels']
        img_augmented = augment_image(img)
        augmented_data.append(img_augmented)

    df_augmented = pd.DataFrame({
        'age': df['age'],
        # 'ethnicity': df['ethnicity'],
        'gender': df['gender'],
        'img_name': df['img_name'],
        'pixels': augmented_data
    })
    return df_augmented


if __name__ == '__main__':
    max_age = 116
    data_dir = os.path.join('./UTKFace')
    df = load_data_to_dataframe(data_dir)
    #df_augmented = augment_dataframe(df)
    # df_augmented.to_csv('augmented_data.csv', index=False)  # 保存为CSV文件

    #进行数据增强前先划分数据集
    # 指定划分比例
    train_ratio = 0.8  # 训练集比例
    val_ratio = 0.2    # 验证集比例

    # 划分数据集
    train_df, val_df = train_test_split(df, test_size=val_ratio, random_state=456)

    # 输出划分结果
    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(val_df)}")

    #仅对训练集进行数据增强
    df_augmented = augment_dataframe(train_df)
    # df_augmented.to_csv('augmented_data.csv', index=False)  # 保存为CSV文件

