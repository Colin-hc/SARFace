import os
from PIL import Image

import cv2
import numpy as np

import torch
from torchvision.transforms import transforms
from torchvision import utils as vutils
from tqdm import tqdm
# 定义根目录和train目录路径
root_dir = 'F:/download/512/train_img'
label_dir = 'F:/download/512/train_label_r'
train_dir = root_dir
img_path_list = []
label_path_list = []
img_crop_path = []
label_crop_path = []


# 遍历train目录下的所有子文件夹
# detector = MTCNN(image_size=128, margin=0, post_process=False, device='cuda')
# fp = create_swiftformer_sr_model(SwiftFormer_depth['l1'], SwiftFormer_width['l1'], SwiftFormer_depth['l1'], SwiftFormer_width['l1'])
# fp.load_state_dict(torch.load('swiftl1_fp_128_only.pth', map_location=torch.device('cpu')))
# fp.eval()
#
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
# data_aug = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=mean, std=std)
# ])
#
# for subdir in sorted(os.listdir(train_dir)):
#     subdir_path = os.path.join(train_dir, subdir)
#
#     # 确保当前路径是一个目录
#     # if os.path.isdir(subdir_path):
#         # 遍历子文件夹中的所有图片文件
#     # for image_name in sorted(os.listdir(subdir_path)):
#     image_path = subdir_path
#     label_path = subdir_path.replace('train_img','train_label_r').replace('jpg', 'png')
#     image_crop_dir = subdir_path.replace('train_img','train_img_crop')
#     label_crop_dir = subdir_path.replace('train_img','train_label_crop').replace('jpg', 'png')
#     img_path_list.append(image_path)
#     label_path_list.append(label_path)
#     img_crop_path.append(image_crop_dir)
#     label_crop_path.append(label_crop_dir)
#
#         # target_parsing_list.append(target_parsing)
#         # target_pcolor_list.append(target_pcolor)
#         # 检查文件是否是图片（可选）


# def face_split(input):
#
#     # input = torch.cat([input, input], dim=0)
#     batch_size, height, width = input.shape
#     # label_colors = {
#     #     0: (0, 0, 0),         # background
#     #     1: (255, 224, 189),   # skin
#     #     2: (255, 0, 0),       # nose
#     #     3: (0, 255, 0),       # eye_g
#     #     4: (0, 0, 255),       # l_eye
#     #     5: (255, 255, 0),     # r_eye
#     #     6: (255, 165, 0),     # l_brow
#     #     7: (255, 105, 180),   # r_brow
#     #     8: (0, 255, 255),     # l_ear
#     #     9: (255, 20, 147),    # r_ear
#     #     10: (75, 0, 130),     # mouth
#     #     11: (238, 130, 238),  # u_lip
#     #     12: (139, 69, 19),    # l_lip
#     #     13: (128, 0, 128),    # hair
#     #     14: (0, 128, 128),    # hat
#     #     15: (0, 0, 128),      # ear_r
#     #     16: (128, 128, 0),    # neck_l
#     #     17: (128, 128, 128),  # neck
#     #     18: (192, 192, 192),  # cloth
#     #     19: (0, 255, 127),    # top-left skin pixels relative to nose center
#     #     20: (255, 0, 255),    # top-right skin pixels relative to nose center
#     #     21: (0, 128, 0),      # bottom-left skin pixels relative to nose center
#     #     22: (255, 255, 0),    # bottom-right skin pixels relative to nose center
#     # }
#     nose_centers = torch.zeros((batch_size, 2), dtype=torch.long).to(input.device)
#     for i in range(batch_size):
#         parsed_image = input[i]
#         # Find indices of nose pixels
#         nose_pixels = (parsed_image == 2).nonzero(as_tuple=False)
#
#         if len(nose_pixels) > 0:
#             nose_center = nose_pixels.float().mean(dim=0).long()
#         else:
#             # If no nose pixels are found, find the center of the skin pixels
#             skin_pixels = (parsed_image == 1).nonzero(as_tuple=False)
#             if len(skin_pixels) > 0:
#                 nose_center = skin_pixels.float().mean(dim=0).long()
#             else:
#                 # Default to image center if no skin pixels are found
#                 nose_center = torch.tensor([height // 2, width // 2], dtype=torch.long)
#         nose_centers[i] = nose_center
#
#     # Create coordinate grids
#     y_grid = torch.arange(height).view(1, -1, 1).expand(batch_size, -1, width).to(input.device)
#     x_grid = torch.arange(width).view(1, 1, -1).expand(batch_size, height, -1).to(input.device)
#
#     # Broadcast nose centers to grid shape
#     center_y_grid = nose_centers[:, 0].view(-1, 1, 1).expand(-1, height, width)
#     center_x_grid = nose_centers[:, 1].view(-1, 1, 1).expand(-1, height, width)
#
#     # Create masks for each quadrant
#     top_left_mask = (input == 1) & (y_grid <= center_y_grid) & (x_grid < center_x_grid)
#     top_right_mask = (input == 1) & (y_grid < center_y_grid) & (x_grid >= center_x_grid)
#     bottom_left_mask = (input == 1) & (y_grid > center_y_grid) & (x_grid <= center_x_grid)
#     bottom_right_mask = (input == 1) & (y_grid >= center_y_grid) & (x_grid > center_x_grid)
#
#     # Update pixel values based on quadrant
#     input[top_left_mask] = 19
#     input[top_right_mask] = 20
#     input[bottom_left_mask] = 21
#     input[bottom_right_mask] = 22
#     # input_0 = input[0].cpu().numpy()
#     # color_image = np.zeros((height, width, 3), dtype=np.uint8)
#     # for label, color in label_colors.items():
#     #     mask = input_0 == label
#     #     color_image[mask] = color
#     #
#     # # Convert the numpy array to an image
#     # colored_image = Image.fromarray(color_image)
#     # colored_image.save("output_image.png")
#     return input


def face_split(input):
    # 获取图像的大小
    batch_size, height, width = input.shape
    # Step 1: 合并嘴和唇部（10：嘴，11：上唇，12：下唇）
    mouth_mask = (input == 10) | (input == 11) | (input == 12)
    input[mouth_mask] = 10  # 将嘴、上唇、下唇统一设为10

    # Step 2: 将嘴（像素值10）以后的像素值向前移动
    for i in range(13, 19):  # 因为我们知道最大像素值是18
        input[input == i] = i - 2

    # 获取当前分割图像的最大像素值
    max_pixel = 16

    # 初始化鼻子中心坐标
    nose_centers = torch.zeros((batch_size, 2), dtype=torch.long).to(input.device)
    skin_centers = torch.zeros((batch_size, 2), dtype=torch.long).to(input.device)
    # mouth_centers = torch.zeros((batch_size, 2), dtype=torch.long).to(input.device)
    for i in range(batch_size):
        parsed_image = input[i]
        # 找到鼻子像素的索引
        nose_pixels = (parsed_image == 2).nonzero(as_tuple=False)
        skin_pixels = (parsed_image == 1).nonzero(as_tuple=False)
        if len(skin_pixels) > 0:
            skin_center = skin_pixels.float().mean(dim=0).long()
        else:
            # 如果没有皮肤像素，则默认图像中心为鼻子中心
            skin_center = torch.tensor([height // 2, width // 2], dtype=torch.long)

        if len(nose_pixels) > 0:
            nose_center = nose_pixels.float().mean(dim=0).long()
        else:
            # 如果没有找到鼻子像素，查找皮肤像素的中心
            nose_center = skin_center
        nose_centers[i] = nose_center
        skin_centers[i] = skin_center

    # 创建坐标网格
    y_grid = torch.arange(height).view(1, -1, 1).expand(batch_size, -1, width).to(input.device)
    x_grid = torch.arange(width).view(1, 1, -1).expand(batch_size, height, -1).to(input.device)

    # Step 3: 背景象限划分（以皮肤中心为基准）
    center_y_grid = skin_centers[:, 0].view(-1, 1, 1).expand(-1, height, width)
    center_x_grid = skin_centers[:, 1].view(-1, 1, 1).expand(-1, height, width)

    # 背景象限的掩码
    bg_top_left_mask = (input == 0) & (y_grid <= center_y_grid) & (x_grid < center_x_grid)
    bg_top_right_mask = (input == 0) & (y_grid < center_y_grid) & (x_grid >= center_x_grid)
    bg_bottom_left_mask = (input == 0) & (y_grid > center_y_grid) & (x_grid <= center_x_grid)
    bg_bottom_right_mask = (input == 0) & (y_grid >= center_y_grid) & (x_grid > center_x_grid)

    # 动态给背景象限赋值，从 max_pixel+1 到 max_pixel+4
    input[bg_top_left_mask] = max_pixel + 1
    input[bg_top_right_mask] = max_pixel + 2
    input[bg_bottom_left_mask] = max_pixel + 3
    input[bg_bottom_right_mask] = max_pixel + 4

    # Step 4: 分割皮肤的四象限
    top_left_mask = (input == 1) & (y_grid <= center_y_grid) & (x_grid < center_x_grid)
    top_right_mask = (input == 1) & (y_grid < center_y_grid) & (x_grid >= center_x_grid)
    bottom_left_mask = (input == 1) & (y_grid > center_y_grid) & (x_grid <= center_x_grid)
    bottom_right_mask = (input == 1) & (y_grid >= center_y_grid) & (x_grid > center_x_grid)

    # 动态给皮肤象限赋值，从 max_pixel+5 到 max_pixel+8
    input[top_left_mask] = max_pixel + 5
    input[top_right_mask] = max_pixel + 6
    input[bottom_left_mask] = max_pixel + 7
    input[bottom_right_mask] = max_pixel + 8

    # Step 5: 删除头发像素（像素值为13）
    left_hair_mask = (input == 11) & (x_grid < center_x_grid)  # 左侧头发
    right_hair_mask = (input == 11) & (x_grid >= center_x_grid)
    input[left_hair_mask] = max_pixel + 9
    input[right_hair_mask] = max_pixel + 10
    # input[input == 13] = 0  # 删除所有头发像素

    # Step 6: 删除1：skin和11：hair前移
    for i in range(2, max_pixel + 11):  # max_pixel + 9 是头发之后的最大值
        if i < 11:
            input[input == i] = i - 2
        else:
            input[input == i] = i - 3

    # color_map = {
    #     0: [0, 0, 0],          # 鼻子
    #     1: [255, 224, 189],    # 眼镜绿色
    #     2: [255, 0, 0],        # 左眼
    #     3: [0, 255, 0],        # 右眼
    #     4: [0, 0, 255],        # 左眉
    #     5: [255, 255, 0],      # 右眉
    #     6: [255, 165, 0],      # 左耳
    #     7: [255, 105, 180],    # 右耳
    #     8: [0, 255, 255],      # 嘴
    #     9: [255, 20, 147],     # 帽子
    #     10: [75, 0, 130],      # 耳环
    #     11: [238, 130, 238],   # 项链
    #     12: [139, 69, 19],     # 颈部
    #     13: [128, 0, 128],     # 衣服
    #     14: [0, 128, 128],     # 背景象限1
    #     15: [0, 0, 128],       # 背景象限2
    #     16: [128, 128, 0],     # 背景象限3
    #     17: [128, 128, 128],   # 背景象限4
    #     18: [192, 192, 192],   # 皮肤象限1
    #     19: [255, 0, 255],     # 皮肤象限2
    #     20: [255, 255, 255],   # 皮肤象限3
    #     21: [0, 128, 0],       # 皮肤象限4
    #     22: [255, 0, 128],     # 头发左
    #     23: [255, 255, 255]    # 头发右
    # }
    #
    # # 创建一个RGB图像
    # output_image = np.zeros((input.shape[1], input.shape[2], 3), dtype=np.uint8)
    # input = input[0].numpy()
    # # 为每个像素应用颜色
    # for pixel_value, color in color_map.items():
    #     output_image[input == pixel_value] = color

    return input

def detect_faces_and_crop_images(idx, detector):
    detector = detector
    img_path = img_path_list[idx]
    try:
        img = Image.open(img_path).resize((512,512))
        label = Image.open(label_path_list[idx])
        faces, box = detector(img, label)
        if len(faces) == 0:
            print(f'No face detected: {img_path}')
            return

        # # crop face
        # tensor = np.transpose(faces.numpy(), (1, 2, 0))
        # # 步骤 2: 将 float32 转换为 uint8
        # tensor = tensor.astype(np.uint8)
        # # 步骤 3: 使用 PIL 保存图像
        # image = Image.fromarray(tensor)
        # # 生成图片路径的上一级目录
        label = box.squeeze().numpy()
        cv2.imwrite(label_crop_path[idx], label)

        vutils.save_image((faces/255.0).detach().cpu(), img_crop_path[idx])
        # Save the cropped image
        print(f'Cropped image saved: {img_crop_path[idx]}')

        # Parsing
        # img = data_aug(img)
        # img = img.unsqueeze(0)
        # pred = fp(img)
        # # pred = pred.squeeze().detach().cpu().numpy()
        # index = torch.argmax(pred, dim=1)
        # face_split(index, idx)
        return

    except Exception as e:
        print(f'Error processing {img_path}: {e}')
        return

# # start_idx = 1716000
# for idx, img_path in tqdm(enumerate(img_path_list[0:],start=0),
#                           total=len(img_path_list)-0):
#     detect_faces_and_crop_images(idx, detector)
