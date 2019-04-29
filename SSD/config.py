scale = [0.2, 0.34, 0.48, 0.62, 0.76, 0.9, 1.04]
feature_map_size = [38, 19, 10, 5, 3, 1]
box_num = [4, 6, 6, 6, 4, 4]
path = '/home/user/Documents/dataset/VOC/VOC/VOCdevkit/VOC2007'
class_dict = {'aeroplane': 0,
              'bicycle': 1,
              'bird': 2,
              'boat': 3,
              'bottle': 4,
              'bus': 5,
              'car': 6,
              'cat': 7,
              'chair': 8,
              'cow': 9,
              'diningtable': 10,
              'dog': 11,
              'horse': 12,
              'motorbike': 13,
              'person': 14,
              'pottedplant': 15,
              'sheep': 16,
              'sofa': 17,
              'train': 18,
              'tvmonitor': 19,
              'back_ground': 20}
class_num = 20 + 1  # +1 for back ground
train = '/home/user/Documents/dataset/VOC/VOC/VOCdevkit/VOC2007/ImageSets/Main/train.txt'
val = '/home/user/Documents/dataset/VOC/VOC/VOCdevkit/VOC2007/ImageSets/Main/val.txt'
trainval = '/home/user/Documents/dataset/VOC/VOC/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'
