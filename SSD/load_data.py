import cv2
import os
import config
import xml.etree.ElementTree as ET
import numpy as np

def cal_iou(a, b):
    """

    Args:
        a: bbox, [xmin, ymin, xmax, ymax]
        b: bbox

    Returns: IoU

    """
    # intersection
    ixmin = np.maximum(a[0], b[0])
    iymin = np.maximum(a[1], b[1])
    ixmax = np.minimum(a[2], b[2])
    iymax = np.minimum(a[3], b[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = (a[2] - a[0] + 1.) * (a[3] - a[1] + 1.) +\
          (b[2] - b[0] + 1.) * (b[3] - b[1] + 1.) -\
          inters

    overlaps = inters / uni
    return overlaps

def load_data(path, name_list):

    obj_cnt = 0
    bg_cnt = 0

    batch_size = len(name_list)
    images = np.zeros([batch_size, 300, 300, 3])
    loc = np.zeros([batch_size, 8732, 4])
    cls = np.full([batch_size, 8732], config.class_num-1)  # default to back ground
    for batch_index, name in enumerate(name_list):
        img = cv2.imread(os.path.join(path, 'JPEGImages', name+'.jpg'))
        img = cv2.resize(img, (300, 300))
        images[batch_index, :, :, :] = img[:, :, :]
        # cv2.imshow(name, img)
        # cv2.waitKey(500)

        labels_each_batch = []
        tree = ET.parse(os.path.join(path, 'Annotations', name+'.xml'))
        root = tree.getroot()
        width = int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)
        for object in root.iter('object'):
            # each object
            # x-width, y-height
            bbox = object.find('bndbox')
            xmin = int(bbox.find('xmin').text) * 300 / width  # left
            ymin = int(bbox.find('ymin').text) * 300 / height  # top
            xmax = int(bbox.find('xmax').text) * 300 / width  # right
            ymax = int(bbox.find('ymax').text) * 300 / height  # bottom
            box_truth = [xmin, ymin, xmax, ymax]

            start_index = 0
            for k, (num, size) in enumerate(zip(config.box_num, config.feature_map_size)):
                # kth feature map, number of boxes in one pixel and feature map size
                for i in range(size):
                    for j in range(size):
                        # position (i, j) in feature map
                        center_x = (i + 0.5) / size
                        center_y = (j + 0.5) / size
                        if num == 4:
                            aspect_ratio = [1, 1, 2, 0.5]
                            scale = [1, np.sqrt(config.scale[k] * config.scale[k+1]), 1, 1]
                        if num == 6:
                            aspect_ratio = [1, 1, 2, 0.5, 3, 1/3.0]
                            scale = [1, np.sqrt(config.scale[k] * config.scale[k+1]), 1, 1, 1, 1]
                        for box_index, (ar, s) in enumerate(zip(aspect_ratio, scale)):
                            s = config.scale[k]
                            w = s * np.sqrt(ar)
                            h = s / np.sqrt(ar)
                            default_xmin = (center_x - w / 2) * 300
                            default_ymin = (center_y - h / 2) * 300
                            default_xmax = (center_x + w / 2) * 300
                            default_ymax = (center_y + h / 2) * 300
                            box_default = [default_xmin, default_ymin, default_xmax, default_ymax]
                            loc[batch_index, start_index + (i * size + j) * num + box_index, :] = box_default[:]
                            IoU = cal_iou(box_truth, box_default)
                            if IoU > 0.5:
                                label = object.find('name').text
                                # obj_cnt += 1
                            else:
                                label = 'back_ground'
                                # bg_cnt += 1
                            if cls[batch_index, start_index + (i * size + j) * num + box_index] == config.class_num - 1:
                                # TODO IoU mark
                                cls[batch_index, start_index + (i * size + j) * num + box_index] = config.class_dict[label]
                start_index += size * size * num


            # print(bg_cnt, obj_cnt)
            # if obj_cnt == 0:
            #     img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            #     cv2.imshow(name, img)
            #     cv2.waitKey(2000)
            #
            # obj_cnt = 0
            # bg_cnt = 0
            # print(xmin, ymin, xmax, ymax)
            # img = cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            # cv2.rectangle(img, (left, top), (right, bottom), (r, g, b), thickness)

    return images, loc, cls


if __name__ == '__main__':
    with open(config.trainval, 'r') as f:
        for name in f:
            images, loc, cls = load_data(config.path, [name[0:6]])
            # print(np.shape(images), np.shape(labels))
            # cls = labels[:, :, 4]
            # loc = labels[:, :, 0:4]
            print(np.shape(cls), np.shape(loc))
            # print(cls)
            ind = np.where(cls < 20)
            print(np.shape(ind), ind)
