import cv2
import os
import xml.etree.ElementTree as ET
import numpy as np
import config
import time

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

def generate_default_boxes(form):
    """

    Args:
        form: 'xywh' is for center x, y and size width and height
              'ltrb' is for left, top, right and bottom

    Returns:

    """
    boxes = np.zeros([8732, 4])
    default_box = np.zeros([4])
    start_index = 0
    for k, (num, size) in enumerate(zip(config.box_num, config.feature_map_size)):
        # kth feature map, number of boxes in one pixel and feature map size
        for y in range(size):
            for x in range(size):
                # position (x, y) in feature map
                # above order is important.
                # in math width (x) comes first
                # in data structure height (y) comes first
                center_x = (x + 0.5) / size
                center_y = (y + 0.5) / size
                if num == 4:
                    aspect_ratio = [1, 1, 2, 0.5]
                    scale = np.full([4], config.scale[k])
                    scale[1] = np.sqrt(config.scale[k] * config.scale[k + 1])
                if num == 6:
                    aspect_ratio = [1, 1, 2, 0.5, 3, 1 / 3.0]
                    scale = np.full([6], config.scale[k])
                    scale[1] = np.sqrt(config.scale[k] * config.scale[k + 1])

                for box_index, (ar, s) in enumerate(zip(aspect_ratio, scale)):
                    w = s * np.sqrt(ar)
                    h = s / np.sqrt(ar)

                    if form == 'xywh':
                        default_box[0] = center_x * 300
                        default_box[1] = center_y * 300
                        default_box[2] = w * 300
                        default_box[3] = h * 300
                    if form == 'ltrb':
                        default_box[0] = (center_x - w / 2) * 300
                        default_box[1] = (center_y - h / 2) * 300
                        default_box[2] = (center_x + w / 2) * 300
                        default_box[3] = (center_y + h / 2) * 300

                    boxes[start_index + (y * size + x) * num + box_index, :] = default_box[:]
        start_index += size * size * num
    return boxes


def load_data(path, name_list, default_boxes):
    batch_size = len(name_list)
    images = np.zeros([batch_size, 300, 300, 3], dtype=np.float32)
    cls = np.full([batch_size, 8732], config.class_num-1, dtype=np.int32)  # default to back ground
    loc = np.zeros([batch_size, 8732, 4], dtype=np.float32)  # default to no offsets
    for batch_index, name in enumerate(name_list):
        print(batch_index)
        obj_cnt = 0
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

            for i, default_box in enumerate(default_boxes):
                IoU = cal_iou(box_truth, default_box)
                if IoU > 0.5:
                    label = object.find('name').text
                else:
                    label = 'back_ground'
                if cls[batch_index, i] == config.class_num - 1:
                    # if is default to back ground
                    # TODO IoU mark
                    cls[batch_index, i] = config.class_dict[label]
                    if label != 'back_ground':
                        obj_cnt += 1
                        # save offsets, center x, y and width, height

                        # g for ground truth box
                        gx = (box_truth[0] + box_truth[2]) / 2
                        gy = (box_truth[1] + box_truth[3]) / 2
                        gw = box_truth[2] - box_truth[0]
                        gh = box_truth[3] - box_truth[1]

                        # d for default box
                        dx = (default_box[0] + default_box[2]) / 2
                        dy = (default_box[1] + default_box[3]) / 2
                        dw = default_box[2] - default_box[0]
                        dh = default_box[3] - default_box[1]

                        loc[batch_index, i, 0] = (gx - dx) / dw
                        loc[batch_index, i, 1] = (gy - dy) / dh
                        loc[batch_index, i, 2] = np.log(gw / dw)
                        loc[batch_index, i, 3] = np.log(gh / dh)
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
        print(obj_cnt)

    return images, loc, cls


if __name__ == '__main__':
    default_boxes = generate_default_boxes('ltrb')
    with open(config.train, 'r') as f:
        name_list = []
        for name in f:
            name_list.append(name[0:6])

    print(len(name_list))
    images, loc, cls = load_data(config.path, name_list, default_boxes)
    np.save('preload/images.npy', images)
    np.save('preload/loc.npy', loc)
    np.save('preload/cls.npy', cls)
    # for i in range(300):
    #     t = time.time()
    #     images, loc, cls = load_data(config.path, name_list[i: i+8], default_boxes)
    #     print(time.time() - t)
    #     np.save('images.npy', images)
    #     np.save('loc.npy', loc)
    #     np.save('cls.npy', cls)
    #     break
    # with open(config.trainval, 'r') as f:
    #     for name in f:
    #         t = time.time()
    #         images, loc, cls = load_data(config.path, [name[0:6]], default_boxes)
    #         print(time.time() - t)
            # print(images.dtype, loc.dtype, cls.dtype)
            # print(np.shape(images), np.shape(labels))
            # cls = labels[:, :, 4]
            # loc = labels[:, :, 0:4]
            # print(np.shape(cls), np.shape(loc))
            # # print(cls)
            # ind = np.where(cls < 20)
            # print(np.shape(ind), ind)

