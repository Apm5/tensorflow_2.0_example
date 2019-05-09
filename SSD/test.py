import tensorflow as tf
import numpy as np
import cv2
from load_data import load_data, generate_default_boxes
from SSD import Model
import config

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

def nms(boxes, scores, class_num, max_boxes=50, score_threshold=0.50, iou_threshold=0.50):
    """

    Args:
        boxes: shape [boxes_num, 4]
        scores: shape [boxes_num, class_num + 1]
        class_num:
        max_boxes:
        iou_threshold:

    Returns:

    """
    labels = np.array([np.argmax(_) for _ in scores])
    # ares = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    boxes_output = []
    scores_output = []
    labels_output = []

    for i in range(class_num):
        # select
        ind = np.where(labels == i)[0]
        boxes_ = boxes[ind]
        scores_ = scores[ind]
        # labels_ = labels[idx]

        # sort by score
        ind = np.argsort(scores_[:, i])   # up
        ind = ind[::-1]  # down
        boxes_ = boxes_[ind]
        scores_ = scores_[ind]
        # labels_ = labels_[idx]
        print(scores_, i)

        cnt = 0
        for j, box in enumerate(boxes_):
            if scores_[j, i] > score_threshold:
                if j == 0:
                    boxes_output.append(boxes_[j])
                    scores_output.append(scores_[j])
                    labels_output.append(i)
                    cnt += 1
                else:
                    flag = True  # check iou
                    for k in range(j):  # with before
                        if cal_iou(boxes_[j], boxes_[k]) > iou_threshold:
                            flag = False
                            break
                    if flag:
                        boxes_output.append(boxes_[j])
                        scores_output.append(scores_[j])
                        labels_output.append(i)
                        cnt += 1
            if cnt == max_boxes:
                break

    return boxes_output, scores_output, labels_output

def test(model, data, label, class_num, iou_threshold=0.5):
    """

    Args:
        model:
        data: shape [images_num / batch, batch, height, width, 3]
        label: shape [images_num, box_num, 5]
        class_num:

    Returns:

    """
    batch_size = len(data[0])
    boxes_true = label[:, :, 0:4]
    labels_true = label[:, :, 4]
    matched = np.zeros(np.shape(labels_true))

    scores_pred = []
    accurate = []
    obj_num = np.zeros(class_num)
    for i in range(class_num):
        scores_pred.append([])
        accurate.append([])
        obj_num[i] = len(np.where(labels_true == i))


    # pass model and nms
    for i, data_on_batch in enumerate(data):
        boxes_on_batch, scores_on_batch = model(data_on_batch)
        for j, (boxes, scores) in enumerate(zip(boxes_on_batch, scores_on_batch)):
            boxes_output, scores_output, labels_output = nms(boxes, scores, class_num)

            for p in range(len(boxes_output)):
                for q in range(len(boxes_true[i*batch_size+j])):
                    if not matched[i*batch_size+j, q] and labels_output[p]==labels_true[i*batch_size+j, q]:
                        # 未被检测到且类别匹配
                        IoU = cal_iou(boxes_output[p], boxes_true[i*batch_size+j, q])
                        if IoU > iou_threshold:
                            matched[i * batch_size + j, q] = 1  # 标记已检测到
                            scores_pred[labels_output[p]].append(scores_pred[p])  # 按类别存放
                            accurate[labels_output[p]].append(scores_pred[p])  # 按类别存放

    mAP = 0.0
    for i in range(class_num):
        idx = np.argsort(scores_pred[i])  # up
        idx = idx[::-1]  # down
        accurate = accurate[idx]  # sort by scores

        box_num = len(scores_pred)
        tp = np.zeros([box_num])
        fp = np.zeros([box_num])
        for i in range(box_num):
            if accurate[i]:
                tp[i] = 1
            else:
                fp[i] = 1
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)

        prec = tp / (tp + fp)
        rec = tp / obj_num[i]

        AP = prec[0] * rec[0]
        for j in range(1, len(prec)):
            AP += prec[j] * (rec[j] - rec[j-1])
        mAP += AP
    mAP /= class_num
    return mAP

def test_image(model, images):
    default_boxes = generate_default_boxes('xywh')
    model.load_weights('weights/weights_499')
    cls, loc = model(images, train=False)
    cls, loc = cls.numpy(), loc.numpy()
    # print(loc)
    # print(np.shape(loc), np.shape(cls))
    ind = np.where(cls_true < 20)

    tmp = np.zeros([4])
    for box, score in zip(loc, cls):
        print(np.shape(score))
        # print(score[ind[1]])
        # print(box[ind[1]])
        # print(score[:, 6])
        for i in range(len(box)):
            # print(box)
            # offset to xywh
            tmp[0] = box[i, 0] * default_boxes[i, 2] + default_boxes[i, 0]
            tmp[1] = box[i, 1] * default_boxes[i, 3] + default_boxes[i, 1]
            tmp[2] = np.exp(box[i, 2]) * default_boxes[i, 2]
            tmp[3] = np.exp(box[i, 3]) * default_boxes[i, 3]

            # xywh to ltrb
            box[i, 0] = tmp[0] - tmp[2] / 2
            box[i, 1] = tmp[1] - tmp[3] / 2
            box[i, 2] = tmp[0] + tmp[2] / 2
            box[i, 3] = tmp[1] + tmp[3] / 2

        boxes_output, scores_output, labels_output = nms(box, score, 20)
        print(np.shape(boxes_output))
        print(np.shape(scores_output))
        print(np.shape(labels_output))

        print(boxes_output)
        print(labels_output)
        # xmin > 90 < / xmin > < ymin > 125 < / ymin > < xmax > 337 < / xmax > < ymax > 212 < / ymax >
        img = np.array(images[0], dtype=np.uint8)
        # img = cv2.rectangle(img, (54, 113), (202, 191), (0, 255, 0), 2)
        # print(np.shape(img))
        # print(img.dtype)
        cv2.imshow('show', img)
        cv2.waitKey(500)
        for pos in boxes_output:
            img = cv2.rectangle(img, (pos[0], pos[1]), (pos[2], pos[3]), (0, 255, 0), 2)
            cv2.imshow('show', img)
            cv2.waitKey(500)

def fuck(model, images, loc_true, cls_true ):

    model.load_weights('weights/weights_37')
    cls, loc = model(images, train=True)
    # cls, loc = cls.numpy(), loc.numpy()
    cls, loc = [cls.numpy()[0]], [loc.numpy()[0]]
    # print(loc)
    print(np.shape(loc), np.shape(cls))
    ind = np.where(cls_true < 20)
    # print(ind)
    # print(cls_true[ind])
    # print(loc_true[ind])
    for i in ind[1]:
        print('true:', cls_true[0][i])
        print('pred:', cls[0][i])
        print('true:', loc_true[0][i])
        print('pred:', loc[0][i])
        img = np.array(images[0], dtype=np.uint8)
        cv2.imshow('show', img)
        cv2.waitKey(500)

        default_boxes = generate_default_boxes('ltrb')
        img = cv2.rectangle(img, (int(default_boxes[i][0]), int(default_boxes[i][1])), (int(default_boxes[i][2]), int(default_boxes[i][3])), (0, 255, 0), 2)

        default_boxes = generate_default_boxes('xywh')
        tmp = np.zeros([4])
        box = loc[0][i]
        tmp[0] = box[0] * default_boxes[i, 2] + default_boxes[i, 0]
        tmp[1] = box[1] * default_boxes[i, 3] + default_boxes[i, 1]
        tmp[2] = np.exp(box[2]) * default_boxes[i, 2]
        tmp[3] = np.exp(box[3]) * default_boxes[i, 3]

        # xywh to ltrb
        box[0] = tmp[0] - tmp[2] / 2
        box[1] = tmp[1] - tmp[3] / 2
        box[2] = tmp[0] + tmp[2] / 2
        box[3] = tmp[1] + tmp[3] / 2
        img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

        cv2.imshow('show', img)
        cv2.waitKey(500)

if __name__ == '__main__':
    model = Model()
    default_boxes = generate_default_boxes('ltrb')
    with open(config.train, 'r') as f:
        name_list = []
        for name in f:
            name_list.append(name[0:6])
    for i in range(2, len(name_list)):
        # images, loc_true, cls_true = load_data(config.path, name_list[i:i+5], default_boxes)
        images, loc_true, cls_true = load_data(config.path, name_list[i:i + 1], default_boxes)
        fuck(model, images, loc_true, cls_true)
        # test_image(model, images)

    # print(loc_true)
    # fuck(model, images, loc_true, cls_true)
    # f = open('SSD_result.txt', 'a')
    # f.write('test')



    # images_list = np.load('preload/images.npy')
    # cls_list = np.load('preload/cls.npy')
    # loc_list = np.load('preload/loc.npy')
    #
    # # tf.Tensor([0.76544    0.4478188  0.02661571 ... 0.13020377 0.99397826 0.31444868], shape=(2612,), dtype=float32)
    # # tf.Tensor([1. 1. 1. ... 1. 1. 1.], shape=(2612,), dtype=float32)
    # cnt = 0
    # default_boxes = generate_default_boxes('ltrb')
    # for i in range(10):
    #     images, loc_true, cls_true = images_list[i], loc_list[i], cls_list[i]
    #     print(np.shape(loc_true), np.shape(cls_true))
    #     ind = np.where(cls_true < 20)
    #     print(np.shape(ind))
    #     cnt += np.shape(ind)[1]
    #
    #     boxes = default_boxes[ind[0]]
    #     images = np.array(images, dtype=np.uint8)
    #     for box in boxes:
    #         print(box)
    #         box = np.array(box, dtype=np.int32)
    #         img = cv2.rectangle(images, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    #         cv2.imshow('show', img)
    #         cv2.waitKey(500)
    # print(cnt)
