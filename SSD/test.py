import tensorflow as tf
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

def nms(boxes, scores, class_num, max_boxes=50, score_threshold=0.5, iou_threshold=0.5):
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
        idx = np.where(labels == i)[0]
        boxes_ = boxes[idx]
        scores_ = scores[idx]
        # labels_ = labels[idx]

        # sort by score
        idx = np.argsort(scores_)   # up
        idx = idx[::-1]  # down
        boxes_ = boxes_[idx]
        scores_ = scores_[idx]
        # labels_ = labels_[idx]

        cnt = 0
        for j, box in enumerate(boxes_):
            if j == 0:
                boxes_output.append(boxes_[j])
                scores_output.append(scores_[j])
                labels_output.append(i)
                cnt += 1
            else:
                if scores_[j] > score_threshold:
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