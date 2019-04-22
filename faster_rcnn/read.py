import cv2
import os
import xml.etree.ElementTree as ET

# path_image = '/home/user/Documents/dataset/VOC/VOC/VOCdevkit/VOC2007/JPEGImages/000009.jpg'
# path_annotation = '/home/user/Documents/dataset/VOC/VOC/VOCdevkit/VOC2007/Annotations/000009.xml'

def show(path_image, path_annotation):
    img = cv2.imread(path_image)
    img = cv2.resize(img, (1000, 600))
    cv2.imshow('show', img)
    cv2.waitKey(1000)

    #打开xml文档


    tree = ET.parse(path_annotation)
    # 获取 XML 文档对象的根结点 Element
    root = tree.getroot()
    # 打印根结点的名称
    print(root.tag)

    width = int(root.find('size').find('width').text)
    height = int(root.find('size').find('height').text)
    for object in root.iter('object'):
        # x-width, y-height
        print(object.find('name').text)
        bbox = object.find('bndbox')
        xmin = int(bbox.find('xmin').text) * 224 / width
        ymin = int(bbox.find('ymin').text) * 224 / height
        xmax = int(bbox.find('xmax').text) * 224 / width
        ymax = int(bbox.find('ymax').text) * 224 / height
        # print(xmin, ymin, xmax, ymax)
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)


    # cv2.imshow('show', img)
    # cv2.waitKey(1000)

list = os.listdir('/home/user/Documents/dataset/VOC/VOC/VOCdevkit/VOC2007/JPEGImages/')
for _ in list:
    path = '/home/user/Documents/dataset/VOC/VOC/VOCdevkit/VOC2007/'
    show(path+'JPEGImages/'+_, path+'Annotations/'+_.replace('jpg', 'xml'))