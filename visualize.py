# -*- coding: utf-8
import numpy as np
import os
import cv2

def draw(image_data, line):
    line = line.strip()
    points_list = line.split(' ')[:-2]
    if points_list:
        points = [int(point) for point in points_list]
        points = np.reshape(points, (-1, 2))
        contours = np.asarray([[list(p) for p in points]])
        cv2.drawContours(image_data, contours, -1, (0, 255, 0), 2)

def visualize(image_root, txt_root, output_root):
    def read_gt_file(image_name):
        gt_file = os.path.join(txt_root, '%s.txt'%(image_name.split('.bmp')[0]))
        with open(gt_file, 'r') as gt_f:
            return gt_f.readlines()
    
    def read_image_file(image_name):
        img_file = os.path.join(image_root, image_name)
        img_array = cv2.imread(img_file)
        return img_array

    for image_name in os.listdir(image_root):
        image_data = read_image_file(image_name)
        gt_image_data = image_data.copy()
        if txt_root:
            gt_list = read_gt_file(image_name)
            for gt in gt_list:
                draw(gt_image_data, gt)
                cv2.imwrite(os.path.join(output_root, image_name), gt_image_data)


if __name__ == '__main__':    
    image_root = r'E:\Data\\picture\\train\\images'
    txt_root = r'E:\Data\\picture\\train\\labelTxt'
    output_root = r'E:\Data\\picture\\train\\vis'
    visualize(image_root, txt_root, output_root)
