#!/usr/bin/env python
# 执行以下语句
# python labelme2voc.py data_annotated data_dataset_voc --labels label.txt

from __future__ import print_function

import argparse
import glob
import os
import os.path as osp
import sys

import imgviz
import numpy as np

import labelme
from multiprocessing import Process,Pool

'''
1,加入多线程

'''


def mkrs(args):
    # 创建文件夹
    if not osp.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # if osp.exists(args.output_dir):
    #     print("Output directory already exists:", args.output_dir)
    #     sys.exit(1)
    # os.makedirs(args.output_dir)
    if not osp.exists(osp.join(args.output_dir, "ImageSets/Segmentation")):
        os.makedirs(osp.join(args.output_dir, "ImageSets/Segmentation"))
    text_create('train')
    text_create('val')
    text_create('trainval')

    # os.makedirs(osp.join(args.output_dir, "SegmentationClassNpy"))
    if not osp.exists(osp.join(args.output_dir, "JPEGImages")):
        os.makedirs(osp.join(args.output_dir, "JPEGImages"))
    if not osp.exists(osp.join(args.output_dir, "SegmentationClass")):
        os.makedirs(osp.join(args.output_dir, "SegmentationClass"))
    if not args.noviz:
        if not osp.exists(osp.join(args.output_dir, "SegmentationClassVisualization")):
            os.makedirs(osp.join(args.output_dir, "SegmentationClassVisualization"))
        # os.makedirs(
        #     osp.join(args.output_dir, "SegmentationClassVisualization")
        # )
    print("Creating dataset:", args.output_dir)


# 创建一个txt文件，文件名为mytxtfile,并向文件写入msg
def text_create(name):
    desktop_path = "./VOCdevkit/VOC2007/ImageSets/Segmentation/"  # 新创建的txt文件的存放路径
    full_path = desktop_path + name + '.txt'  # 也可以创建一个.doc的word文档
    file = open(full_path, 'w')
    file.close()
    print(f'{name}.txt已创建')

def generate_dataset(args, class_names, class_name_to_id, filename):
    print("Generating dataset from:", filename)
    # 读取json文件
    label_file = labelme.LabelFile(filename=filename)
    base = osp.splitext(osp.basename(filename))[0]
    # 读取图片
    out_img_file = osp.join(args.output_dir, "JPEGImages", base + ".jpg")
    # npy文件
        # out_lbl_file = osp.join(
        #     args.output_dir, "SegmentationClassNpy", base + ".npy"
        # )
    # png文件
    out_png_file = osp.join(
            args.output_dir, "SegmentationClass", base + ".png"
        )
    # 可视化文件
    if not args.noviz:
        out_viz_file = osp.join(
                args.output_dir,
                "SegmentationClassVisualization",
                base + ".jpg",
            )
    # 保存图片
    with open(out_img_file, "wb") as f:
        f.write(label_file.imageData)
    img = labelme.utils.img_data_to_arr(label_file.imageData)

    lbl, _ = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            shapes=label_file.shapes,
            label_name_to_value=class_name_to_id,
        )
    labelme.utils.lblsave(out_png_file, lbl)

        # np.save(out_lbl_file, lbl)

    if not args.noviz:
        viz = imgviz.label2rgb(
                lbl,
                imgviz.rgb2gray(img),
                font_size=15,
                label_names=class_names,
                loc="rb",
            )
        imgviz.io.imsave(out_viz_file, viz)

def label2voc(paras):
    #获取参数
    args = paras
    #创建文件夹
    mkrs(args)
    #获取类别
    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # 开始处 -1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == "__ignore__"
            continue#跳过本次循环
        elif class_id == 0:
            assert class_name == "_background_"
        class_names.append(class_name)
    class_names = tuple(class_names)
    print("class_names:", class_names)
    print("class_name_to_id:", class_name_to_id)
    out_class_names_file = osp.join(args.output_dir, "class_names.txt")
    with open(out_class_names_file, "w") as f:
        f.writelines("\n".join(class_names))
    print("Saved class_names:", out_class_names_file)

    # 文件处理
    pool=Pool()
    for filename in glob.glob(osp.join(args.input_dir, "*.json")):
        # generate_dataset(args, class_names, class_name_to_id, filename)
        pool.apply_async(generate_dataset, (args, class_names, class_name_to_id, filename))
    pool.close()
    pool.join()

