import time
import argparse
from utils.labelme2voc import label2voc
from utils.voc_trainval import voc_trainval

def add_paras():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_dir",default='wait_to_process\E collapse angle', help="input annotated directory")
    parser.add_argument("--output_dir",default='VOCdevkit\VOC2007', help="output dataset directory")
    parser.add_argument("--labels",default='label.txt', help="labels file")
    parser.add_argument("--noviz", help="no visualization", action="store_true")
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    # 要修改参数就修改add_paras（）、
    paras=add_paras()
    label2voc(paras)

    path=paras.output_dir
    train_percent=0.9
    voc_trainval(path,train_percent)