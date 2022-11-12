import os
import random

from datetime import datetime

import ntpath
import argparse

import json

import torch
import numpy as np


def set_seed(seedNum, device):
    torch.manual_seed(seedNum)
    torch.cuda.manual_seed(seedNum)
    torch.cuda.manual_seed_all(seedNum) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seedNum)
    random.seed(seedNum)


def parse_args():
    parser = argparse.ArgumentParser(description="sentiment analysis")
    parser.add_argument(
        "--train_data", type=str, default="../data/input_data_v1/train.json",
        help="train file"
    )
    parser.add_argument(
        "--test_data", type=str, default="../data/input_data_v1/test.json",
        help="test file"
    )
    parser.add_argument(
        "--dev_data", type=str, default="../data/input_data_v1/dev.json",
        help="dev file"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-5
    )
    parser.add_argument(
        "--eps", type=float, default=1e-8
    )
    parser.add_argument(
        "--do_train", action="store_true"
    )
    parser.add_argument(
        "--do_eval", action="store_true"
    )
    parser.add_argument(
        "--do_test", action="store_true"
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=3
    )
    parser.add_argument(
        "--base_model", type=str, default="kclectra"
    )
    parser.add_argument(
        "--entity_property_model_path", type=str, default="./saved_models/default_path/"
    )
    parser.add_argument(
        "--polarity_model_path", type=str, default="./saved_models/default_path/"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output/default_path/"
    )
    parser.add_argument(
        "--do_demo", action="store_true"
    )
    parser.add_argument(
        "--max_len", type=int, default=256
    )

    parser.add_argument(
        "--classifier_hidden_size", type=int, default=768
    )
    parser.add_argument(
        "--classifier_dropout_prob", type=int, default=0.1, help="dropout in classifier"
    )
    parser.add_argument(
        "--pred_data_path", type=str
    )
    parser.add_argument(
        "--load_pred_data", type=str
    )
    
    args = parser.parse_args()
    return args


##get Path functions
def path_fpath(path):
    fpath, fname = ntpath.split(path)
    return fpath #fpath or ntpath.basename(fname)
def path_leaf(path):
    fpath, fname = ntpath.split(path)
    return ntpath.basename(fname) #fpath or ntpath.basename(fname)
def getFName(fname):
    fname_split = fname.split('.') #name, extenstion
    new_fname=fname_split[0]#+'.jpg'
    return new_fname

##get parent/home directory path##
def getParentPath(pathStr):
    return os.path.abspath(pathStr+"../../")
#return parentPth/parentPth of pathStr -> hdd1/
def getHomePath(pathStr):
    return getParentPath(getParentPath(getParentPath(pathStr))) #ast/src/

def print_timeNow():
    cur_day_time = datetime.now().strftime("%m/%d, %H:%M:%S") #Date %m/%d %H:%M:%S
    return cur_day_time

def jsonload(fname, encoding="utf-8"):
    with open(fname, encoding=encoding) as f:
        j = json.load(f)

    return j

# json 개체를 파일이름으로 깔끔하게 저장
def jsondump(j, fname):
    with open(fname, "w", encoding="UTF8") as f:
        json.dump(j, f, ensure_ascii=False)
        
# json list를 jsonl 형태로 저장
def jsonldump(j_list, fname):
    f = open(fname, "w", encoding='utf-8')
    for json_data in j_list:
        f.write(json.dumps(json_data, ensure_ascii=False)+'\n')


# jsonl 파일 읽어서 list에 저장
def jsonlload(fname, encoding="utf-8"):
    json_list = []
    with open(fname, encoding=encoding) as f:
        for line in f.readlines():
            json_list.append(json.loads(line))
    return json_list




