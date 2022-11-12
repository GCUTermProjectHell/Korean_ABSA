import os
import random
from datetime import datetime
import argparse
import json
import torch
import numpy as np
from transformers import (
    ElectraConfig,
    ElectraTokenizer,
    AutoConfig,
    RobertaModel,
    ElectraModel,
    AutoTokenizer,
    AutoModel,
    ElectraForQuestionAnswering
)

MODEL_CLASSES = {
    'roberta-base': (AutoConfig, RobertaModel, AutoTokenizer),
    'koelectra': (ElectraConfig, ElectraModel, ElectraTokenizer),
    'koelectraQA': (ElectraConfig, ElectraForQuestionAnswering, ElectraTokenizer),
    'koelectra_tunib':(AutoConfig, AutoModel, AutoTokenizer),
    'klue/roberta-large':(AutoConfig, AutoModel, AutoTokenizer),
    'beomi/KcELECTRA-base':(AutoConfig, AutoModel, AutoTokenizer),
    'BM-K/KoMiniLM-68M':(AutoConfig, AutoModel, AutoTokenizer),
    'kykim/electra-kor-base':(ElectraConfig, ElectraModel, ElectraTokenizer),
}

MODEL_PATH_MAP = {
    'koelectra': 'monologg/koelectra-base-v3-discriminator',
    'koelectraQA': 'monologg/koelectra-base-discriminator',
    'roberta-base': 'klue/roberta-base',
    'koelectra_tunib': 'tunib/electra-ko-base',
    'klue/roberta-large': 'klue/roberta-large',
    'beomi/KcELECTRA-base': 'beomi/KcELECTRA-base',
    'BM-K/KoMiniLM-68M': 'BM-K/KoMiniLM-68M',
    'kykim/electra-kor-base':'kykim/electra-kor-base'
}

DATASET_PATHS = {
    'ABSA' : ('task_ABSA/','nikluge-sa-2022-train.jsonl', 'nikluge-sa-2022-dev.jsonl', 'nikluge-sa-2022-test.jsonl')
} #train, dev ,test


TOKEN_MAX_LENGTH = {
    'ABSA' : 256,
} 


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
        "--load_pretrain",action="store_true"
    )
    parser.add_argument(
        "--load_model_path",type=str
    )
    parser.add_argument(
        "--save_path",type=str
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
        "--base_model", type=str, default="kykim/electra-kor-base"
    )
    parser.add_argument(
        "--model_entity", type=str, default="kykim/electra-kor-base"
    )
    parser.add_argument(
        "--model_polarity", type=str, default="kykim/electra-kor-base"
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
        "--polarity_classifier_hidden_size", type=int, default=768
    )
    parser.add_argument(
        "--classifier_dropout_prob", type=int, default=0.1, help="dropout in classifier"
    )
    parser.add_argument(
        "--run_name",type=str
    )
    args = parser.parse_args()
    return args



#get parent/home directory path
def getParentPath(pathStr):
    return os.path.abspath(pathStr+"../../")
#return parentPth/parentPth of pathStr 
def getHomePath(pathStr):
    return getParentPath(getParentPath(getParentPath(pathStr))) 

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


# jsonl 파일 읽어서 list에 저장
def jsonlload(fname, encoding="utf-8"):
    json_list = []
    with open(fname, encoding=encoding) as f:
        for line in f.readlines():
            json_list.append(json.loads(line))
    return json_list


