from ast import arg

import torch
from torch.utils.data import DataLoader

import os

from transformers import AdamW
from tqdm import tqdm
from utils import set_seed, parse_args
from utils import getParentPath,  DATASET_PATHS #print_timeNow
from transformers import get_linear_schedule_with_warmup
from torch.utils.data.dataset import random_split
from ACD_dataset import special_tokens_dict, label_id_to_name, get_dataset
from datetime import datetime

from ACD_models import ACD_model

from transformers import AutoTokenizer

device = torch.device("cuda:1")


task_name = 'ABSA' 
taskDir_path, fname_train, fname_dev, fname_test,  = DATASET_PATHS[task_name]

args = parse_args()
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def train_sentiment_analysis(args):
    timestamp = datetime.now().strftime("%m%d-%H%M") #일시-시간:분 을 모델 run으로 선정, 이후 이 run의 파라미터와 결과를 따로 파일로 한 줄씩 저장


    if not os.path.exists(args.entity_property_model_path):
        os.makedirs(args.entity_property_model_path)
    if not os.path.exists(args.polarity_model_path):
        os.makedirs(args.polarity_model_path)
        
    homePth = getParentPath(os.getcwd())
    homePth = getParentPath(homePth)
    
    datasetPth = homePth+'/test/dataset/'
    print('homePth:',homePth,', curPth:',os.getcwd())

    tsvPth_train = datasetPth+taskDir_path+fname_train 
    tsvPth_dev = datasetPth+taskDir_path+fname_dev 

    random_seed_int = 5    
    set_seed(random_seed_int, device) #random seed 정수로 고정.

    print('tokenizing train data')
    tokenizer = AutoTokenizer.from_pretrained('paust/pko-t5-large')
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print('We have added', num_added_toks, 'tokens')
    print(tsvPth_train)
    entity_property_train_data = get_dataset(tsvPth_train, tokenizer)
    entity_property_dev_data = get_dataset(tsvPth_dev, tokenizer)
    
    print('entity_data: ', len(entity_property_train_data))
    print('entity_dev_count: ', len(entity_property_dev_data))
    
    
    #데이터 스플릿 함수
    # dataset_size = len(entity_property_train_data)
    # train_size = int(dataset_size * 0.95)
    # validation_size = dataset_size - train_size
    
    # print(f"Training Data Size : ",dataset_size)
    # print(f"Validation Data Size : ",train_size)
    # print(f"Testing Data Size : ",validation_size)
    
    # train_dataset, val_dataset = random_split(entity_property_train_data, [train_size,validation_size])
    
    entity_property_train_dataloader = DataLoader(entity_property_train_data, shuffle=True,
                                batch_size=args.batch_size,
                                num_workers=8,
                                pin_memory=True,)
    entity_property_dev_dataloader = DataLoader(entity_property_dev_data, shuffle=True,
                                batch_size=args.batch_size,
                                num_workers=8,
                                pin_memory=True,)
    
    #모델 로딩 부분
    print('loading model')
    entity_property_model = ACD_model(args, len(label_id_to_name), len(tokenizer))
    entity_property_model.to(device)
    
    print('end loading')
    
    #하이퍼 파라미터 튜닝 부분
    FULL_FINETUNING = True
    if FULL_FINETUNING:
        entity_property_param_optimizer = list(entity_property_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        entity_property_optimizer_grouped_parameters = [
            {'params': [p for n, p in entity_property_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in entity_property_param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        entity_property_param_optimizer = list(entity_property_model.classifier.named_parameters())
        entity_property_optimizer_grouped_parameters = [{"params": [p for n, p in entity_property_param_optimizer]}]

    entity_property_optimizer = AdamW(
        entity_property_optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.eps
    )
    epochs = args.num_train_epochs
    max_grad_norm = 1.0
    total_steps = epochs * len(entity_property_train_dataloader)

    entity_property_scheduler = get_linear_schedule_with_warmup(
        entity_property_optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    #모델 학습
    print('[Training Phase]')
    epoch_step = 0
    best_acc= -1
    best_f1= -1
    
    for _ in tqdm(range(epochs), desc="Epoch"):
        epoch_step += 1

        entity_property_model.train()

        # entity_property train
        entity_property_total_loss = 0

        for step, batch in enumerate(tqdm(entity_property_train_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            e_input_ids, e_input_mask, d_input_ids,d_attention_mask= batch
            
            labels = d_input_ids
            labels = torch.where(labels!=0,labels,-100)
        
        
            entity_property_model.zero_grad()
            loss, logits = entity_property_model(e_input_ids, e_input_mask, d_input_ids,d_attention_mask,labels)

            loss.backward()

            entity_property_total_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(parameters=entity_property_model.parameters(), max_norm=max_grad_norm)
            entity_property_optimizer.step()
            entity_property_scheduler.step()

        avg_train_loss = entity_property_total_loss / len(entity_property_train_dataloader)
        print("Entity_Property_Epoch: ", epoch_step)
        print("Average train loss: {}".format(avg_train_loss))

        model_saved_path = args.entity_property_model_path + f"{args.learning_rate}" + 'saved_model_epoch_' + str(epoch_step) + '.pt'
        torch.save(entity_property_model.state_dict(), model_saved_path)

        
        if args.do_eval:
            entity_property_model.eval()

            pred_list = []
            label_list = []
            for batch in entity_property_dev_dataloader:
                batch = tuple(t.to(device) for t in batch)
                e_input_ids, e_input_mask, d_input_ids,d_attention_mask = batch

                with torch.no_grad():
                    predictions = entity_property_model.test(e_input_ids, e_input_mask)
                    
                pred_list.extend(tokenizer.batch_decode(predictions, skip_special_tokens=True))
                label_list.extend(tokenizer.batch_decode(d_input_ids, skip_special_tokens=True))
                
            data_num = 0
            correct = 0 
            pred_num = 0  
            
            #Decoder 모델이 뽑아내는 정답의 형태가 대회 제출 형태가아닌 아래와 같은 형태로 뽑아 내기 때문에 이를 대회 제출형태로 바꿔줌
            entity_property_pair = [
            '제품 전체 품질', '제품 전체 디자인','제품 전체 다양성','제품 전체 인지도','제품 전체 일반','제품 전체 편의성','제품 전체 가격',
            '패키지 구성품 디자인',  '패키지 구성품 가격','패키지 구성품 다양성', '패키지 구성품 일반','패키지 구성품 편의성','패키지 구성품 품질',  
            '본품 일반',  '본품 다양성', '본품 품질', '본품 인지도', '본품 편의성',  '본품 디자인', '본품 가격',
            '브랜드 일반',  '브랜드 인지도', '브랜드 디자인',  '브랜드 품질', '브랜드 가격' ]

            #먼저 Category를 두가지로 나누어 제품 전체 , 품질 과같이 각각 찾아내면 조합하는 형식으로 정확도 비교
            for true,pred in zip(label_list,pred_list):
                true_l = []
                pred_l = []
                for i in entity_property_pair:
                    if i in true:
                        true_l.append(i)
                data_num = data_num + len(true_l)
                for i in entity_property_pair:
                    if i in pred:
                        pred_l.append(i)
                pred_num = pred_num + len(pred_l)
                for i in true_l:
                    if i in pred_l:
                        correct = correct+1
                        
            print(correct,data_num,pred_num)
            
            print("dev acc = ",correct/data_num)
            if (correct/data_num) > best_acc:
                best_acc = correct/data_num
            
            precision = correct/pred_num
            Recall = correct/data_num
            f1 = 2*precision*Recall/(precision+Recall)
            
            if f1 > best_f1:
                best_f1 = f1
            
            print(f1)
                

    print("training is done")

    
if __name__ == "__main__":
    
    args = parse_args()

    train_sentiment_analysis(args)
    

    