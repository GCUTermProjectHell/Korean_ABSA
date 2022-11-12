import torch
from torch.utils.data import  DataLoader
import os
from transformers import AdamW
from tqdm import trange
from utils import  set_seed, parse_args
from utils import MODEL_PATH_MAP
from utils import getParentPath, DATASET_PATHS 
from utils import MODEL_CLASSES, MODEL_PATH_MAP 
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from ky_ACD_dataset import special_tokens_dict
from ky_ACD_dataset import get_dataset
from ky_ACD_models import model_ABSA
from transformers import AutoTokenizer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'kykim/electra-kor-base'
_1, _2, model_tokenizer = MODEL_CLASSES[model_name] #config_class, model_class, model_tokenizer
task_name = 'ABSA' 
taskDir_path, fname_train, fname_dev, fname_test,  = DATASET_PATHS[task_name]

model_path = MODEL_PATH_MAP[model_name]
label_id_to_name = ['True', 'False']
polarity_id_to_name = ['positive', 'negative', 'neutral']

args=parse_args()


def evaluation(y_true, y_pred, label_len):
    count_list = [0]*label_len
    hit_list = [0]*label_len
    for i in range(len(y_true)):
        count_list[y_true[i]] += 1
        if y_true[i] == y_pred[i]:
            hit_list[y_true[i]] += 1
    acc_list = []

    for i in range(label_len):
        acc_list.append(hit_list[i]/count_list[i])

    y_true = list(map(int, y_true))
    y_pred = list(map(int, y_pred))
  
    print(count_list)
    print(hit_list)
    print(hit_list[0],",",count_list[0],",",hit_list[0]+count_list[1]-hit_list[1])
    print(acc_list)
    print('Entity_Property_accuracy: ', (sum(hit_list) / sum(count_list)))
    print('Entity_Property_macro_accuracy: ', sum(acc_list) / 2)

    y_true = list(map(int, y_true))
    y_pred = list(map(int, y_pred))

    print('Entity_Property_f1_score: ', f1_score(y_true, y_pred, average=None))
    print('Entity_Property_f1_score_micro: ', f1_score(y_true, y_pred, average='micro'))
    print('Entity_Property_f1_score_macro: ', f1_score(y_true, y_pred, average='macro'))
    
        
def train_sentiment_analysis(args):
    if not os.path.exists(args.entity_property_model_path):
        os.makedirs(args.entity_property_model_path)
    if not os.path.exists(args.polarity_model_path):
        os.makedirs(args.polarity_model_path)
    homePth = getParentPath(os.getcwd())
    print('homePth:',homePth,', curPth:',os.getcwd())

    tsvPth_train = args.train_data  
    tsvPth_dev ='../../dataset/task_ABSA/nikluge-sa-2022-dev.jsonl'
    random_seed_int = 5 # 랜덤시드 넘버=5 로 고정
    set_seed(random_seed_int, device) #random seed  고정.

    print('tokenizing train data')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    entity_property_train_data, polarity_train_data = get_dataset(tsvPth_train, tokenizer, args.max_len)
    entity_property_dev_data, polarity_dev_data = get_dataset(tsvPth_dev, tokenizer, args.max_len)
    
    entity_property_train_dataloader = DataLoader(entity_property_train_data, shuffle=True,
                                  batch_size=args.batch_size)
    entity_property_dev_dataloader = DataLoader(entity_property_dev_data, shuffle=True,
                                batch_size=args.batch_size)
    
    print('loading model')
    entity_property_model = model_ABSA(args, len(label_id_to_name), len(tokenizer))
    entity_property_model.to(device)
    
    print('end loading')
    
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
    
    print('[Training Phase]')
    epoch_step = 0

    for _ in trange(epochs, desc="Epoch"):
        epoch_step += 1
    
        entity_property_model.train()
        

        # entity_property train
        entity_property_total_loss = 0

        for step, batch in enumerate(entity_property_train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            entity_property_model.zero_grad()

            loss, _ = entity_property_model(b_input_ids, b_input_mask, b_labels)

            loss.backward()

            entity_property_total_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(parameters=entity_property_model.parameters(), max_norm=max_grad_norm)
            entity_property_optimizer.step()
            entity_property_scheduler.step()

        avg_train_loss = entity_property_total_loss / len(entity_property_train_dataloader)
        print("Entity_Property_Epoch: ", epoch_step)
        print("Average train loss: {}".format(avg_train_loss))

        model_saved_path = args.entity_property_model_path + 'saved_model_epoch_' + str(epoch_step) + '.pt'
        torch.save(entity_property_model.state_dict(), model_saved_path)

        if args.do_eval:
            entity_property_model.eval()

            pred_list = []
            label_list = []

            for batch in entity_property_dev_dataloader:
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                with torch.no_grad():
                    loss, logits = entity_property_model(b_input_ids, b_input_mask, b_labels)

                predictions = torch.argmax(logits, dim=-1)
                pred_list.extend(predictions)
                label_list.extend(b_labels)

            evaluation(label_list, pred_list, len(label_id_to_name))


    print("training is done")


if __name__ == "__main__":

    if args.do_train:
        train_sentiment_analysis(args)
    


