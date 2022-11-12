import torch

from torch.utils.data import DataLoader

import os


from transformers import AdamW
from utils import set_seed,  parse_args
from utils import MODEL_PATH_MAP
from utils import getParentPath, DATASET_PATHS 
from utils import MODEL_PATH_MAP 
from transformers import get_linear_schedule_with_warmup
from torch.utils.data.dataset import random_split
from tqdm import tqdm

from ASC_dataset import get_dataset, polarity_id_to_name, special_tokens_dict
from ASC_models import ASC_model

from transformers import AutoTokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_name = 'kyelectra'

task_name = 'ABSA' 
taskDir_path, fname_train, fname_dev, fname_test,  = DATASET_PATHS[task_name]

model_path = MODEL_PATH_MAP[model_name]


args = parse_args()
args.base_model = model_path
os.environ["TOKENIZERS_PARALLELISM"] = "true"
        
def train_sentiment_analysis(args):
    if not os.path.exists(args.entity_property_model_path):
        os.makedirs(args.entity_property_model_path)
    if not os.path.exists(args.polarity_model_path):
        os.makedirs(args.polarity_model_path)
    homePth = getParentPath(os.getcwd())
    datasetPth = homePth+'/dataset/'
    print('homePth:',homePth,', curPth:',os.getcwd())


    tsvPth_train = datasetPth+taskDir_path+fname_train 
    tsvPth_dev = datasetPth+taskDir_path+fname_dev 

    random_seed_int = 5 
    set_seed(random_seed_int, device)

    print('tokenizing train data')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print('We have added', num_added_toks, 'tokens')

    _, polarity_train_data = get_dataset(tsvPth_train, tokenizer, args.max_len)
    _, polarity_dev_data = get_dataset(tsvPth_dev, tokenizer, args.max_len)
    
    print('polarity_data: ', len(polarity_train_data))
    print('polarity_dev_count: ', len(polarity_dev_data))
    
    #데이터 스플릿후 테스트
    dataset_size = len(polarity_train_data)
    train_size = int(dataset_size * 0.9)
    validation_size = dataset_size - train_size
    
    print(f"Training Data Size : ",dataset_size)
    print(f"Validation Data Size : ",train_size)
    print(f"Testing Data Size : ",validation_size)
    
    
    train_dataset, val_dataset = random_split(polarity_train_data, [train_size,validation_size])
    
    
    polarity_train_dataloader = DataLoader(train_dataset, shuffle=True,
                                                  batch_size=args.batch_size,num_workers=8,
                                pin_memory=True)
    polarity_dev_dataloader = DataLoader(val_dataset, shuffle=True,
                                                batch_size=args.batch_size,num_workers=8,
                                pin_memory=True)
    
    print('loading model')
    
    polarity_model = ASC_model(args, len(polarity_id_to_name), len(tokenizer))
    polarity_model.to(device)
    
    
    print('end loading')
    
    FULL_FINETUNING = True

    # polarity_model_optimizer_setting
    if FULL_FINETUNING:
        polarity_param_optimizer = list(polarity_model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        polarity_optimizer_grouped_parameters = [
            {'params': [p for n, p in polarity_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in polarity_param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        polarity_param_optimizer = list(polarity_model.classifier.named_parameters())
        polarity_optimizer_grouped_parameters = [{"params": [p for n, p in polarity_param_optimizer]}]

    polarity_optimizer = AdamW(
        polarity_optimizer_grouped_parameters,
        lr=args.learning_rate,
        eps=args.eps
    )
    epochs = args.num_train_epochs
    max_grad_norm = 1.0
    total_steps = epochs * len(polarity_train_dataloader)

    polarity_scheduler = get_linear_schedule_with_warmup(
        polarity_optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    
    print('[Training Phase]')
    epoch_step = 0

    for _ in tqdm(range(epochs), desc="Epoch"):
        epoch_step += 1
    
        polarity_model.train()

        # polarity train
        polarity_total_loss = 0

        for step, batch in enumerate(tqdm(polarity_train_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            polarity_model.zero_grad()

            loss, _ = polarity_model(b_input_ids, b_input_mask, b_labels)

            loss.backward()

            polarity_total_loss += loss.item()
            # print('batch_loss: ', loss.item())

            torch.nn.utils.clip_grad_norm_(parameters=polarity_model.parameters(), max_norm=max_grad_norm)
            polarity_optimizer.step()
            polarity_scheduler.step()

        avg_train_loss = polarity_total_loss / len(polarity_train_dataloader)
        print("Entity_Property_Epoch: ", epoch_step)
        print("Average train loss: {}".format(avg_train_loss))

        model_saved_path = args.polarity_model_path + 'saved_model_epoch_' + str(epoch_step) + '.pt'
        torch.save(polarity_model.state_dict(), model_saved_path)

        if args.do_eval:
            polarity_model.eval()

            pred_list = []
            label_list = []

            for batch in polarity_dev_dataloader:
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                with torch.no_grad():
                    loss, logits = polarity_model(b_input_ids, b_input_mask, b_labels)

                predictions = torch.argmax(logits, dim=-1)
                pred_list.extend(predictions)
                label_list.extend(b_labels)

            evaluation(label_list, pred_list, len(polarity_id_to_name))

        
    print("training is done")
    


    
if __name__ == "__main__":

    if args.do_train:
        train_sentiment_analysis(args)

    


