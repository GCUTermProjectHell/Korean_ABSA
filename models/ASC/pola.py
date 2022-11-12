import torch

import os
import copy


from utils import jsonload, parse_args
from utils import MODEL_PATH_MAP
from utils import getParentPath, DATASET_PATHS 
from utils import MODEL_PATH_MAP ,jsonldump


from ASC_dataset import polarity_id_to_name, special_tokens_dict
from ASC_models import ASC_model

from transformers import AutoTokenizer



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_name = 'kyelectra'

task_name = 'ABSA' 
taskDir_path, fname_train, fname_dev, fname_test,  = DATASET_PATHS[task_name]

model_path = MODEL_PATH_MAP[model_name]
polarity_id_to_name = ['positive', 'negative', 'neutral']



def predict_from_korean_form(tokenizer, pc_model, data):
    
    count = 0
    for sentence in data:
        form = sentence['sentence_form']
        annotations = sentence['annotation']         
        sentence['annotation'] = []
        count += 1
        if type(form) != str:
            print("form type is wrong: ", form)
            continue
        for annotation in annotations:
            pair = annotation[0]
            sentence2 = pair.replace("#","의 ")
            tokenized_data = tokenizer(form, sentence2, padding='max_length', max_length=256, truncation=True)
            input_ids = torch.tensor([tokenized_data['input_ids']]).to(device)
            attention_mask = torch.tensor([tokenized_data['attention_mask']]).to(device)
            pc_model.to(device)
            pc_model.eval()
            with torch.no_grad():
                _, pc_logits = pc_model(input_ids, attention_mask)

                pc_predictions = torch.argmax(pc_logits, dim=-1)
                pc_result = polarity_id_to_name[pc_predictions[0]]
                sentence['annotation'].append([pair, pc_result])


    return data
        
def test_sentiment_analysis(args):
    homePth = getParentPath(os.getcwd())
    print('homePth:',homePth,', curPth:',os.getcwd())
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    Final = jsonload("../../saved_result/ensemble/final.json")
            
    polarity_model = ASC_model(args, len(polarity_id_to_name), len(tokenizer))
    polarity_model.load_state_dict(torch.load("../../saved_model/ASC/ASC.pt", map_location=device))
    polarity_model.to(device)
    polarity_model.eval()
    
    
    pred_data = predict_from_korean_form(tokenizer, polarity_model, copy.deepcopy(Final))
    print("ASC 출력중")
    jsonldump(pred_data, '../../final.jsonl')
    
    


if __name__ == "__main__":
    args = parse_args()
    args.base_model = model_path

    if args.do_test:
        test_sentiment_analysis(args)
    