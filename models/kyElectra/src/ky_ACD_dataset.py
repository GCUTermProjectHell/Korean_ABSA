import torch
from torch.utils.data import Dataset 

import pandas as pd
import os

from utils import MODEL_CLASSES, MODEL_PATH_MAP 
from utils import jsonlload
from torch.utils.data import TensorDataset

model_name = 'kykim/electra-kor-base' #'kobert', 'roberta-base', 'koelectra', 'koelectra_QA', 'koelectra_tunib'
_1, _2, model_tokenizer = MODEL_CLASSES[model_name] #config_class, model_class, model_tokenizer
model_path = MODEL_PATH_MAP[model_name]


polarity_id_to_name = ['positive', 'negative', 'neutral']
polarity_name_to_id = {polarity_id_to_name[i]: i for i in range(len(polarity_id_to_name))}

entity_property_pair = [
    '제품 전체#품질', '패키지/구성품#디자인', '본품#일반', '제품 전체#편의성', '본품#다양성', '제품 전체#디자인', 
    '패키지/ 구성품#가격', '본품#품질', '브랜드#인지도', '제품 전체#일반', '브랜드#일반', '패키지/구성품#다양성', '패키지/구성품#일반', 
    '본품#인지도', '제품 전체#가격', '본품#편의성', '패키지/구성품#편의성', '본품#디자인', '브랜드#디자인', '본품#가격', '브랜드#품질', 
    '제품 전체#인지도', '패키지/구성품#품질', '제품 전체#다양성', '브랜드#가격']


label_id_to_name = ['True', 'False']
label_name_to_id = {label_id_to_name[i]: i for i in range(len(label_id_to_name))}

special_tokens_dict = {
    'additional_special_tokens': ['&name&', '&affiliation&', '&social-security-num&', '&tel-num&', '&card-num&', '&bank-account&', '&num&', '&online-account&']
}
polarity_count = 0
entity_property_count = 0


class ABSA_dataset_Abandon(Dataset): 
    def __init__(self, data_filename,tokenizer): 
        super(ABSA_dataset_Abandon,self).__init__()
        self.data = jsonlload(os.path.join(os.getcwd(),data_filename))
        self.tokenizer = tokenizer
    
    def __len__(self): 
        return len(self.data)
        
    def __getitem__(self, item): 
        data = self.data
        input_ids_list = []
        attention_mask_list = []
        token_labels_list = []

        polarity_input_ids_list = []
        polarity_attention_mask_list = []
        polarity_token_labels_list = []

        for utterance in data:
            entity_property_data_dict, polarity_data_dict = tokenize_and_align_labels(self.tokenizer, utterance['sentence_form'], utterance['annotation'], 256)
            input_ids_list.extend(entity_property_data_dict['input_ids'])
            attention_mask_list.extend(entity_property_data_dict['attention_mask'])
            token_labels_list.extend(entity_property_data_dict['label'])

            polarity_input_ids_list.extend(polarity_data_dict['input_ids'])
            polarity_attention_mask_list.extend(polarity_data_dict['attention_mask'])
            polarity_token_labels_list.extend(polarity_data_dict['label'])

        print('polarity_data_count: ', polarity_count)
        print('entity_property_data_count: ', entity_property_count)

        return entity_property_data_dict, polarity_data_dict
                            

def tokenize_and_align_labels(tokenizer, form, annotations, max_len):
    
    global polarity_count
    global entity_property_count

    entity_property_data_dict = {
        'input_ids': [],
        'attention_mask': [],
        'label': []
    }
    polarity_data_dict = {
        'input_ids': [],
        'attention_mask': [],
        'label': []
    }

    for pair in entity_property_pair:
        isPairInOpinion = False
        if pd.isna(form):
            break
        tokenized_data = tokenizer(form, pair, padding='max_length', max_length=max_len, truncation=True)
        for annotation in annotations:
            entity_property = annotation[0]
            polarity = annotation[2]

            if polarity == '------------':
                continue


            if entity_property == pair:
                
                polarity_count += 1
                entity_property_data_dict['input_ids'].append(tokenized_data['input_ids'])
                entity_property_data_dict['attention_mask'].append(tokenized_data['attention_mask'])
                entity_property_data_dict['label'].append(label_name_to_id['True'])

                polarity_data_dict['input_ids'].append(tokenized_data['input_ids'])
                polarity_data_dict['attention_mask'].append(tokenized_data['attention_mask'])
                if polarity not in polarity_name_to_id:
                    print(polarity)
                    print(form)
                polarity_data_dict['label'].append(polarity_name_to_id[polarity])
                
                isPairInOpinion = True
                break

        if isPairInOpinion is False:
            entity_property_count += 1
            entity_property_data_dict['input_ids'].append(tokenized_data['input_ids'])
            entity_property_data_dict['attention_mask'].append(tokenized_data['attention_mask'])
            entity_property_data_dict['label'].append(label_name_to_id['False'])

    return entity_property_data_dict, polarity_data_dict


def get_dataset(data_path, tokenizer, max_len):
    raw_data = jsonlload(data_path)
    input_ids_list = []
    attention_mask_list = []
    token_labels_list = []

    polarity_input_ids_list = []
    polarity_attention_mask_list = []
    polarity_token_labels_list = []

    for utterance in raw_data:
        entity_property_data_dict, polarity_data_dict = tokenize_and_align_labels(tokenizer, utterance['sentence_form'], utterance['annotation'], max_len)
        input_ids_list.extend(entity_property_data_dict['input_ids'])
        attention_mask_list.extend(entity_property_data_dict['attention_mask'])
        token_labels_list.extend(entity_property_data_dict['label'])

        polarity_input_ids_list.extend(polarity_data_dict['input_ids'])
        polarity_attention_mask_list.extend(polarity_data_dict['attention_mask'])
        polarity_token_labels_list.extend(polarity_data_dict['label'])

    print('polarity_data_count: ', polarity_count)
    print('entity_property_data_count: ', entity_property_count)

    return TensorDataset(torch.tensor(input_ids_list), torch.tensor(attention_mask_list),
                         torch.tensor(token_labels_list)), TensorDataset(torch.tensor(polarity_input_ids_list), torch.tensor(polarity_attention_mask_list),
                         torch.tensor(polarity_token_labels_list))
