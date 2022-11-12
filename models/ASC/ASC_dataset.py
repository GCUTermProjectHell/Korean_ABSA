import torch

import pandas as pd

from utils import jsonlload
from torch.utils.data import TensorDataset



polarity_id_to_name = ['positive', 'negative', 'neutral']
polarity_name_to_id = {polarity_id_to_name[i]: i for i in range(len(polarity_id_to_name))}

entity_property_pair = [
    '제품 전체#품질', '제품 전체#디자인','제품 전체#다양성','제품 전체#인지도','제품 전체#일반','제품 전체#편의성','제품 전체#가격',
    '패키지/구성품#디자인',  '패키지/구성품#가격','패키지/구성품#다양성', '패키지/구성품#일반','패키지/구성품#편의성','패키지/구성품#품질',  
    '본품#일반',  '본품#다양성', '본품#품질', '본품#인지도', '본품#편의성',  '본품#디자인', '본품#가격',
    '브랜드#일반',  '브랜드#인지도', '브랜드#디자인',  '브랜드#품질', '브랜드#가격' ]

label_id_to_name = ['True', 'False']
label_name_to_id = {label_id_to_name[i]: i for i in range(len(label_id_to_name))}

special_tokens_dict = {
    'additional_special_tokens': ['&name&', '&affiliation&', '&social-security-num&', '&tel-num&', '&card-num&', '&bank-account&', '&num&', '&online-account&']
}



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

        sentence = pair.replace("#","의 ")
        tokenized_data = tokenizer(form,sentence, padding='max_length', max_length=max_len, truncation=True)
        for annotation in annotations:
            entity_property = annotation[0]
            polarity = annotation[2]

            if polarity == '------------':
                continue


            if entity_property == pair:
                entity_property_data_dict['input_ids'].append(tokenized_data['input_ids'])
                entity_property_data_dict['attention_mask'].append(tokenized_data['attention_mask'])
                entity_property_data_dict['label'].append(label_name_to_id['True'])

                polarity_data_dict['input_ids'].append(tokenized_data['input_ids'])
                polarity_data_dict['attention_mask'].append(tokenized_data['attention_mask'])
                polarity_data_dict['label'].append(polarity_name_to_id[polarity])
                

                isPairInOpinion = True
                break

        if isPairInOpinion is False:
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


    return TensorDataset(torch.tensor(input_ids_list), torch.tensor(attention_mask_list),
                         torch.tensor(token_labels_list)), TensorDataset(torch.tensor(polarity_input_ids_list), torch.tensor(polarity_attention_mask_list),
                         torch.tensor(polarity_token_labels_list))
