import torch

from utils import jsonlload
from torch.utils.data import TensorDataset


polarity_id_to_name = ['positive', 'negative', 'neutral']
polarity_name_to_id = {polarity_id_to_name[i]: i for i in range(len(polarity_id_to_name))}

entity_property_pair = [
    '제품 전체#품질', '제품 전체#디자인','제품 전체#다양성','제품 전체#인지도','제품 전체#일반','제품 전체#편의성','제품 전체#가격',
    '패키지/구성품#디자인',  '패키지/구성품#가격','패키지/구성품#다양성', '패키지/구성품#일반','패키지/구성품#편의성','패키지/구성품#품질',  
    '본품#일반',  '본품#다양성', '본품#품질', '본품#인지도', '본품#편의성',  '본품#디자인', '본품#가격',
    '브랜드#일반',  '브랜드#인지도', '브랜드#디자인',  '브랜드#품질', '브랜드#가격' ]
entity2str = dict(zip(entity_property_pair, map(lambda x: x.replace("#", " ").replace("/", " "), entity_property_pair)))

label_id_to_name = ['True', 'False']
label_name_to_id = {label_id_to_name[i]: i for i in range(len(label_id_to_name))}

special_tokens_dict = {
    'additional_special_tokens': ['&name&', '&affiliation&', '&social-security-num&', '&tel-num&', '&card-num&', '&bank-account&', '&num&', '&online-account&']
}



def tokenize_and_align_labels(tokenizer, form, annotations):
    
    global polarity_count
    global entity_property_count

    entity_encode_data_dict = {
        'input_ids': [],
        'attention_mask': []
    }
    entity_decode_data_dict = {
        'input_ids': [],
        'attention_mask': []
    }
    
    answer_label = "<pad>"
    sentence = "문장에서 속성을 찾으시오: " + form 
    tokenized_data = tokenizer(sentence, padding='max_length', max_length=130, truncation=True)

    
    for annotation in annotations:
        entity_property = annotation[0]
        answer_label = answer_label+ entity2str[entity_property]+ " "
        
    tokenized_label = tokenizer(answer_label[:-1], padding='max_length', max_length=20, truncation=True)
    
    entity_encode_data_dict['input_ids'].append(tokenized_data['input_ids'])
    entity_encode_data_dict['attention_mask'].append(tokenized_data['attention_mask'])
    
    entity_decode_data_dict['input_ids'].append(tokenized_label['input_ids'])
    entity_decode_data_dict['attention_mask'].append(tokenized_label['attention_mask'])
    
    

    return entity_encode_data_dict, entity_decode_data_dict


def get_dataset(data_path, tokenizer):
    raw_data = jsonlload(data_path)
    input_ids_list = []
    attention_mask_list = []

    decode_input_ids_list = []
    decode_attention_mask_list = []

    for utterance in raw_data:
        entity_encode_data_dict, entity_decode_data_dict = tokenize_and_align_labels(tokenizer, utterance['sentence_form'], utterance['annotation'])
        input_ids_list.extend(entity_encode_data_dict['input_ids'])
        attention_mask_list.extend(entity_encode_data_dict['attention_mask'])

        decode_input_ids_list.extend(entity_decode_data_dict['input_ids'])
        decode_attention_mask_list.extend(entity_decode_data_dict['attention_mask'])


    return TensorDataset(torch.tensor(input_ids_list), torch.tensor(attention_mask_list), torch.tensor(decode_input_ids_list), torch.tensor(decode_attention_mask_list))
