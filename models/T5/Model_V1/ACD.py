import torch


from utils import jsonlload, parse_args,jsondump
from ACD_dataset import special_tokens_dict

import copy

from ACD_models import ACD_model

from transformers import AutoTokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

task_name = 'ABSA' 

label_id_to_name = ['True', 'False']



def predict_from_korean_form(tokenizer, CD_model, data):
    
    ACD_pair = [
        '제품 전체 품질', '제품 전체 디자인','제품 전체 다양성','제품 전체 인지도','제품 전체 일반','제품 전체 편의성','제품 전체 가격',
        '패키지 구성품 디자인',  '패키지 구성품 가격','패키지 구성품 다양성', '패키지 구성품 일반','패키지 구성품 편의성','패키지 구성품 품질',  
        '본품 일반',  '본품 다양성', '본품 품질', '본품 인지도', '본품 편의성',  '본품 디자인', '본품 가격',
        '브랜드 일반',  '브랜드 인지도', '브랜드 디자인',  '브랜드 품질', '브랜드 가격' ]   
    
    for sentence in data:
        form = sentence['sentence_form']       
        sentence['annotation'] = []
        
        sentences = "문장에서 속성을 찾으시오: " + form 
        tokenized_data = tokenizer(sentences, padding='max_length', max_length=130, truncation=True)



        input_ids = torch.tensor([tokenized_data['input_ids']]).to(device)
        attention_mask = torch.tensor([tokenized_data['attention_mask']]).to(device)
        #생성 모델 생성
        outputs = CD_model.model_PLM.generate(
            input_ids=input_ids,
            attention_mask=attention_mask)
        
        pred_t = [tokenizer.decode(g, skip_special_tokens=True) for g in outputs]
        for pred in pred_t:
            pred_l = []
            for i in ACD_pair:
                if i in pred:
                    if '제품 전체' in i:
                        k = i.replace("체 ","체#")
                        pred_l.append(k)
                    elif '패키지 구성품' in i:
                        k = i.replace("패키지 구성품 ","패키지/구성품#")
                        pred_l.append(k)
                    else:
                        k =i.replace(" ","#")
                        pred_l.append(k)
            for i in pred_l:
                sentence['annotation'].append([i,'none'])

    return data
        
def test_sentiment_analysis(args):
    
    #inference할 모델을 불러오는 경로
    GCU_T5_1_Path = "../../../saved_model/T5/GCU_T5_1.pt"
    GCU_T5_2_Path = "../../../saved_model/T5/GCU_T5_2.pt"
    GCU_T5_3_Path = "../../../saved_model/T5/GCU_T5_3.pt"
    tsvPth_test = "../../../dataset/task_ABSA/nikluge-sa-2022-test.jsonl"
    
    #토크나이저를 불러와서 스페셜 토큰 형태로 추가
    tokenizer = AutoTokenizer.from_pretrained('paust/pko-t5-large')
    tokenizer.add_special_tokens(special_tokens_dict)
     
    test_data = jsonlload(tsvPth_test)
            
    GCU_T5_1 = ACD_model(args, len(label_id_to_name), len(tokenizer))
    GCU_T5_1.load_state_dict(torch.load(GCU_T5_1_Path, map_location=device))
    GCU_T5_1.to(device)
    GCU_T5_1.eval()
    
    
    pred_data = predict_from_korean_form(tokenizer, GCU_T5_1, copy.deepcopy(test_data))
    print("GCU_T5_1 출력중")
    jsondump(pred_data, '../../../saved_result/T5/GCU_T5_1.json')
    
    GCU_T5_2 = ACD_model(args, len(label_id_to_name), len(tokenizer))
    GCU_T5_2.load_state_dict(torch.load(GCU_T5_2_Path, map_location=device))
    GCU_T5_2.to(device)
    GCU_T5_2.eval()
    
    
    pred_data = predict_from_korean_form(tokenizer, GCU_T5_2, copy.deepcopy(test_data))
    print("GCU_T5_2 출력중")
    jsondump(pred_data, '../../../saved_result/T5/GCU_T5_2.json')
    
    GCU_T5_3 = ACD_model(args, len(label_id_to_name), len(tokenizer))
    GCU_T5_3.load_state_dict(torch.load(GCU_T5_3_Path, map_location=device))
    GCU_T5_3.to(device)
    GCU_T5_3.eval()
    
    
    pred_data = predict_from_korean_form(tokenizer, GCU_T5_3, copy.deepcopy(test_data))
    print("GCU_T5_3 출력중")
    jsondump(pred_data, '../../../saved_result/T5/GCU_T5_3.json')
    
    


if __name__ == "__main__":
    args = parse_args()

    if args.do_test:
        test_sentiment_analysis(args)
    