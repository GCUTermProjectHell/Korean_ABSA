#기존 모델과 다른 방식으로 대회 제출형태로 만들어주기위한 파일

import torch

from utils import jsonlload, parse_args,jsondump

from ACD_dataset import special_tokens_dict

import copy

from ACD_models import ACD_model

from transformers import AutoTokenizer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


label_id_to_name = ['True', 'False']


def predict_cd(tokenizer, CD_model, data):
    
    #Encoder- Decoder모델 에서 생성하는 단어를 대회 제출형태로 만들기 위해 생성되는 것들의 카테고리를 나열
    ACD_pair = [
        '제품 전체 품질', '제품 전체 디자인','제품 전체 다양성','제품 전체 인지도','제품 전체 일반','제품 전체 편의성','제품 전체 가격',
        '패키지 구성품 디자인',  '패키지 구성품 가격','패키지 구성품 다양성', '패키지 구성품 일반','패키지 구성품 편의성','패키지 구성품 품질',  
        '본품 일반',  '본품 다양성', '본품 품질', '본품 인지도', '본품 편의성',  '본품 디자인', '본품 가격',
        '브랜드 일반',  '브랜드 인지도', '브랜드 디자인',  '브랜드 품질', '브랜드 가격' ]   
    
    for sentence in data:
        form = sentence['sentence_form']       
        sentence['annotation'] = []
        
        sentences = "문장에서 속성을 찾으시오: " + form + " 이문장의 속성은 <extra_id_0> <extra_id_1> <extra_id_2> <extra_id_3><extra_id_4>이다"
        tokenized_data = tokenizer(sentences, padding='max_length', max_length=255, truncation=True)



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
    GCU_T5_4_Path = "../../../saved_model/T5/GCU_T5_4.pt"
    GCU_T5_5_Path = "../../../saved_model/T5/GCU_T5_5.pt"
    GCU_T5_6_Path = "../../../saved_model/T5/GCU_T5_6.pt"
    tsvPth_test = "../../../dataset/task_ABSA/nikluge-sa-2022-test.jsonl"
    
    #토크나이저를 불러와서 스페셜 토큰 형태로 추가
    tokenizer = AutoTokenizer.from_pretrained('paust/pko-t5-large')
    tokenizer.add_special_tokens(special_tokens_dict)
     
    test_data = jsonlload(tsvPth_test)
            
    GCU_T5_4 = ACD_model(args, len(label_id_to_name), len(tokenizer))
    GCU_T5_4.load_state_dict(torch.load(GCU_T5_4_Path, map_location=device))
    GCU_T5_4.to(device)
    GCU_T5_4.eval()
    
    
    pred_data = predict_cd(tokenizer, GCU_T5_4, copy.deepcopy(test_data))
    #ac8 = predict_pola(tokenizer_pola, polarity_model, copy.deepcopy(ac8))
    print("GCU_T5_4 출력중")
    jsondump(pred_data, '../../../saved_result/T5/GCU_T5_4.json')
    
    GCU_T5_5 = ACD_model(args, len(label_id_to_name), len(tokenizer))
    GCU_T5_5.load_state_dict(torch.load(GCU_T5_5_Path, map_location=device))
    GCU_T5_5.to(device)
    GCU_T5_5.eval()
    
    pred_data = predict_cd(tokenizer, GCU_T5_5, copy.deepcopy(test_data))
    #ac10 = predict_pola(tokenizer_pola, polarity_model, copy.deepcopy(ac8))
    print("GCU_T5_5 출력중")
    jsondump(pred_data, '../../../saved_result/T5/GCU_T5_5.json')
    
    GCU_T5_6 = ACD_model(args, len(label_id_to_name), len(tokenizer))
    GCU_T5_6.load_state_dict(torch.load(GCU_T5_6_Path, map_location=device))
    GCU_T5_6.to(device)
    GCU_T5_6.eval()
    
    pred_data = predict_cd(tokenizer, GCU_T5_6, copy.deepcopy(test_data))
    #ac11 = predict_pola(tokenizer_pola, polarity_model, copy.deepcopy(ac8))
    print("GCU_T5_6 출력중")
    jsondump(pred_data, '../../../saved_result/T5/GCU_T5_6.json')

if __name__ == "__main__":
    args = parse_args()

    if args.do_test:
        test_sentiment_analysis(args)
        
    