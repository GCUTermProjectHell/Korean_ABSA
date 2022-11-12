import json
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="ensemble")
    parser.add_argument(
        "--result_path_list", type=str, nargs='+'
    )
    parser.add_argument(
        "--save_file", type=str
    )
    parser.add_argument(
        "--is_final", type=bool, default=False
    )
    args = parser.parse_args()
    return args

def jsonlload(fname, encoding="utf-8"):
    json_list = []
    with open(fname, encoding=encoding) as f:
        for line in f.readlines():
            json_list.append(json.loads(line))
    return json_list

def jsondump(j, fname):
    with open(fname, "w", encoding="UTF8") as f:
        json.dump(j, f, ensure_ascii=False)
    
    
    
args=parse_args()

polarity_id_to_name = ['positive', 'negative', 'neutral']

entity_id_to_name = [
    '제품 전체#품질', '패키지/구성품#디자인', '본품#일반', '제품 전체#편의성', '본품#다양성', '제품 전체#디자인', 
    '패키지/ 구성품#가격', '본품#품질', '브랜드#인지도', '제품 전체#일반', '브랜드#일반', '패키지/구성품#다양성', '패키지/구성품#일반', 
    '본품#인지도', '제품 전체#가격', '본품#편의성', '패키지/구성품#편의성', '본품#디자인', '브랜드#디자인', '본품#가격', '브랜드#품질', 
    '제품 전체#인지도', '패키지/구성품#품질', '제품 전체#다양성', '브랜드#가격']
    
polarity_name_to_id = {polarity_id_to_name[i]: i for i in range(len(polarity_id_to_name))}
entity_name_to_id = {entity_id_to_name[i]: i for i in range(len(entity_id_to_name))}



result=[]
for path in args.result_path_list:
    result.extend(jsonlload(path))

line_length=len(result[0])#test데이터 길이

final_result=dict()
final_result["id"] = ""
final_result["sentence_form"] = ""
final_result["annotation"] = ""

#jsonl 파일 생성
jsonlfile = args.save_file.split(".json")[0] + '.jsonl'

data = []
with open(jsonlfile, "w", encoding="utf-8") as f:
    for i in range(line_length):# 데이터 길이
        entity_count = [0 for i in range(25)]
        polarity_count = [0 for i in range(3)]
        
        for j in range(len(result)):# 모델 개수
            line=result[j][i]
            id=line['id']
            sentence=line['sentence_form']
            annotation=[]

            for k in range(len(line['annotation'])): # entity 각각 count
                entity=line['annotation'][k][0]
                index=entity_name_to_id[entity]
                entity_count[index]+=1

        if not args.is_final:
            required_vote = len(result)/2+1
        else:
            required_vote = 2

        max_count=max(entity_count)
        for m in range(len(entity_count)): # 과반수 이상인 entity 적용
            if entity_count[m] >= (required_vote):
                temp=[]
                temp.append(entity_id_to_name[m])
                temp.append("none")
                annotation.append(temp)

            else: #과반수 넘는 entity 없을 경우, 제일 많은 entity로 적용
                if max_count!=0 and entity_count[m]==max_count:
                    temp=[]
                    temp.append(entity_id_to_name[m])
                    temp.append("none")
                    annotation.append(temp)
                       

        final_result['sentence_form']=sentence
        final_result['id']=id
        final_result['annotation']=annotation
        # print(final_result)
        json.dump(final_result, f, ensure_ascii=False) 
        f.write("\n")

#json파일 생성
data = jsonlload(jsonlfile)
jsondump(data, args.save_file)
os.remove(jsonlfile)

print(args.save_file, " ensemble finished!!")
