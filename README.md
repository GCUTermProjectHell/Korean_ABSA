# Korean_ABSA
2022 국립국어원 인공 지능 언어 능력 평가 Team GCU_텀프지옥입니다

# 대회 개요
과제에서 사용하는 말뭉치는 국립국어원이 구축한 '속성 기반 감성 분석' 말뭉치입니다. 속성 기반 감성 분석이란 언어에 나타난 개체의 속성 정보에 대한극성을 분류하는 과제입니다. 참가팀은 입력 문장에 대해 (1) 속성범주 (예: 제춤 전체#인지도)와 (2) 감성 (긍정/부정/중립)의 쌍을 추출하고, 정답 튜플과 예측된 튜플과의 비교를 통해 계산된 F1점수로 참가팀의 인공지능 모델의 성능을 평가합니다.

# 최종 Inference 방법
### 1. Environment Setting
  <code>
      pip install -r requirements.txt
  </code>
    
    
    


### 2. 평가용 데이터 삽입

  dataset/taskABSA/ 폴더안에 nikluge-sa-2022-test.jsonl 파일의 형태로 삽입
  
  
### 3. 모델 다운로드

  폴더 경로 내에 saved_models안에 모델을 저장
  
  
  다운 경로: https://gachonackr-my.sharepoint.com/:f:/g/personal/teryas_gachon_ac_kr/EiGJJYKoC7NBs7qj_V6chkIBcG3PUtZvOKhspakbs7WCDQ?e=ogBSAy
  
  
  
### 4. Inference 코드 실행
  <code>
      bash.All.sh
  </code>         쉘 스크립트 실행

### 5. 평가가 완료되면 final.jsonl 형태로 최종 저장된다.



# 소스코드 파일 설명

![image](https://user-images.githubusercontent.com/87708360/201470860-a3916267-56ff-4ff7-99fd-fa64b4a9a140.png)
