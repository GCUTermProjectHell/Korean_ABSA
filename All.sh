# install requirements.txt
pip install -r requirements.txt

#T5 모델 inference
cd models/T5
echo T5Model v1
bash v1.sh \

echo T5Model v2
bash v2.sh \

# kyElectra 모델 inference
cd ../kyElectra
echo KyElectra
bash inference.sh \

# ACD 앙상블
cd ../../ensemble
echo 앙상블 시작
bash ensemble.sh \

# ASC inference하여 최종 제출파일 생성
cd ../models/ASC
echo ASC 추출
bash pola.sh \

