#!/bin/bash

#. conda/bin/activate
#conda activate [가상환경 이름:team#] #임시(lss)

cd Model_V2
echo Model_V2 추출  !!
python ACD.py \
  --do_test \
  --classifier_hidden_size 768 \


