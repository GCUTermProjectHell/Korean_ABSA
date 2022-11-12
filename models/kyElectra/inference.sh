#!/bin/bash

# Inference kyelectra
cd src
echo run task: Inference entity_property model  !!
python ky_ACD.py \
  --base_model kykim/electra-kor-base\
  --do_test \
  --entity_property_model_path '../../../saved_model/kyElectra/gcu_ky1.pt' \
  --save_path '../../../saved_result/kyElectra/gcu_ky1.json'\
  --classifier_hidden_size 768\

echo run task: Inference entity_property model  !!
python ky_ACD.py \
  --base_model kykim/electra-kor-base\
  --do_test \
  --entity_property_model_path '../../../saved_model/kyElectra/gcu_ky2.pt' \
  --save_path '../../../saved_result/kyElectra/gcu_ky2.json'\
  --classifier_hidden_size 768\

echo run task: Inference entity_property model  !!
python ky_ACD.py \
  --base_model kykim/electra-kor-base\
  --do_test \
  --entity_property_model_path '../../../saved_model/kyElectra/gcu_ky3.pt' \
  --save_path '../../../saved_result/kyElectra/gcu_ky3.json'\
  --classifier_hidden_size 768\

echo run task: Inference entity_property model  !!
python ky_ACD.py \
  --base_model kykim/electra-kor-base\
  --do_test \
  --entity_property_model_path '../../../saved_model/kyElectra/gcu_ky4.pt' \
  --save_path '../../../saved_result/kyElectra/gcu_ky4.json'\
  --classifier_hidden_size 768\

echo run task: Inference entity_property model  !!
python ky_ACD.py \
  --base_model kykim/electra-kor-base\
  --do_test \
  --entity_property_model_path '../../../saved_model/kyElectra/gcu_ky5.pt' \
  --save_path '../../../saved_result/kyElectra/gcu_ky5.json'\
  --classifier_hidden_size 768\

