#!/bin/bash

# Inference kyelectra
cd src
echo run task: Inference entity_property model  !!
python ky_ACD.py \
  --base_model kimy1119/gcu_ky1\
  --do_test \
  --save_path '../../../saved_result/kyElectra/gcu_ky1.json'\
  --classifier_hidden_size 768\

echo run task: Inference entity_property model  !!
python ky_ACD.py \
  --base_model kimy1119/gcu_ky2\
  --do_test \
  --save_path '../../../saved_result/kyElectra/gcu_ky2.json'\
  --classifier_hidden_size 768\

echo run task: Inference entity_property model  !!
python ky_ACD.py \
  --base_model kimy1119/gcu_ky3\
  --do_test \
  --save_path '../../../saved_result/kyElectra/gcu_ky3.json'\
  --classifier_hidden_size 768\

echo run task: Inference entity_property model  !!
python ky_ACD.py \
  --base_model kimy1119/gcu_ky4\
  --do_test \
  --save_path '../../../saved_result/kyElectra/gcu_ky4.json'\
  --classifier_hidden_size 768\

echo run task: Inference entity_property model  !!
python ky_ACD.py \
  --base_model kimy1119/gcu_ky5\
  --do_test \
  --save_path '../../../saved_result/kyElectra/gcu_ky5.json'\
  --classifier_hidden_size 768\

