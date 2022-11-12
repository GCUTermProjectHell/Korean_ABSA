#!/bin/bash


echo ensemble1 : gcu_ky2 + gcu_ky1 + GCU_T5_1
python entity_property_ensemble.py \
  --save_file '../saved_result/ensemble/ensemble_1.json'\
  --result_path_list\
    '../saved_result/kyElectra/gcu_ky2.json' \
    '../saved_result/kyElectra/gcu_ky1.json' \
    '../saved_result/T5/GCU_T5_1.json'

echo ensemble2 : gcu_ky2 + gcu_ky1 + GCU_T5_2 + GCU_T5_2 + GCU_T5_3
python entity_property_ensemble.py \
  --save_file '../saved_result/ensemble/ensemble_2.json'\
  --result_path_list\
    '../saved_result/kyElectra/gcu_ky2.json' \
    '../saved_result/kyElectra/gcu_ky1.json' \
    '../saved_result/T5/GCU_T5_1.json' \
    '../saved_result/T5/GCU_T5_2.json' \
    '../saved_result/T5/GCU_T5_3.json' 


echo ensemble3 : ensemble_1 + ensemble_2 + GCU_T5_4 + GCU_T5_5
python entity_property_ensemble.py \
  --save_file '../saved_result/ensemble/ensemble_3.json'\
  --result_path_list\
    '../saved_result/ensemble/ensemble_1.json' \
    '../saved_result/ensemble/ensemble_2.json' \
    '../saved_result/T5/GCU_T5_4.json' \
    '../saved_result/T5/GCU_T5_5.json' 

echo ensemble4 : gcu_ky2 + gcu_ky3 + GCU_T5_1
python entity_property_ensemble.py \
  --save_file '../saved_result/ensemble/ensemble_4.json'\
  --result_path_list\
    '../saved_result/kyElectra/gcu_ky1.json' \
    '../saved_result/kyElectra/gcu_ky3.json' \
    '../saved_result/kyElectra/gcu_ky4.json' \
    '../saved_result/kyElectra/gcu_ky5.json' \
    '../saved_result/T5/GCU_T5_1.json' 


echo Final : ensemble_3 + ensemble_4 + GCU_T5_6
python entity_property_ensemble.py \
  --save_file '../saved_result/ensemble/final.json'\
  --result_path_list\
    '../saved_result/ensemble/ensemble_3.json' \
    '../saved_result/ensemble/ensemble_4.json' \
    '../saved_result/T5/GCU_T5_6.json' \
  --is_final True




