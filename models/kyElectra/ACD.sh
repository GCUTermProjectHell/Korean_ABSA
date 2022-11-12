#!/bin/bash

# Train kyelectra
cd src
echo run task:pipeline_entity_property !!
python ky_ACD_pipeline.py \
  --base_model kykim/electra-kor-base \
  --do_train \
  --do_eval \
  --train_data '../../../dataset/'
  --learning_rate 5e-6 \
  --eps 9e-9 \
  --num_train_epochs 30 \
  --entity_property_model_path ../../../saved_result/kyElectra/ \
  --batch_size 64 \
  --max_len 256\




