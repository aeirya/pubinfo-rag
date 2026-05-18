#!/bin/bash

python run.py \
    --query "for the article 'Exploring Dimensions of Trauma-Linked Somatic Complaints in Refugees: A Qualitative Study in Canada' the authors are: " \
    --template rag \
    --limit 0 \
    --num-predict 10
    