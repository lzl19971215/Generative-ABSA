#!/usr/bin/env bash
EPOCHS="10 15 20 25"
ANNOTATIONS="extraction annotation"
DATASETS="rest15 rest16"
for ds in ${DATASETS};
do
    for ann in ${ANNOTATIONS};
    do
        for epoch in ${EPOCHS};
        do
            rm -rf outputs/
            python main.py --task tasd \
                        --dataset ${ds} \
                        --model_name_or_path google-t5/t5-base \
                        --paradigm ${ann} \
                        --n_gpu 0 \
                        --do_train \
                        --do_direct_eval \
                        --train_batch_size 16 \
                        --gradient_accumulation_steps 2 \
                        --eval_batch_size 16 \
                        --learning_rate 3e-4 \
                        --num_train_epochs ${epoch} 
        done
    done
done

# rm -rf outputs/
# python main.py --task tasd \
#             --dataset rest16 \
#             --model_name_or_path google-t5/t5-base \
#             --paradigm annotation \
#             --n_gpu 0 \
#             --do_train \
#             --do_direct_eval \
#             --train_batch_size 16 \
#             --gradient_accumulation_steps 2 \
#             --eval_batch_size 16 \
#             --learning_rate 3e-4 \
#             --num_train_epochs 30 


