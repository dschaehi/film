#!/bin/bash
date
echo "=========="
python scripts/run_model.py \
    --sw_name existential \
    --sw_variant test \
    --sw_config ~/test \
    --sw_mode test \
    --program_generator ~/test/test.ckpt \
    --execution_engine ~/test/test.ckpt
python scripts/run_model.py \
    --sw_name existential \
    --sw_variant test \
    --sw_config ~/test \
    --sw_mode train \
    --program_generator ~/test/test.ckpt \
    --execution_engine ~/test/test.ckpt
python scripts/run_model.py \
    --sw_name existential \
    --sw_variant test2 \
    --sw_config ~/test \
    --program_generator ~/test/test.ckpt \
    --execution_engine ~/test/test.ckpt
python scripts/run_model.py \
    --sw_name existential \
    --sw_variant test2 \
    --sw_config ~/test \
    --sw_mode none \
    --program_generator ~/test/test.ckpt \
    --execution_engine ~/test/test.ckpt
python scripts/run_model.py \
    --sw_name existential \
    --sw_config ~/ShapeWorld/configs/agreement/existential/oneshape.json \
    --num_samples 100 \
    --program_generator ~/test/test.ckpt \
    --execution_engine ~/test/test.ckpt
echo "=========="
date
