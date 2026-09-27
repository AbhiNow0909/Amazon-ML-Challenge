#!/bin/sh
cd /home/jupyter-1nt23cb058/Amazon-ML-Challenge || exit 1
export PATH=/opt/tljh/user/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONPATH=code/business_entity_resolution/src
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=true
export ENCODER_FEATURE_WORKERS=4
exec python -u -m business_entity_resolution.encoder_pipeline --stage all
