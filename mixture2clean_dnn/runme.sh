#!/bin/bash

MINIDATA=1
if [ $MINIDATA -eq 1 ]; then
  WORKSPACE="workspace"
  mkdir $WORKSPACE
  TR_SPEECH_DIR="mini_data/train_speech"
  TR_NOISE_DIR="mini_data/train_noise"
  TE_SPEECH_DIR="mini_data/test_speech"
  TE_NOISE_DIR="mini_data/test_noise"
  echo "Using mini data. "
else
  WORKSPACE="/vol/vssp/msos/qk/workspaces/speech_enhancement"
  TR_SPEECH_DIR="/vol/vssp/msos/qk/workspaces/speech_enhancement/timit_wavs/train"
  TR_NOISE_DIR="/vol/vssp/msos/qk/workspaces/speech_enhancement/nosie_wavs/train"
  TE_SPEECH_DIR="/vol/vssp/msos/qk/workspaces/speech_enhancement/timit_wavs/subtest"
  TE_NOISE_DIR="/vol/vssp/msos/qk/workspaces/speech_enhancement/nosie_wavs/test"
  echo "Using full data. "
fi

# Create mixture csv. 
python3.6 prepare_data.py create_mixture_csv --workspace=$WORKSPACE --speech_dir=$TR_SPEECH_DIR --noise_dir=$TR_NOISE_DIR --data_type=train --magnification=2
python3.6 prepare_data.py create_mixture_csv --workspace=$WORKSPACE --speech_dir=$TE_SPEECH_DIR --noise_dir=$TE_NOISE_DIR --data_type=test

# Calculate mixture features.
TR_SNR=0
TE_SNR=0 
python3.6 prepare_data.py calculate_mixture_features --workspace=$WORKSPACE --speech_dir=$TR_SPEECH_DIR --noise_dir=$TR_NOISE_DIR --data_type=train --snr=$TR_SNR
python3.6 prepare_data.py calculate_mixture_features --workspace=$WORKSPACE --speech_dir=$TE_SPEECH_DIR --noise_dir=$TE_NOISE_DIR --data_type=test --snr=$TE_SNR

# Pack features. 
N_CONCAT=7
N_HOP=3
python3.6 prepare_data.py pack_features --workspace=$WORKSPACE --data_type=train --snr=$TR_SNR --n_concat=$N_CONCAT --n_hop=$N_HOP
python3.6 prepare_data.py pack_features --workspace=$WORKSPACE --data_type=test --snr=$TE_SNR --n_concat=$N_CONCAT --n_hop=$N_HOP

# Compute scaler. 
python3.6 prepare_data.py compute_scaler --workspace=$WORKSPACE --data_type=train --snr=$TR_SNR



# Train. 
LEARNING_RATE=1e-4
python3.6 main_dnn.py train --workspace=$WORKSPACE --tr_snr=$TR_SNR --te_snr=$TE_SNR --lr=$LEARNING_RATE

# Plot training stat. 
python3.6 evaluate.py plot_training_stat --workspace=$WORKSPACE --tr_snr=$TR_SNR --bgn_iter=0 --fin_iter=10001 --interval_iter=1000

# Inference, enhanced wavs will be created. 
ITERATION=5000
CUDA_VISIBLE_DEVICES=3 python3.6 main_dnn.py inference --workspace=$WORKSPACE --tr_snr=$TR_SNR --te_snr=$TE_SNR --n_concat=$N_CONCAT --iteration=$ITERATION

# Calculate PESQ of all enhanced speech. 
python3.6 evaluate.py calculate_pesq --workspace=$WORKSPACE --speech_dir=$TE_SPEECH_DIR --te_snr=$TE_SNR

# Calculate overall stats. 
python3.6 evaluate.py get_stats

