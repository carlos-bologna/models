#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script is used to run local test on PASCAL VOC 2012. Users could also
# modify from this script for their use case.
#
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   sh ./local_test.sh
#
#

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Steps to run:
STEP_1=1 #preprocess
STEP_2=0 #convert images with build_voc2012_data
STEP_3=0 # download pre-treined model from internet
STEP_4=0 # train
STEP_5=0 # val
STEP_6=0 # viz
STEP_7=0 # export model

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"

# Preprocess MESSIDOR images
DATASET_DIR="datasets"
MESSIDOR_FOLDER="MESSIDOR"
MESSIDOR_EXT="MESSIDOR_EXT"

if [ ${STEP_1} -eq 1 ]
then

  python "${WORK_DIR}"/preprocess.py \
    --fullpath_origin="${WORK_DIR}/${DATASET_DIR}/${MESSIDOR_FOLDER}" \
    --fullpath_JPEGDestination="${WORK_DIR}/${DATASET_DIR}/JPEGImages" \
    --fullpath_destination="${WORK_DIR}/${DATASET_DIR}/SegmentationClass" \
    --fullpath_train_val_list="${WORK_DIR}/${DATASET_DIR}/Splits"

  # Remove the colormap in the ground truth annotations.
  SEG_FOLDER="${WORK_DIR}/${DATASET_DIR}/SegmentationClass"
  SEMANTIC_SEG_FOLDER="${WORK_DIR}/${DATASET_DIR}/SegmentationClassRaw"

  if [ -d "$SEMANTIC_SEG_FOLDER" ]; then
    rm -r "${SEMANTIC_SEG_FOLDER}"
  fi

  echo "Removing the color map in ground truth annotations..."
  python "${WORK_DIR}"/datasets/remove_gt_colormap.py \
    --original_gt_folder="${SEG_FOLDER}" \
    --output_dir="${SEMANTIC_SEG_FOLDER}"

fi

# Convert Images
MESSIDOR_DATASET="${WORK_DIR}/${DATASET_DIR}/tfrecord"
mkdir -p "${MESSIDOR_DATASET}"

if [ ${STEP_2} -eq 1 ]
then
  python "${WORK_DIR}"/datasets/build_voc2012_data.py \
    --image_folder="${WORK_DIR}/${DATASET_DIR}/JPEGImages" \
    --semantic_segmentation_folder="${WORK_DIR}/${DATASET_DIR}/SegmentationClassRaw" \
    --list_folder="${WORK_DIR}/${DATASET_DIR}/Splits" \
    --image_format="jpg" \
    --output_dir="${MESSIDOR_DATASET}"
fi

# Set up the working directories.
EXP_FOLDER="exp/train_on_trainval_set"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${MESSIDOR_EXT}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${MESSIDOR_EXT}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${MESSIDOR_EXT}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${MESSIDOR_EXT}/${EXP_FOLDER}/export"
mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

# Copy locally the trained checkpoint as the initial checkpoint.
TF_INIT_ROOT="http://download.tensorflow.org/models"
TF_INIT_CKPT="deeplabv3_pascal_train_aug_2018_01_04.tar.gz"
cd "${INIT_FOLDER}"
if [ ${STEP_3} -eq 1 ]
then
  wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
  tar -xf "${TF_INIT_CKPT}"
fi

cd "${CURRENT_DIR}"

# Train 10 iterations.
NUM_ITERATIONS=10

if [ ${STEP_4} -eq 1 ]
then
  python "${WORK_DIR}"/train.py \
    --logtostderr \
    --train_split="train" \
    --dataset="messidor" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --train_crop_size="513,513" \
    --train_batch_size=4 \
    --training_number_of_steps="${NUM_ITERATIONS}" \
    --fine_tune_batch_norm=true \
    --tf_initial_checkpoint="${INIT_FOLDER}/deeplabv3_pascal_train_aug/model.ckpt" \
    --train_logdir="${TRAIN_LOGDIR}" \
    --dataset_dir="${MESSIDOR_DATASET}"
fi

# Run evaluation. This performs eval over the full val split  and
# will take a while.
# Using the provided checkpoint, one should expect mIOU=82.20%.

if [ ${STEP_5} -eq 1 ]
then

  python "${WORK_DIR}"/eval.py \
    --logtostderr \
    --eval_split="val" \
    --dataset="messidor" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --eval_crop_size="513,513" \
    --checkpoint_dir="${TRAIN_LOGDIR}" \
    --eval_logdir="${EVAL_LOGDIR}" \
    --dataset_dir="${MESSIDOR_DATASET}" \
    --max_number_of_evaluations=1
fi

## Visualize the results.

if [ ${STEP_6} -eq 1 ]
then

  python "${WORK_DIR}"/vis.py \
    --logtostderr \
    --vis_split="val" \
    --dataset="messidor" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --vis_crop_size="513,513" \
    --checkpoint_dir="${TRAIN_LOGDIR}" \
    --vis_logdir="${VIS_LOGDIR}" \
    --dataset_dir="${PASCAL_DATASET}" \
    --max_number_of_iterations=1

fi

## Export the trained checkpoint.
CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"

if [ ${STEP_7} -eq 1 ]
then

  python "${WORK_DIR}"/export_model.py \
    --logtostderr \
    --checkpoint_path="${CKPT_PATH}" \
    --export_path="${EXPORT_PATH}" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --num_classes=21 \
    --crop_size=513 \
    --crop_size=513 \
    --inference_scales=1.0
fi

## Run inference with the exported checkpoint.
## Please refer to the provided deeplab_demo.ipynb for an example.
