# Custom Models

The best approach was the follow model:
* xception_65
* 30.000 iterations
* Batch size of 2
* fine_tune_batch_norm=false
* Learning Rate = 0.00001
* At feature_extractor.py:
  * 'xception_65': \_preprocess_zero_mean_unit_range,

The total los was 0.382

Below is the code to reproduce that:

```
MODEL_VARIANT="xception_65"
BACKBONE_MODEL="deeplabv3_xception_2018_01_04/xception"
NUM_ITERATIONS=30000

python "${WORK_DIR}"/train.py \
  --logtostderr \
  --train_split="train" \
  --dataset="messidor" \
  --model_variant="${MODEL_VARIANT}" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --train_crop_size="513,513" \
  --train_batch_size=2 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=false \
  --base_learning_rate=0.00001 \
  --tf_initial_checkpoint="${INIT_FOLDER}/${BACKBONE_MODEL}/model.ckpt" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${MESSIDOR_DATASET}"
```


### Other attempt but without success

* xception_65
* 10.000 iterations
* Batch size of 2
* fine_tune_batch_norm=true
* Learning Rate = 0.0001
* At feature_extractor.py:
  * \_MEAN_RGB = [104.54, 107.17, 111.68]
  * 'xception_65': \_preprocess_subtract_imagenet_mean,

The total los was 0.471


This was the command at train.sh:

```
MODEL_VARIANT="xception_65"
BACKBONE_MODEL="deeplabv3_xception_2018_01_04/xception"
NUM_ITERATIONS=10000

python "${WORK_DIR}"/train.py \
  --logtostderr \
  --train_split="train" \
  --dataset="messidor" \
  --model_variant="${MODEL_VARIANT}" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --train_crop_size="513,513" \
  --train_batch_size=2 \
  --training_number_of_steps="${NUM_ITERATIONS}" \
  --fine_tune_batch_norm=true \
  --base_learning_rate=0.0001 \
  --tf_initial_checkpoint="${INIT_FOLDER}/${BACKBONE_MODEL}/model.ckpt" \
  --train_logdir="${TRAIN_LOGDIR}" \
  --dataset_dir="${MESSIDOR_DATASET}"

```
