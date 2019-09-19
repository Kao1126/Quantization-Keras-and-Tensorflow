# QuantizationCNN-Tensorflow
Implementation quantization-aware training on classification model with Tensorflow and Keras

## Requirement 
tensorflow - 1.15.0

Ubuntu - 16.04

## Introduction
It is based on Tensorflow 1.15.0, it can help your CNN model to do quantization-aware training simply, all you need to do is prepare your Keras model and dataset.

The quantization-aware training will transform int8 from float32.Int8 can be compiled on Edge TPU or mobile device.
It uses fake quantization nodes to simulate the effect of quantization when you're training.
The code will use quantization-aware training to produce checkpoints(.data, .mete, .index) in checkpoint folder, and then export them into the new .pd file.

## How to use this code

1. You can refer to model/example_VGG/VGG.py to choose the learning rate
   , optimizer to create your own model (save as model.h5).

2. Prepare your data in data folder:
##
    train--
          -class1
          -class2
              - 1.jpg
              - 2.jpg
                  '
                  '
                  '
    val--
          -class1
          -class2
     '
     '
     '
             
3. Try the quantize model works by running it through the main.py script:
##
    python main.py \
    --model_path  your/model/path \
    --epoch  ChooseYourEpoch \
    --batch_size  ChooseYourBatchSize
    
4. You can get the .pd file that your model has been quantized

## Example
    python main.py \
    --model_path  ./model/example_VGG/model.h5 \
    --epoch  10 \
    --batch_size  16

## Reference
1. asr_edgetpu_demo:
  https://github.com/s123600g/asr_edgetpu_demo?fbclid=IwAR2qHKWR7wgg6_90-1o2ZOapm4a7OC1xlKCBcB0BE3qlYdEN8tVHwWUjJZU

2. coral:
  https://coral.withgoogle.com/docs/edgetpu/models-intro/

3. tensorflow:
   https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/quantize
