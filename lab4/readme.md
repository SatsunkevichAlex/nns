**Скрипт**
```
__author__ = 'Alexander Soroka, soroka.a.m@gmail.com'
__copyright__ = """Copyright 2020 Alexander Soroka"""

import os
from datetime import datetime
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers
import tensorflow_io as tfio
import numpy as np

SHUFFLE_BUFFER = 4
BATCH_SIZE = 128
NUM_CLASSES = 6
PARALLEL_CALLS = 4
RESIZE_TO = 224
TRAINSET_SIZE = 14034
VALSET_SIZE = 3000
TRAIN_FOLDER = 'train_tfr'
VALIDATION_FOLDER = 'validation_tfr'


config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)


def generator_train():
    data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomCrop(112, 112),
        layers.experimental.preprocessing.Resizing(224, 224),
        layers.experimental.preprocessing.RandomRotation(factor=0.45),
        layers.experimental.preprocessing.RandomFlip(mode='horizontal')
    ])

    current_dir = os.path.dirname(os.path.realpath(__file__))
    train_dir = Path(current_dir + f"/{TRAIN_FOLDER}")
    file_list_train = [str(pp) for pp in train_dir.glob("*")]
    i = 0;
    for fn in file_list_train:
        for dataset in create_dataset(fn, BATCH_SIZE):
            dataset = data_augmentation(dataset)
            x = np.reshape(dataset[:, :, :, 0], (-1, RESIZE_TO, RESIZE_TO, 1))
            y = np.reshape(dataset[:, :, :, 1:], (-1, RESIZE_TO, RESIZE_TO, 2))
            i = i + 1
            yield (x, y)


def generator_valid():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    train_dir = Path(current_dir + f"/{VALIDATION_FOLDER}")
    file_list_train = [str(pp) for pp in train_dir.glob("*")]
    i = 0;
    for fn in file_list_train:
        for dataset in create_dataset(fn, BATCH_SIZE):
            x = np.reshape(dataset[:, :, :, 0], (-1, RESIZE_TO, RESIZE_TO, 1))
            y = np.reshape(dataset[:, :, :, 1:], (-1, RESIZE_TO, RESIZE_TO, 2))
            i = i + 1
            yield (x, y)


def visualize_images(epoch, model, dataset, writer):
    item = iter(dataset).next()

    l_channel = item[0]
    target_ab = item[1]
    target_image = np.zeros((l_channel.shape[0], l_channel.shape[1], l_channel.shape[2], 3))
    target_image[:, :, :, 1:] = target_ab
    target_image[:, :, :, 0] = np.reshape(l_channel, (-1, 224, 224))

    predicted_ab = model(np.reshape(l_channel, (-1, 224, 224, 1)))
    predicted_image = np.zeros((l_channel.shape[0], l_channel.shape[1], l_channel.shape[2], 3))
    predicted_image[:, :, :, 0] = np.reshape(l_channel, (-1, 224, 224))
    predicted_image[:, :, :, 1:] = predicted_ab
    maxl = tf.math.reduce_max(target_image)
    minl = tf.math.reduce_min(target_image)
    meanl = tf.math.reduce_mean(target_image)
    print(f'Info about image in Lab format')
    print(f'Max in target: {maxl}, min: {minl}, mean: {meanl}')

    maxlp = tf.math.reduce_max(predicted_image)
    minlp = tf.math.reduce_min(predicted_image)
    meanlp = tf.math.reduce_mean(predicted_image)
    print(f'Max in predicted: {maxlp}, min: {minlp}, mean: {meanlp}')

    target_rgb = tfio.experimental.color.lab_to_rgb(target_image)
    target_rgb = tf.math.multiply(target_rgb, 256)
    predicted_rgb = tfio.experimental.color.lab_to_rgb(predicted_image) * 256

    max = tf.math.reduce_max(target_rgb)
    min = tf.math.reduce_min(target_rgb)
    mean = tf.math.reduce_mean(target_rgb)
    print(f'Info about image in RGB format')
    print(f'Max in target: {max}, min: {min}, mean: {mean}')

    maxp = tf.math.reduce_max(predicted_rgb)
    minp = tf.math.reduce_min(predicted_rgb)
    meanp = tf.math.reduce_mean(predicted_rgb)
    print(f'Max in predicted: {maxp}, min: {minp}, mean: {meanp}')

    with writer.as_default():
        tf.summary.image('Target Lab', np.reshape(target_image, (-1, 224, 224, 3)), step=epoch)
        tf.summary.image('Result Lab', np.reshape(predicted_image, (-1, 224, 224, 3)), step=epoch)
        tf.summary.image('Target RGB', target_rgb, step=epoch)
        tf.summary.image('Result RGB', predicted_rgb, step=epoch)


def visualize_images_augmented(epoch, dataset, writer):
    item = iter(dataset).next()
    l_channel = item[0]
    target_ab = item[1]
    target_image = np.zeros((l_channel.shape[0], l_channel.shape[1], l_channel.shape[2], 3))
    target_image[:, :, :, 1:] = target_ab
    target_image[:, :, :, 0] = np.reshape(l_channel, (-1, 224, 224))

    with writer.as_default():
        tf.summary.image('Augmented', target_image, step=epoch)


def parse_proto_example(proto):
    keys_to_features = {
        'image/encoded': tf.io.FixedLenFeature((), tf.string, default_value='')
    }
    example = tf.io.parse_single_example(proto, keys_to_features)
    example['image'] = tf.image.decode_jpeg(example['image/encoded'], channels=3)
    example['image'] = tf.image.convert_image_dtype(example['image'], dtype=tf.float32)
    example['image'] = tf.image.resize(example['image'], tf.constant([RESIZE_TO, RESIZE_TO]))
    return example['image']


def create_dataset(filenames, batch_size):
    """Create dataset from tfrecords file
    :tfrecords_files: Mask to collect tfrecords file of dataset
    :returns: tf.data.Dataset
    """
    return tf.data.TFRecordDataset(filenames)\
        .map(parse_proto_example)\
        .batch(batch_size)\
        .prefetch(batch_size)


def display_image_count():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    train_dir = Path(current_dir + f"/{TRAIN_FOLDER}")
    file_list_train = [str(pp) for pp in train_dir.glob("*")]
    file_list_train = tf.random.shuffle(file_list_train)
    c = 0
    for fn in file_list_train:
        for record in tf.data.TFRecordDataset(fn):
            c += 1
    print(f'Count of train images: {c}')

    valid_dir = Path(current_dir + f"/{VALIDATION_FOLDER}")
    file_list_valid = [str(pp) for pp in valid_dir.glob("*")]
    file_list_valid = tf.random.shuffle(file_list_valid)
    v = 0
    for fn in file_list_valid:
        for record in tf.data.TFRecordDataset(fn):
            v += 1
    print(f'Count of validation images: {v}')


def main():
    log_dir = "C:/Users/Alex/Desktop/logs/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(log_dir)

    display_image_count()

    train = tf.data.Dataset.from_generator(
        generator_train,
        (tf.float32, tf.float32))

    valid = tf.data.Dataset.from_generator(
        generator_valid,
        (tf.float32, tf.float32))

    IMG_SHAPE = (RESIZE_TO, RESIZE_TO, 3)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False

    prediction_layer = tf.keras.layers.Dense(2)
    inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 1))
    x = tf.image.grayscale_to_rgb(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=4, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=2, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=2, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2DTranspose(2, (2, 2), strides=2, activation='relu', padding='same')(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
         optimizer=tf.optimizers.SGD(lr=0.001, momentum=0.9),
         loss=tf.keras.losses.mean_squared_error
    )

    print(model.summary())

    model.fit(
        train,
        epochs=100,
        validation_data=valid,
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir),
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: visualize_images(epoch, model, valid, file_writer)
            ),
            # tf.keras.callbacks.LambdaCallback(
            #     on_epoch_end=lambda epoch, logs: visualize_images_augmented(epoch, train, file_writer)
            # )
        ]
    )


if __name__ == '__main__':
    main()
```

**Логи**
```
C:\Python38\python.exe D:/nns/lab4/transfer-learn.py
2020-12-23 17:03:57.464821: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudart64_110.dll
2020-12-23 17:04:03.347147: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2020-12-23 17:04:03.349597: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library nvcuda.dll
2020-12-23 17:04:03.380606: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties: 
pciBusID: 0000:06:00.0 name: GeForce GTX 1660 SUPER computeCapability: 7.5
coreClock: 1.83GHz coreCount: 22 deviceMemorySize: 6.00GiB deviceMemoryBandwidth: 312.97GiB/s
2020-12-23 17:04:03.380749: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudart64_110.dll
2020-12-23 17:04:03.863065: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublas64_11.dll
2020-12-23 17:04:03.863157: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublasLt64_11.dll
2020-12-23 17:04:03.888402: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cufft64_10.dll
2020-12-23 17:04:03.922522: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library curand64_10.dll
2020-12-23 17:04:04.173938: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cusolver64_10.dll
2020-12-23 17:04:04.459519: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cusparse64_11.dll
2020-12-23 17:04:05.413270: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudnn64_8.dll
2020-12-23 17:04:05.413423: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1862] Adding visible gpu devices: 0
2020-12-23 17:04:06.004651: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1261] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-12-23 17:04:06.004740: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1267]      0 
2020-12-23 17:04:06.004783: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1280] 0:   N 
2020-12-23 17:04:06.004951: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1406] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 4616 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1660 SUPER, pci bus id: 0000:06:00.0, compute capability: 7.5)
2020-12-23 17:04:06.005570: I tensorflow/compiler/jit/xla_gpu_device.cc:99] Not creating XLA devices, tf_xla_enable_xla_devices not set
2020-12-23 17:04:06.006374: I tensorflow/compiler/jit/xla_cpu_device.cc:41] Not creating XLA devices, tf_xla_enable_xla_devices not set
2020-12-23 17:04:06.006556: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties: 
pciBusID: 0000:06:00.0 name: GeForce GTX 1660 SUPER computeCapability: 7.5
coreClock: 1.83GHz coreCount: 22 deviceMemorySize: 6.00GiB deviceMemoryBandwidth: 312.97GiB/s
2020-12-23 17:04:06.006711: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudart64_110.dll
2020-12-23 17:04:06.006781: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublas64_11.dll
2020-12-23 17:04:06.006849: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublasLt64_11.dll
2020-12-23 17:04:06.006931: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cufft64_10.dll
2020-12-23 17:04:06.007005: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library curand64_10.dll
2020-12-23 17:04:06.007077: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cusolver64_10.dll
2020-12-23 17:04:06.007144: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cusparse64_11.dll
2020-12-23 17:04:06.007211: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudnn64_8.dll
2020-12-23 17:04:06.007304: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1862] Adding visible gpu devices: 0
2020-12-23 17:04:06.007739: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties: 
pciBusID: 0000:06:00.0 name: GeForce GTX 1660 SUPER computeCapability: 7.5
coreClock: 1.83GHz coreCount: 22 deviceMemorySize: 6.00GiB deviceMemoryBandwidth: 312.97GiB/s
2020-12-23 17:04:06.007877: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudart64_110.dll
2020-12-23 17:04:06.007953: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublas64_11.dll
2020-12-23 17:04:06.008026: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublasLt64_11.dll
2020-12-23 17:04:06.008096: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cufft64_10.dll
2020-12-23 17:04:06.008168: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library curand64_10.dll
2020-12-23 17:04:06.008236: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cusolver64_10.dll
2020-12-23 17:04:06.008302: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cusparse64_11.dll
2020-12-23 17:04:06.008375: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudnn64_8.dll
2020-12-23 17:04:06.008468: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1862] Adding visible gpu devices: 0
2020-12-23 17:04:06.008988: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1261] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-12-23 17:04:06.009066: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1267]      0 
2020-12-23 17:04:06.009108: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1280] 0:   N 
2020-12-23 17:04:06.009240: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1406] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 4616 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1660 SUPER, pci bus id: 0000:06:00.0, compute capability: 7.5)
2020-12-23 17:04:06.009363: I tensorflow/compiler/jit/xla_gpu_device.cc:99] Not creating XLA devices, tf_xla_enable_xla_devices not set
2020-12-23 17:04:06.030716: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
Count of train images: 50000
Count of validation images: 4999
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         [(None, 224, 224, 1)]     0         
_________________________________________________________________
tf.image.grayscale_to_rgb (T (None, 224, 224, 3)       0         
_________________________________________________________________
mobilenetv2_1.00_224 (Functi (None, 7, 7, 1280)        2257984   
_________________________________________________________________
conv2d_transpose (Conv2DTran (None, 28, 28, 128)       2621568   
_________________________________________________________________
conv2d_transpose_1 (Conv2DTr (None, 56, 56, 128)       65664     
_________________________________________________________________
conv2d_transpose_2 (Conv2DTr (None, 112, 112, 64)      32832     
_________________________________________________________________
conv2d_transpose_3 (Conv2DTr (None, 224, 224, 2)       514       
_________________________________________________________________
dense (Dense)                (None, 224, 224, 2)       6         
=================================================================
Total params: 4,978,568
Trainable params: 2,720,584
Non-trainable params: 2,257,984
_________________________________________________________________
None
2020-12-23 17:04:10.121063: I tensorflow/core/profiler/lib/profiler_session.cc:136] Profiler session initializing.
2020-12-23 17:04:10.121127: I tensorflow/core/profiler/lib/profiler_session.cc:155] Profiler session started.
2020-12-23 17:04:10.121185: I tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1365] Profiler found 1 GPUs
2020-12-23 17:04:10.146223: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cupti64_110.dll
2020-12-23 17:04:10.213932: I tensorflow/core/profiler/lib/profiler_session.cc:172] Profiler session tear down.
2020-12-23 17:04:10.214060: I tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1487] CUPTI activity buffer flushed
Epoch 1/100
2020-12-23 17:04:12.832760: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublas64_11.dll
2020-12-23 17:04:13.340386: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublasLt64_11.dll
2020-12-23 17:04:13.712603: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cudnn64_8.dll
2020-12-23 17:04:16.796401: I tensorflow/core/platform/windows/subprocess.cc:308] SubProcess ended with return code: 0

2020-12-23 17:04:16.915335: I tensorflow/core/platform/windows/subprocess.cc:308] SubProcess ended with return code: 0

2020-12-23 17:04:18.486290: W tensorflow/core/common_runtime/bfc_allocator.cc:248] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.54GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2020-12-23 17:04:18.486465: W tensorflow/core/common_runtime/bfc_allocator.cc:248] Allocator (GPU_0_bfc) ran out of memory trying to allocate 1.54GiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
      1/Unknown - 12s 12s/step - loss: 0.28412020-12-23 17:04:23.307214: I tensorflow/core/profiler/lib/profiler_session.cc:136] Profiler session initializing.
2020-12-23 17:04:23.307283: I tensorflow/core/profiler/lib/profiler_session.cc:155] Profiler session started.
      2/Unknown - 14s 1s/step - loss: 0.2486 2020-12-23 17:04:24.006179: I tensorflow/core/profiler/lib/profiler_session.cc:71] Profiler session collecting data.
2020-12-23 17:04:24.006393: I tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1487] CUPTI activity buffer flushed
2020-12-23 17:04:24.105807: I tensorflow/core/profiler/internal/gpu/cupti_collector.cc:228]  GpuTracer has collected 681 callback api events and 653 activity events. 
2020-12-23 17:04:24.120809: I tensorflow/core/profiler/lib/profiler_session.cc:172] Profiler session tear down.
2020-12-23 17:04:24.158024: I tensorflow/core/profiler/rpc/client/save_profile.cc:137] Creating directory: C:/Users/Alex/Desktop/logs/train_data/20201223-170406\train\plugins\profile\2020_12_23_14_04_24
2020-12-23 17:04:24.170079: I tensorflow/core/profiler/rpc/client/save_profile.cc:143] Dumped gzipped tool data for trace.json.gz to C:/Users/Alex/Desktop/logs/train_data/20201223-170406\train\plugins\profile\2020_12_23_14_04_24\DESKTOP-81L2T6G.trace.json.gz
2020-12-23 17:04:24.224299: I tensorflow/core/profiler/rpc/client/save_profile.cc:137] Creating directory: C:/Users/Alex/Desktop/logs/train_data/20201223-170406\train\plugins\profile\2020_12_23_14_04_24
2020-12-23 17:04:24.230812: I tensorflow/core/profiler/rpc/client/save_profile.cc:143] Dumped gzipped tool data for memory_profile.json.gz to C:/Users/Alex/Desktop/logs/train_data/20201223-170406\train\plugins\profile\2020_12_23_14_04_24\DESKTOP-81L2T6G.memory_profile.json.gz
2020-12-23 17:04:24.244074: I tensorflow/core/profiler/rpc/client/capture_profile.cc:251] Creating directory: C:/Users/Alex/Desktop/logs/train_data/20201223-170406\train\plugins\profile\2020_12_23_14_04_24Dumped tool data for xplane.pb to C:/Users/Alex/Desktop/logs/train_data/20201223-170406\train\plugins\profile\2020_12_23_14_04_24\DESKTOP-81L2T6G.xplane.pb
Dumped tool data for overview_page.pb to C:/Users/Alex/Desktop/logs/train_data/20201223-170406\train\plugins\profile\2020_12_23_14_04_24\DESKTOP-81L2T6G.overview_page.pb
Dumped tool data for input_pipeline.pb to C:/Users/Alex/Desktop/logs/train_data/20201223-170406\train\plugins\profile\2020_12_23_14_04_24\DESKTOP-81L2T6G.input_pipeline.pb
Dumped tool data for tensorflow_stats.pb to C:/Users/Alex/Desktop/logs/train_data/20201223-170406\train\plugins\profile\2020_12_23_14_04_24\DESKTOP-81L2T6G.tensorflow_stats.pb
Dumped tool data for kernel_stats.pb to C:/Users/Alex/Desktop/logs/train_data/20201223-170406\train\plugins\profile\2020_12_23_14_04_24\DESKTOP-81L2T6G.kernel_stats.pb

392/392 [==============================] - 191s 457ms/step - loss: 0.1330 - val_loss: 0.0844
Info about image in Lab format
Max in target: 1.0, min: 0.0, mean: 0.5222531512537213
Max in predicted: 1.0, min: 0.0, mean: 0.43186299506802345
Info about image in RGB format
Max in target: 7.37346924186825, min: 0.0, mean: 1.997530414160506
Max in predicted: 5.196532558031158, min: 0.0, mean: 1.960777319000825
Epoch 100/100
392/392 [==============================] - 161s 411ms/step - loss: 0.0752 - val_loss: 0.0828
Info about image in Lab format
Max in target: 1.0, min: 0.0, mean: 0.5222531512537213
Max in predicted: 1.0, min: 0.0, mean: 0.4362925887074163
Info about image in RGB format
Max in target: 7.37346924186825, min: 0.0, mean: 1.997530414160506
Max in predicted: 5.192003512639624, min: 0.0, mean: 1.9611567474539484

Process finished with exit code 0
```

**Рузльтаты**
Запуск 1. optimizer=tf.optimizers.Adam(lr=0.01)
![1run](https://github.com/SatsunkevichAlex/nns/blob/main/lab4/runs/images_1.png)
![1run](https://github.com/SatsunkevichAlex/nns/blob/main/lab4/runs/scalars1.png)

Запуск 2. optimizer=tf.optimizers.Adam(lr=0.01)
![2run](https://github.com/SatsunkevichAlex/nns/blob/main/lab4/runs/images_2.png)
![2run](https://github.com/SatsunkevichAlex/nns/blob/main/lab4/runs/scalars2.png)

Запуск 3. optimizer=tf.optimizers.SGD(lr=0.001, momentum=0.9)
![3run](https://github.com/SatsunkevichAlex/nns/blob/main/lab4/runs/images_3.png)
![3run](https://github.com/SatsunkevichAlex/nns/blob/main/lab4/runs/scalars3.png)
