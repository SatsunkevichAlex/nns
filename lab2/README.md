
Набор входных данных: tfr записи изображений еды в lab формате.

Набор данных для валидации: tfr записи изображений еды в lab формате.

```
**Пример вызова скрипта:**
C:\Python38\python.exe F:/Google-Drive/Университет/магистратура/сорока-лабы/lab2/src/train.py --train \food_tfr --test F:\Google-Drive\Университет\магистратура\сорока-лабы\lab2\archive\validation\food
2020-12-13 09:16:17.039111: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'cudart64_101.dll'; dlerror: cudart64_101.dll not found
2020-12-13 09:16:17.039989: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
2020-12-13 09:16:23.741498: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library nvcuda.dll
2020-12-13 09:16:23.809696: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties: 
pciBusID: 0000:06:00.0 name: GeForce GTX 1660 SUPER computeCapability: 7.5
coreClock: 1.83GHz coreCount: 22 deviceMemorySize: 6.00GiB deviceMemoryBandwidth: 312.97GiB/s
2020-12-13 09:16:23.810408: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'cudart64_101.dll'; dlerror: cudart64_101.dll not found
2020-12-13 09:16:23.810784: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'cublas64_10.dll'; dlerror: cublas64_10.dll not found
2020-12-13 09:16:23.811278: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'cufft64_10.dll'; dlerror: cufft64_10.dll not found
2020-12-13 09:16:23.811832: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'curand64_10.dll'; dlerror: curand64_10.dll not found
2020-12-13 09:16:23.812316: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'cusolver64_10.dll'; dlerror: cusolver64_10.dll not found
2020-12-13 09:16:23.812728: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'cusparse64_10.dll'; dlerror: cusparse64_10.dll not found
2020-12-13 09:16:23.813123: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'cudnn64_7.dll'; dlerror: cudnn64_7.dll not found
2020-12-13 09:16:23.813295: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1753] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
2020-12-13 09:16:23.816637: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2020-12-13 09:16:23.852891: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x1f572bd9f20 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-12-13 09:16:23.853005: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2020-12-13 09:16:23.854020: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1257] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-12-13 09:16:23.854093: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1263]      
Images saved
2020-12-13 09:16:45.530518: I tensorflow/core/profiler/lib/profiler_session.cc:164] Profiler session started.
2020-12-13 09:16:45.530909: I tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1391] Profiler found 1 GPUs
2020-12-13 09:16:45.532094: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'cupti64_101.dll'; dlerror: cupti64_101.dll not found
2020-12-13 09:16:45.532458: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'cupti.dll'; dlerror: cupti.dll not found
2020-12-13 09:16:45.532545: E tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1441] function cupti_interface_->Subscribe( &subscriber_, (CUpti_CallbackFunc)ApiCallback, this)failed with error CUPTI could not be loaded or symbol could not be found.
Epoch 1/200
2020-12-13 09:16:49.480103: I tensorflow/core/profiler/lib/profiler_session.cc:164] Profiler session started.
2020-12-13 09:16:49.480256: E tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1441] function cupti_interface_->Subscribe( &subscriber_, (CUpti_CallbackFunc)ApiCallback, this)failed with error CUPTI could not be loaded or symbol could not be found.
1/8 [==>...........................] - ETA: 0s - loss: 0.2230 - categorical_accuracy: 0.7718WARNING:tensorflow:From C:\Python38\lib\site-packages\tensorflow\python\ops\summary_ops_v2.py:1277: stop (from tensorflow.python.eager.profiler) is deprecated and will be removed after 2020-07-01.
Instructions for updating:
use `tf.profiler.experimental.stop` instead.
2020-12-13 09:16:52.804706: I tensorflow/core/profiler/internal/gpu/device_tracer.cc:223]  GpuTracer has collected 0 callback api events and 0 activity events. 
2020-12-13 09:16:52.816383: I tensorflow/core/profiler/rpc/client/save_profile.cc:176] Creating directory: C:/Users/Alex/Desktop/logs/train_data/20201213-091623\train\plugins\profile\2020_12_13_06_16_52
2020-12-13 09:16:52.821024: I tensorflow/core/profiler/rpc/client/save_profile.cc:182] Dumped gzipped tool data for trace.json.gz to C:/Users/Alex/Desktop/logs/train_data/20201213-091623\train\plugins\profile\2020_12_13_06_16_52\DESKTOP-81L2T6G.trace.json.gz
2020-12-13 09:16:52.829803: I tensorflow/core/profiler/rpc/client/save_profile.cc:176] Creating directory: C:/Users/Alex/Desktop/logs/train_data/20201213-091623\train\plugins\profile\2020_12_13_06_16_52
2020-12-13 09:16:52.831946: I tensorflow/core/profiler/rpc/client/save_profile.cc:182] Dumped gzipped tool data for memory_profile.json.gz to C:/Users/Alex/Desktop/logs/train_data/20201213-091623\train\plugins\profile\2020_12_13_06_16_52\DESKTOP-81L2T6G.memory_profile.json.gz
2/8 [======>.......................] - ETA: 10s - loss: 0.2338 - categorical_accuracy: 0.80502020-12-13 09:16:52.840771: I tensorflow/python/profiler/internal/profiler_wrapper.cc:111] Creating directory: C:/Users/Alex/Desktop/logs/train_data/20201213-091623\train\plugins\profile\2020_12_13_06_16_52Dumped tool data for xplane.pb to C:/Users/Alex/Desktop/logs/train_data/20201213-091623\train\plugins\profile\2020_12_13_06_16_52\DESKTOP-81L2T6G.xplane.pb
Dumped tool data for overview_page.pb to C:/Users/Alex/Desktop/logs/train_data/20201213-091623\train\plugins\profile\2020_12_13_06_16_52\DESKTOP-81L2T6G.overview_page.pb
Dumped tool data for input_pipeline.pb to C:/Users/Alex/Desktop/logs/train_data/20201213-091623\train\plugins\profile\2020_12_13_06_16_52\DESKTOP-81L2T6G.input_pipeline.pb
Dumped tool data for tensorflow_stats.pb to C:/Users/Alex/Desktop/logs/train_data/20201213-091623\train\plugins\profile\2020_12_13_06_16_52\DESKTOP-81L2T6G.tensorflow_stats.pb
Dumped tool data for kernel_stats.pb to C:/Users/Alex/Desktop/logs/train_data/20201213-091623\train\plugins\profile\2020_12_13_06_16_52\DESKTOP-81L2T6G.kernel_stats.pb

8/8 [==============================] - 24s 3s/step - loss: 0.1957 - categorical_accuracy: 0.8252
Epoch 2/200
8/8 [==============================] - 24s 3s/step - loss: 0.0948 - categorical_accuracy: 0.8262
...
...
...
Epoch 199/200
8/8 [==============================] - 21s 3s/step - loss: 0.0336 - categorical_accuracy: 0.8177
Epoch 200/200
8/8 [==============================] - 21s 3s/step - loss: 0.0336 - categorical_accuracy: 0.8113
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 224, 224, 1)       17        
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 112, 112, 1)       0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 112, 112, 32)      544       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 56, 56, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 56, 56, 64)        32832     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 28, 28, 64)        0         
_________________________________________________________________
conv2d_transpose (Conv2DTran (None, 28, 28, 128)       131200    
_________________________________________________________________
up_sampling2d (UpSampling2D) (None, 56, 56, 128)       0         
_________________________________________________________________
conv2d_transpose_1 (Conv2DTr (None, 56, 56, 128)       262272    
_________________________________________________________________
up_sampling2d_1 (UpSampling2 (None, 112, 112, 128)     0         
_________________________________________________________________
conv2d_transpose_2 (Conv2DTr (None, 112, 112, 64)      131136    
_________________________________________________________________
up_sampling2d_2 (UpSampling2 (None, 224, 224, 64)      0         
_________________________________________________________________
conv2d_transpose_3 (Conv2DTr (None, 224, 224, 2)       2050      
=================================================================
Total params: 560,051
Trainable params: 560,051
Non-trainable params: 0
_________________________________________________________________
None

Process finished with exit code 0
```

**Графики обучение**
![LearhGraphs](https://github.com/SatsunkevichAlex/nns/blob/main/lab2/src/graphs.png)

**Результат колоризации**
![ResultImages](https://github.com/SatsunkevichAlex/nns/blob/main/lab2/src/images.png)
