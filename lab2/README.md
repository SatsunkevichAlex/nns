
Набор входных данных: tfr записи изображений еды в lab формате.

Набор данных для валидации: tfr записи изображений еды в lab формате.

```
**Пример вызова скрипта:**
C:\Python38\python.exe F:/Google-Drive/Университет/магистратура/сорока-лабы/lab2/src/train.py --train \food_tfr --test F:\Google-Drive\Университет\магистратура\сорока-лабы\lab2\archive\validation\food
2020-12-11 11:09:19.787499: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'cudart64_101.dll'; dlerror: cudart64_101.dll not found
2020-12-11 11:09:19.787603: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
2020-12-11 11:09:21.294521: I tensorflow/stream_executor/platform/default/dso_loader.cc:48] Successfully opened dynamic library nvcuda.dll
2020-12-11 11:09:21.317310: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1716] Found device 0 with properties: 
pciBusID: 0000:06:00.0 name: GeForce GTX 1660 SUPER computeCapability: 7.5
coreClock: 1.83GHz coreCount: 22 deviceMemorySize: 6.00GiB deviceMemoryBandwidth: 312.97GiB/s
2020-12-11 11:09:21.317872: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'cudart64_101.dll'; dlerror: cudart64_101.dll not found
2020-12-11 11:09:21.318226: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'cublas64_10.dll'; dlerror: cublas64_10.dll not found
2020-12-11 11:09:21.318724: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'cufft64_10.dll'; dlerror: cufft64_10.dll not found
2020-12-11 11:09:21.319124: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'curand64_10.dll'; dlerror: curand64_10.dll not found
2020-12-11 11:09:21.319523: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'cusolver64_10.dll'; dlerror: cusolver64_10.dll not found
2020-12-11 11:09:21.319914: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'cusparse64_10.dll'; dlerror: cusparse64_10.dll not found
2020-12-11 11:09:21.320269: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'cudnn64_7.dll'; dlerror: cudnn64_7.dll not found
2020-12-11 11:09:21.320351: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1753] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
2020-12-11 11:09:21.320916: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the following CPU instructions in performance-critical operations:  AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2020-12-11 11:09:21.327229: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x23ec9f62c80 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-12-11 11:09:21.327329: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2020-12-11 11:09:21.327483: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1257] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-12-11 11:09:21.327562: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1263]      
Images saved
2020-12-11 11:09:41.817800: I tensorflow/core/profiler/lib/profiler_session.cc:164] Profiler session started.
2020-12-11 11:09:41.817887: I tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1391] Profiler found 1 GPUs
2020-12-11 11:09:41.818495: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'cupti64_101.dll'; dlerror: cupti64_101.dll not found
2020-12-11 11:09:41.818874: W tensorflow/stream_executor/platform/default/dso_loader.cc:59] Could not load dynamic library 'cupti.dll'; dlerror: cupti.dll not found
2020-12-11 11:09:41.818964: E tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1441] function cupti_interface_->Subscribe( &subscriber_, (CUpti_CallbackFunc)ApiCallback, this)failed with error CUPTI could not be loaded or symbol could not be found.
Epoch 1/1000
1/8 [==>...........................] - ETA: 0s - loss: 0.2606 - categorical_accuracy: 0.74192020-12-11 11:09:43.864774: I tensorflow/core/profiler/lib/profiler_session.cc:164] Profiler session started.
2020-12-11 11:09:43.864919: E tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1441] function cupti_interface_->Subscribe( &subscriber_, (CUpti_CallbackFunc)ApiCallback, this)failed with error CUPTI could not be loaded or symbol could not be found.
WARNING:tensorflow:From C:\Python38\lib\site-packages\tensorflow\python\ops\summary_ops_v2.py:1277: stop (from tensorflow.python.eager.profiler) is deprecated and will be removed after 2020-07-01.
Instructions for updating:
use `tf.profiler.experimental.stop` instead.
2020-12-11 11:09:45.142833: I tensorflow/core/profiler/internal/gpu/device_tracer.cc:223]  GpuTracer has collected 0 callback api events and 0 activity events. 
2020-12-11 11:09:45.148306: I tensorflow/core/profiler/rpc/client/save_profile.cc:176] Creating directory: C:/Users/Alex/Desktop/logs/train_data/20201211-110921\train\plugins\profile\2020_12_11_08_09_45
2020-12-11 11:09:45.150052: I tensorflow/core/profiler/rpc/client/save_profile.cc:182] Dumped gzipped tool data for trace.json.gz to C:/Users/Alex/Desktop/logs/train_data/20201211-110921\train\plugins\profile\2020_12_11_08_09_45\DESKTOP-81L2T6G.trace.json.gz
2020-12-11 11:09:45.152277: I tensorflow/core/profiler/rpc/client/save_profile.cc:176] Creating directory: C:/Users/Alex/Desktop/logs/train_data/20201211-110921\train\plugins\profile\2020_12_11_08_09_45
2020-12-11 11:09:45.154638: I tensorflow/core/profiler/rpc/client/save_profile.cc:182] Dumped gzipped tool data for memory_profile.json.gz to C:/Users/Alex/Desktop/logs/train_data/20201211-110921\train\plugins\profile\2020_12_11_08_09_45\DESKTOP-81L2T6G.memory_profile.json.gz
2/8 [======>.......................] - ETA: 3s - loss: 0.2492 - categorical_accuracy: 0.80212020-12-11 11:09:45.162853: I tensorflow/python/profiler/internal/profiler_wrapper.cc:111] Creating directory: C:/Users/Alex/Desktop/logs/train_data/20201211-110921\train\plugins\profile\2020_12_11_08_09_45Dumped tool data for xplane.pb to C:/Users/Alex/Desktop/logs/train_data/20201211-110921\train\plugins\profile\2020_12_11_08_09_45\DESKTOP-81L2T6G.xplane.pb
Dumped tool data for overview_page.pb to C:/Users/Alex/Desktop/logs/train_data/20201211-110921\train\plugins\profile\2020_12_11_08_09_45\DESKTOP-81L2T6G.overview_page.pb
Dumped tool data for input_pipeline.pb to C:/Users/Alex/Desktop/logs/train_data/20201211-110921\train\plugins\profile\2020_12_11_08_09_45\DESKTOP-81L2T6G.input_pipeline.pb
Dumped tool data for tensorflow_stats.pb to C:/Users/Alex/Desktop/logs/train_data/20201211-110921\train\plugins\profile\2020_12_11_08_09_45\DESKTOP-81L2T6G.tensorflow_stats.pb
Dumped tool data for kernel_stats.pb to C:/Users/Alex/Desktop/logs/train_data/20201211-110921\train\plugins\profile\2020_12_11_08_09_45\DESKTOP-81L2T6G.kernel_stats.pb

8/8 [==============================] - 9s 1s/step - loss: 0.1831 - categorical_accuracy: 0.8188
Epoch 2/1000
8/8 [==============================] - 9s 1s/step - loss: 0.0903 - categorical_accuracy: 0.8262
Epoch 3/1000
8/8 [==============================] - 8s 1s/step - loss: 0.0874 - categorical_accuracy: 0.8254
Epoch 4/1000
8/8 [==============================] - 8s 1s/step - loss: 0.0805 - categorical_accuracy: 0.8239
Epoch 5/1000
8/8 [==============================] - 10s 1s/step - loss: 0.0783 - categorical_accuracy: 0.8225
Epoch 6/1000
8/8 [==============================] - 10s 1s/step - loss: 0.0778 - categorical_accuracy: 0.8242
Epoch 7/1000
8/8 [==============================] - 9s 1s/step - loss: 0.0773 - categorical_accuracy: 0.8242
Epoch 8/1000
8/8 [==============================] - 9s 1s/step - loss: 0.0773 - categorical_accuracy: 0.8242
Epoch 9/1000
8/8 [==============================] - 8s 1s/step - loss: 0.0771 - categorical_accuracy: 0.8242
Epoch 10/1000
8/8 [==============================] - 8s 958ms/step - loss: 0.0771 - categorical_accuracy: 0.8242
Epoch 994/1000
8/8 [==============================] - 8s 1s/step - loss: 0.0315 - categorical_accuracy: 0.8155
Epoch 995/1000
8/8 [==============================] - 8s 1s/step - loss: 0.0313 - categorical_accuracy: 0.8121
Epoch 996/1000
8/8 [==============================] - 8s 999ms/step - loss: 0.0313 - categorical_accuracy: 0.8144
Epoch 997/1000
8/8 [==============================] - 8s 1s/step - loss: 0.0315 - categorical_accuracy: 0.8138
Epoch 998/1000
8/8 [==============================] - 8s 1s/step - loss: 0.0317 - categorical_accuracy: 0.8135
Epoch 999/1000
8/8 [==============================] - 8s 997ms/step - loss: 0.0316 - categorical_accuracy: 0.8100
Epoch 1000/1000
8/8 [==============================] - 8s 1s/step - loss: 0.0312 - categorical_accuracy: 0.8116
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, None, None, 1)     10        
_________________________________________________________________
conv2d_1 (Conv2D)            (None, None, None, 8)     80        
_________________________________________________________________
conv2d_2 (Conv2D)            (None, None, None, 16)    1168      
_________________________________________________________________
conv2d_3 (Conv2D)            (None, None, None, 16)    2320      
_________________________________________________________________
conv2d_4 (Conv2D)            (None, None, None, 32)    4640      
_________________________________________________________________
conv2d_5 (Conv2D)            (None, None, None, 32)    9248      
_________________________________________________________________
up_sampling2d (UpSampling2D) (None, None, None, 32)    0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, None, None, 32)    9248      
_________________________________________________________________
up_sampling2d_1 (UpSampling2 (None, None, None, 32)    0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, None, None, 16)    4624      
_________________________________________________________________
up_sampling2d_2 (UpSampling2 (None, None, None, 16)    0         
_________________________________________________________________
conv2d_8 (Conv2D)            (None, None, None, 2)     290       
=================================================================
Total params: 31,628
Trainable params: 31,628
Non-trainable params: 0
_________________________________________________________________
None

Process finished with exit code 0
```

**Графики обучение**
(https://github.com/SatsunkevichAlex/nns/blob/main/lab2/src/graphs.png)

**Результат колоризации**
(https://github.com/SatsunkevichAlex/nns/blob/main/lab2/src/images.png)
