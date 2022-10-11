# flower_classifier_pytorch
Python script trains and exports .onnx-model.

## Train and export model
The script trains Pytorch model and exports it to the ONNX-file:
```bash
python3 flower_classifier.py
```

## Check the output results of model
### build C++ code:
```bash
g++ -std=c++17 -g -I . -I /usr/local/include/opencv4 -c -O3 -Wall -c -fmessage-length=0 flowers_dnn.cpp
g++ -std=c++17 -L . -L /usr/local/lib -o flowers_dnn flowers_dnn.o -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_ml -lopencv_flann -lopencv_objdetect -lopencv_calib3d -lopencv_features2d -lopencv_video -lopencv_videoio -lopencv_dnn
```
### run C++ program
```bash
./flowers_dnn
```

## URL-path to the dataset
https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
