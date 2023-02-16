# Cat escaping detection and alarming system


### Configurations
* For loading YOLO v3 network, you need to download config file and pretrained weights file [here (237MB)](https://pjreddie.com/media/files/yolov3.weights).
* When train on custom data using transfer learning technique, download pre-trained weights from [darknet53](https://pjreddie.com/darknet/imagenet/#darknet53) for convolutional layers from [here (76MB)](https://pjreddie.com/media/files/darknet53.conv.74).
* Change configuration details in `yolov3.cfg` file in terms of **num_classes** (in my case 5) and corresponding **filter numbers** ((num_class + num_coordinates + 1) * 3) in convolutional layers right above the yolo layers.
* Note that you need to be careful with the proper handling of path as the yolov3 couldn't handle any possible spaces in the file path, hence, will cause problem when accessing relevant config files in training process.


### Results
* The model was trained on 319 images for 6000 iterations and tested on 56 images, got a test mAP of 66.67%.