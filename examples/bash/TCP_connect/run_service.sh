 gst-launch-1.0 v4l2src ! v4l2convert ! videoscale ! \
   video/x-raw,width=1280,height=720,format=RGB,pixel-aspect-ratio=1/1,framerate=30/1 ! \
   sscma_yolov5 model=../net/epoch_300_float.ncnn.bin,../net/epoch_300_float.ncnn.param labels=../net/coco.txt ! \
   v4l2convert ! video/x-raw,format=YVYU ! v4l2h264enc ! 'video/x-h264,level=(string)4.2' ! \
   h264parse config-interval=1 ! rtph264pay ! udpsink host=127.0.0.1 port=7001