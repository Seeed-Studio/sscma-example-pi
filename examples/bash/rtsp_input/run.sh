  gst-launch-1.0 \
  rtspsrc location=rtsp://127.0.0.1:8554/test ! \
    rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! videoscale ! \
    video/x-raw,width=1280,height=720,format=RGB ! queue ! \
    sscma_yolov5 model=../net/epoch_300_float.ncnn.bin,../net/epoch_300_float.ncnn.param labels=../net/coco.txt ! \
    videoconvert ! ximagesink sync=false
