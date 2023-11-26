  gst-launch-1.0 \
  rtspsrc location=rtsp://127.0.0.1:8554/test ! \
    rtph264depay ! h264parse config-interval=1 ! v4l2h264dec ! v4l2convert ! videoscale ! \
    video/x-raw,width=1280,height=720,format=RGB ! queue ! \
    sscma_yolov5 model=../net/epoch_300_float.ncnn.bin,../net/epoch_300_float.ncnn.param labels=../net/coco.txt ! \
    v4l2convert ! ximagesink sync=false
