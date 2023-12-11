 GST_DEBUG=3 gst-launch-1.0 -vvv \
  v4l2src name=cam_src device=/dev/video0 ! videoconvert ! videoscale ! \
    sscma_yolov5 model=../net/epoch_300_float.ncnn.bin,../net/epoch_300_float.ncnn.param labels=../net/coco.txt numthreads=4 ! \
    text/x-json ! \
    mqttsink host=127.0.0.1 port=1883 pub-topic="seeed_webrtc"