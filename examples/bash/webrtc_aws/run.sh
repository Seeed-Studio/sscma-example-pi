AWS_DEFAULT_REGION="us-west-2" AWS_ACCESS_KEY_ID="XXX" AWS_SECRET_ACCESS_KEY="XXX" \
gst-launch-1.0 awskvswebrtcsink name=ws signaller::channel-name="seeed_webrtc" \
 v4l2src ! v4l2convert ! videoscale ! \
 video/x-raw,width=1280,height=720,format=RGB,pixel-aspect-ratio=1/1,framerate=30/1 ! \
 queue ! sscma_yolov5 model=../net/epoch_300_float.ncnn.bin,../net/epoch_300_float.ncnn.param labels=../net/coco.txt ! ws.