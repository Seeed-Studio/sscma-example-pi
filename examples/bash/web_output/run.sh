gst-launch-1.0  v4l2src device=/dev/video0 ! videoconvert ! videoscale ! \
    video/x-raw,width=1280,height=720,format=RGB,pixel-aspect-ratio=1/1,framerate=15/1 ! \
    sscma_yolov5 model=../net/epoch_300_float.ncnn.bin,../net/epoch_300_float.ncnn.param labels=../net/coco.txt ! \
    videoconvert ! x264enc ! h264parse ! \
    hlssink2 max-files=10 location=./record_%05d.ts  playlist-location=./playlist.m3u8