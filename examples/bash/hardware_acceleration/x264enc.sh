gst-launch-1.0 v4l2src ! videoconvert ! videoscale ! video/x-raw,width=1280,height=720,framerate=30/1,format=YVYU ! videoconvert ! x264enc ! 'video/x-h264,level=(string)4.2' ! h264parse ! avdec_h264 ! videoconvert ! xvimagesink sync=false  