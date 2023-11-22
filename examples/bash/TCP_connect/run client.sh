gst-launch-1.0 udpsrc port=7001 caps="application/x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)H264,payload=(int)96" \
   ! rtph264depay ! h264parse ! avdec_h264 ! \ 
   queue ! videoconvert ! ximagesink sync=false