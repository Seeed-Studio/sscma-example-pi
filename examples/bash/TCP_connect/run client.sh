gst-launch-1.0 udpsrc port=7001 ! \
 application/x-rtp,encoding-name=H264 ! \
  rtph264depay ! h264parse ! avdec_h264 ! \
   queue ! ximagesink sync=false