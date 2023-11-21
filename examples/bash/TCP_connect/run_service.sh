 gst-launch-1.0 v4l2src ! videoconvert ! \
  video/x-raw,format=I420 ! x264enc ! video/x-h264,profile=baseline ! \
   h264parse ! rtph264pay ! udpsink host=192.168.130.139 port=7001