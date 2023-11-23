./test-launch "( v4l2src name=cam_src ! videoconvert ! videoscale ! \
  videoconvert ! x264enc ! video/x-h264,profile=baseline ! \
  h264parse ! rtph264pay name=pay0 pt=96 )"


