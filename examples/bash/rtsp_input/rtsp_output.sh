./test-launch "( v4l2src name=cam_src ! v4l2convert ! videoscale ! \
  v4l2convert ! queue ! v4l2h264enc ! 'video/x-h264,level=(string)4.2' ! \
  v4l2convert ! video/x-h264,profile=baseline ! \
  h264parse ! rtph264pay name=pay0 pt=96 )"


