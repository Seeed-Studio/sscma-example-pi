gst-launch-1.0 v4l2src ! v4l2convert  ! videoscale ! video/x-raw,width=1280,height=720,framerate=30/1,format=YVYU ! v4l2convert  ! v4l2h264enc ! 'video/x-h264,level=(string)4.2' ! h264parse config-interval=1 ! v4l2h264dec ! v4l2convert  ! xvimagesink sync=false   