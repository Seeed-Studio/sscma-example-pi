gst-launch-1.0 v4l2src ! video/x-h264,width=1280,height=720,framerate=30/1  ! h264parse config-interval=1 ! v4l2h264dec ! v4l2convert  ! xvimagesink sync=false 