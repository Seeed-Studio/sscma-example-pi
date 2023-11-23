## existing rtsp video streams
    1. Modify the rtsp URL in run.sh
    ```bash
    location=rtsp://127.0.0.1:8554/test
    ```
    2. run gstreamer pipeline
    ```bash
    ./run.sh
    ```

## rtsp video streams from local camera
### run rtsp server
    1. install gst-rtsp-server
    ```bash
    git clone git://anongit.freedesktop.org/gstreamer/gst-rtsp-server
    cd gst-rtsp-server
    meson build
    ninja -C build
    sudo ninja -C build install
    cp ./examples/test-launch ../
    ```
    2. Configuring environment variables
    ```bash
    echo "export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
    source ~/.bashrc
    ```
    3. start Camera RTSP Server
    ```bash
    ./rtsp_output.sh
    ```

### run rtsp client
    ```bash
    ./run.sh
    ```