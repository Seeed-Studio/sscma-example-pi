# Deploying a Model Trained with SSCMA on Raspberry Pi

This guide explains how to deploy a model trained with SSCMA (to be confirmed) on a Raspberry Pi using NCNN as the inference engine.

## Prerequisites

Before getting started, ensure that you have:

1. A Raspberry Pi device with a correctly installed and configured operating system.
2. Successfully installed the NCNN library and its dependencies. Installation instructions can be found on the [NCNN GitHub](https://github.com/Tencent/ncnn) page.
3. The model files trained with SSCMA, including the configuration file and weight file.
4. Gstreamer installed. Refer to the installation guide. Refer to the [Gstreamer installation guide](https://gstreamer.freedesktop.org/documentation/installing/on-linux.html?gi-language=c).
5. Meson and Ninja build tools installed. Refer to the [Meson documentation](https://mesonbuild.com/Getting-meson.html).
6. Json-glib library installed. Refer to the [Json-glib project page](https://wiki.gnome.org/Projects/JsonGlib).


## Steps

Follow these steps to deploy the model on your Raspberry Pi:

1. Transfer the model files to the Raspberry Pi.

   You can use SCP, SFTP, or any other file transfer tool to transfer the model files from your local computer to the Raspberry Pi. Make sure to place the model files in an appropriate directory for later use.

2. Create an inference application on the Raspberry Pi.

   Create an inference application tailored to your model using the NCNN library. Write C++ code to load the model, perform inference, and output the results. Refer to the NCNN documentation and sample code to learn how to create an inference application for your model.

3. Compile and build the application.

   Compile and build the inference application on the Raspberry Pi using the appropriate compiler and build tools. Ensure that you configure the compilation options correctly and link against the NCNN library.

4. Run the inference application.

   Transfer the compiled executable file to the Raspberry Pi and run the application in the terminal. You can provide input images or other input data as required and observe the inference results.

## Example

### Building the Project
Compile NCNN as a static library:
```bash
git clone https://github.com/Seeed-Studio/sscma-example-pi --recursive
cd components/ncnn
mkdir -p build-aarch64-linux-gnu
pushd build-aarch64-linux-gnu
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-linux-gnu.toolchain.cmake -DNCNN_OPENMP=OFF..
make -j4
make install
popd
```
Compile and copy the gst-sscma-yolov5 plugin:
```bash
meson build
ninja -C build
sudo cp ./build/libgstsscmayolov5.so /usr/lib/aarch64-linux-gnu/gstreamer-1.0/
```
After everything goes well, you will be able to see the plugin information in gst-inspect-1.0:
```bash
gst-inspect-1.0 sscmayolov5
```

### Running the Project
```bash
sscma_yolov5 model={model_path},{weights_path} numthreads={numthreads} input={input} output={output} outputtype={outputtype} labels={labels_path} threshold=2500:0.25

Options:
   --model=model_path,weights_path         Path to model file
   --numthreads=numthreads                 Configuring to model numthreads (default: 4)
   --input=input                           Configuring to model input format (default: 3:320:320)
   --output=output                         Configuring to model output format (default: 85:6300:1:1)
   --outputtype=outputtype                 Configuring to model output type (default: float32)
   --labels=labels_path                    Path to model labels file
   --threshold=threshold:threshold         Configuring to model threshold (default: 2500:0.25)
   --is_output_scaled=is_output_scaled     Configuring to model output is scaled (default: false)
```

### Demo 1
```bash
  gst-launch-1.0 \
  v4l2src name=cam_src ! videoconvert ! videoscale ! \
    video/x-raw,width=1280,height=720,format=RGB,pixel-aspect-ratio=1/1,framerate=30/1 ! \
    sscma_yolov5 model=net/epoch_300_float.ncnn.bin,net/epoch_300_float.ncnn.param labels=net/coco.txt ! \
    videoconvert ! autovideosink
```
#### Explanation
The v4l2src name=cam_src is used to capture real-time video stream from the camera. It can also be changed to the path of any video file.
videoconvert is used for automatic format conversion, and videoscale is used for automatic scaling.
video/x-raw,width=1280,height=720,format=RGB,pixel-aspect-ratio=1/1,framerate=30/1 specifies the output format. The resolution can be any, but it must be in RGB format. More formats will be supported in the future.
autovideosink is used for displaying the output window. sync=false is used for asynchronous display, and it can also be used with other plugins to output to other platforms.

### Demo 2
```bash
  gst-launch-1.0 \
  v4l2src name=cam_src ! videoconvert ! videoscale ! \
    sscma_yolov5 model=net/epoch_300_float.ncnn.bin,net/epoch_300_float.ncnn.param labels=net/coco.txt ! \
    text/x-json ! \
    multifilesink location=./result.json
```
#### Explanation
multifilesink is used to replace the output to a file. location=./result.json specifies the output file path. text/x-json is the output format.
The output format is in JSON format and includes the inference results and inference time, as shown below:
```json
{
  "type": 1,
  "name": "INVOKE",
  "code": 0,
  "data": {
    "count": 8, // Inference result quantity
    "perf": [
      8,       // Inference pre-processing time(ms)
      365,     // Reasoning time, measured(ms)
      0        // Inference post-processing time(ms)
    ],
    "image": "<BASE64JPEG:String>" // Original image, base64 encoded
    "boxes": [
      [
        87,
        83,
        77,
        65,
        70,
        0
      ]
    ]
  }
}
```



## Considerations

- Performing model inference on a Raspberry Pi may be subject to hardware resource limitations. Ensure that your model and input data are compatible with the computational capabilities and memory constraints of the Raspberry Pi.
- Specific tuning and optimization may be required for your model and application to achieve optimal performance.
- Additional dependencies or configurations may be necessary on the Raspberry Pi to meet the requirements of model inference. Refer to the NCNN documentation and relevant Raspberry Pi resources for further assistance.

We hope these steps help you successfully deploy a model trained with SSCMA on a Raspberry Pi using NCNN as the inference engine. Good luck!

## References
[SSCMA](https://github.com/Seeed-Studio/SSCMA)
[NCNN](https://github.com/Tencent/ncnn)
[gstreamer](https://gstreamer.freedesktop.org)