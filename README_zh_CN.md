# 树莓派上部署经过SSCMA训练的模型

本文将介绍如何将经过SSCMA训练的模型通过gstreamer插件的形式部署到树莓派，并使用NCNN作为推理引擎。

## 前提条件

在开始之前，确保您具备以下内容：

1. 一个树莓派设备，已正确安装并配置好操作系统。
2. 经过SSCMA训练的模型文件（含配置文件，权重文件和标签文件）。
3. 安装gstreamer。参考[这里](https://gstreamer.freedesktop.org/documentation/installing/on-linux.html?gi-language=c)。
4. 安装meson和ninja编译工具。参考[这里](https://mesonbuild.com/Getting-meson.html)。
5. 安装json-glib库。参考[这里](https://wiki.gnome.org/Projects/JsonGlib)。

## 步骤

遵循以下步骤将模型部署到树莓派上：

1. 将模型文件传输到树莓派。

   您可以使用 SCP、SFTP 或其他文件传输工具将模型文件从本地计算机传输到树莓派。确保将模型文件放在合适的目录中，以便后续使用。

2. 在树莓派上创建推理应用程序。

   基于NCNN库创建一个适用于您的模型的推理应用程序。您可以编写C++代码来加载模型、进行推理并将结果输出。请参考NCNN的文档和示例代码以了解如何创建适用于您的模型的推理应用程序。

3. 编译和构建应用程序。

   使用适当的编译器和构建工具，在树莓派上编译和构建推理应用程序。确保正确配置编译选项和链接NCNN库。

4. 运行推理应用程序。

   将编译生成的可执行文件传输到树莓派上，并在终端中运行该应用程序。您可以根据需要提供输入图像或其他输入数据，并查看推理结果。

## 示例

### 编译工程
编译拷贝gst-sscma-yolov5插件
```bash
meson build
ninja -C build
sudo cp ./build/libgstsscmayolov5.so /usr/lib/aarch64-linux-gnu/gstreamer-1.0/
```
一切顺利后将能在gst-inspect-1.0中看到插件信息
```bash
gst-inspect-1.0 sscmayolov5
```

### 运行推理插件
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
### 示例1
```bash
  gst-launch-1.0 \
  v4l2src name=cam_src ! videoconvert ! videoscale ! \
    video/x-raw,width=1280,height=720,format=RGB,pixel-aspect-ratio=1/1,framerate=30/1 ! \
    sscma_yolov5 model=net/epoch_300_float.ncnn.bin,net/epoch_300_float.ncnn.param labels=net/coco.txt ! \
    videoconvert ! autovideosink
```
#### 说明
其中v4l2src name=cam_src为获取摄像头实时视频流，也可以改为任意视频文件路径，
videoconvert为自动格式转换，videoscale为自动缩放，
video/x-raw,width=1280,height=720,format=RGB,pixel-aspect-ratio=1/1,framerate=30/1为指定输出格式，分辨大小可为任意，但是必须为RGB格式，后续会支持更多格式。
sscma_yolov5为此插件，ximagesink为显示窗口，sync=false为异步显示，也可以任意插件输出到其他平台。

### 示例2
```bash
  gst-launch-1.0 \
  v4l2src name=cam_src ! videoconvert ! videoscale ! \
    sscma_yolov5 model=net/epoch_300_float.ncnn.bin,net/epoch_300_float.ncnn.param labels=net/coco.txt ! \
    text/x-json ! \
    multifilesink location=./result.json
```
#### 说明
multifilesink为替换输出到文件，location=./result.json为输出文件路径，text/x-json为输出格式。
输出格式为json格式，包含推理结果和推理耗时，如下所示：
```json
{
  "type": 1,
  "name": "INVOKE",
  "code": 0,
  "data": {
    "count": 8, // 推理结果数量
    "perf": [
      8,       // 推理前处理耗时，单位ms
      365,     // 推理耗时，单位ms
      0        // 推理后处理耗时，单位ms
    ],
    "image": "<BASE64JPEG:String>" // 原始图片，base64编码
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

## 注意事项

- 在树莓派上进行模型推理可能受到硬件资源限制的影响。请确保您的模型和输入数据适应树莓派的计算能力和内存限制。
- 需要根据具体模型和应用程序进行适当的调优和优化，以获得最佳性能。
- 可能需要在树莓派上安装其他依赖项或进行额外的配置，以满足模型推理的要求。请参考NCNN文档和树莓派的相关资源以获取更多帮助。

希望这些步骤能帮助您成功地将经过SSCMA训练的模型部署到树莓派上，并使用NCNN作为推理引擎。祝您好运！

## 参考资料
[SSCMA](https://github.com/Seeed-Studio/SSCMA)
[NCNN](https://github.com/Tencent/ncnn)
[gstreamer](https://gstreamer.freedesktop.org)

## 待办事项
- [X] 插件支持任意输入尺寸
- [X] 推理结果阈值可配置，模型输出是否归一化可配置
- [X] 自动匹配两种输出格式 1：输出带框原始图片 2：输出json格式结果
