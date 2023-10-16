# 树莓派上部署经过SSCMA训练的模型

本文将介绍如何将经过SSCMA训练的模型部署到树莓派，并使用NCNN作为推理引擎。

## 前提条件

在开始之前，确保您具备以下内容：

1. 一个树莓派设备，已正确安装并配置好操作系统。
2. 已成功安装NCNN库和相关依赖项。可以在[NCNN GitHub](https://github.com/Tencent/ncnn)上找到安装说明。
3. 经过SSCMA训练的模型文件（含配置文件和权重文件）。

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
```bash
git clone https://github.com/Seeed-Studio/sscma-example-pi --recursive
cd sscma-example-pi
mkdir build
cd build
cmake ..
make
```

### 运行推理应用程序
```bash
./sscma-yolov5 
Usage: program_name --model=model_path --weights=weights_path --input=input_path --output=output_path [--thread=thread_num] [--save] [--show] [--classes=class1,class2,class3]...
Options:
  --model=model_path         Path to model file (default: ../models/sscma-yolov8/model.param)
  --weights=weights_path     Path to weights file (default: ../models/sscma-yolov8/model.bin)
  --input=input_path         Path to input file
  --output=output_path       Path to output file
  --thread=thread_num        Number of threads (default: 1)
  --iou=iou_threshold        IoU threshold (default: 0.45)
  --score=score_threshold    Score threshold (default: 0.25)
  --save                     Save result to file (default: false)
  --headless                 Do not show result (default: false)
  --classes=class1           List of classes to detect (default: all classes)
  --input_mean=0,0,0         Input mean (default: 0,0,0)
  --input_std=1,1,1          Input std (default: 0.0039126,0039126,0039126)
  --input_shape=1,3,640,640  Input shape (default: 1,3,640,640)
```

```bash
./sscma-yolov5 --model ../models/yolov5s/yolov5s.param --weights ../models/yolov5s/yolov5s.bin --input ../images/dog.jpg
```


## 注意事项

- 在树莓派上进行模型推理可能受到硬件资源限制的影响。请确保您的模型和输入数据适应树莓派的计算能力和内存限制。
- 需要根据具体模型和应用程序进行适当的调优和优化，以获得最佳性能。
- 可能需要在树莓派上安装其他依赖项或进行额外的配置，以满足模型推理的要求。请参考NCNN文档和树莓派的相关资源以获取更多帮助。

希望这些步骤能帮助您成功地将经过SSCMA训练的模型部署到树莓派上，并使用NCNN作为推理引擎。祝您好运！

## 参考资料
[SSCMA](https://github.com/Seeed-Studio/SSCMA)
[NCNN](https://github.com/Tencent/ncnn)
