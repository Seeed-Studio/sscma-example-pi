# Deploying a Model Trained with SSCMA on Raspberry Pi

This guide explains how to deploy a model trained with SSCMA (to be confirmed) on a Raspberry Pi using NCNN as the inference engine.

## Prerequisites

Before getting started, ensure that you have:

1. A Raspberry Pi device with a correctly installed and configured operating system.
2. Successfully installed the NCNN library and its dependencies. Installation instructions can be found on the [NCNN GitHub](https://github.com/Tencent/ncnn) page.
3. The model files trained with SSCMA, including the configuration file and weight file.

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
```bash
git clone https://github.com/Seeed-Studio/sscma-example-pi --recursive
cd sscma-example-pi
mkdir build
cd build
cmake ..
make
```

### Running the Inference Application
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

## Considerations

- Performing model inference on a Raspberry Pi may be subject to hardware resource limitations. Ensure that your model and input data are compatible with the computational capabilities and memory constraints of the Raspberry Pi.
- Specific tuning and optimization may be required for your model and application to achieve optimal performance.
- Additional dependencies or configurations may be necessary on the Raspberry Pi to meet the requirements of model inference. Refer to the NCNN documentation and relevant Raspberry Pi resources for further assistance.

We hope these steps help you successfully deploy a model trained with SSCMA on a Raspberry Pi using NCNN as the inference engine. Good luck!

## References
[NCNN GitHub](https://github.com/Tencent/ncnn)