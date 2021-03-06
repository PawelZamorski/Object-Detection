# Object Detection

Darknet and YOLOV4 Installation Documentation

YOLOv4 uses Cuda 10.0 and OpenCV library. To build a project on Windows it uses Visual Studio.

1) Clone repository from official Github YOLOv4 release: https://github.com/augmentedstartups/darknet
2) Copy OpenCV ffmpeg and world dlls for 4.1.0 version(can be used for OpenCV video processing) to the directory containing darknet.exe
3) Copy cuDNN dll(for speedup neural network process) to the directory containing darknet.exe
4) Change the cuda version if you’re not using Cuda 10.0.
made change in two files:
- darknet.vcxproj [Line:55]
- yolo_cpp_dll.vcxproj [Line:309]
5) Compile yolo_cpp_dll.vcxproj with updated Cuda version
Open yolo_cpp_dll.vcxproj with VS
Open VS and Select Release and x64
Right click on yolo_cpp_dll and select build
6) Compile darknet.sln
Open darknet.sln with VS
Set Release and x64
Right click on darknet and select properties
Make Four changes
C/C++ -> General -> Additional Include Directories: C:\OpenCV\build\install\include Click Ok and Apply 

If you followed my instructions then path should be same - C/C++ -> Preprocessor -> Preprocessor Definitions Remove CUDNN_HALF as Yolo v4/3 uses Tensor cores We don’t need to use it for 1080Ti [Optional] If you have GPU with Tensor Cores (nVidia Titan V / Tesla V100 / DGX-2 and later) then add CUDNN_HALF for speedup in detection 3x and training 2x.
CUDA C/C++ -> Device -> Code Generation Remove compute_75, sm_75 if you’re not using Cuda 10.0 - Linker -> General Add lib from OpenCV build C:\OpenCV\build\install\x64\vc16\lib
Right click on darknet and select build
7) Run Detection on Images
	darknet.exe detector test cfg/coco.data cfg/yolov4.cfg yolov4.weights It’ll ask for the input path of the image
8) Run Detection on Videos
	darknet.exe detector demo cfg/coco.data cfg/yolov4.cfg yolov4.weights PATH_TO_THE_VIDEO
9) Run script darknet_video_team.py
Copy script to the directory containing darknet.exe
Change path to the video.
Run script
	python darknet_video_team.py
