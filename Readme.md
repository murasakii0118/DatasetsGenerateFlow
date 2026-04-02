## 这是一个我用来利用视频生成适用于AI模型训练的流程工具
目前只适用于Windows平台(只是批处理文件的差异,其他语言源代码可以正常编译的)
### 依赖:
[TransNetv2](https://github.com/soCzech/TransNetV2)

[OpenCV v4.12](https://opencv.org/)

[OpenH264 v1.8.0](https://github.com/cisco/openh264)

[llama.cpp](https://github.com/ggml-org/llama.cpp)

Qwen3模型:Qwen3VL-4B-Instruct-Q8_0.gguf以及mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf

### 注意 
- openh264-1.8.0-win64.dll应放在根目录下
- TransNetv2 你应该克隆到 step 1.split scenes(TransNet) 目录下 最重要的是inference里的内容
- 尽量不使用非英文构成的路径,我无法保证在包含其他语言的路径下我的程序能正常跑
- 最好是有mingw的环境,避免缺这少那
- 将Qwen模型放在根目录的assets文件夹下,如果没有,请创建
- **如果在step.5长时间卡死,可能是模型在说梦话**
- **反推提示词仅供参考,不建议直接使用,强烈建议必须人工审核一遍**

### 已通过环境
windows 11 23H2/gcc 13.2(c++17)/python 3.11

### 怎么用
llama.cpp需自行编译
你只需要把视频放到InputVideo目录下即可,然后从step.1 一直到step.5 一路双击即可,视频文件允许使用非英文,因为程序会自动修改

### 为什么我的数据集没有按视频名称排列而是变成了0001这样的
还是那句话,C++对非英文支持的太烂,一输入其他语言的路径就报错,不过你可以根据生成的000x的索引从videoSegments里找到由原来名字构成的TXT文档

### 此项目正在缓慢更新,可能会产生一些bug 
### 如果你有什么idea可以提pr或者issue

### 文件结构
```txt
DatasetsGenerateFlow.
│  .gitignore
│  openh264-1.8.0-win64.dll
│  Readme.md
│  requirement.txt
│  Step.1-splitScenes.bat
│  step.2-splitVideo(opencv).bat
│  step.3-preliminaryFrame.bat
│  step.4-precisionFrame.bat
│  step.5-generateFrame.bat
│ 
├─assets
│      mmproj-Qwen3VL-4B-Instruct-Q8_0.gguf
│      Qwen3VL-4B-Instruct-Q8_0.gguf
│      
├─InputVideo
│      
├─precisionFrame
│          
├─preliminaryFrame
│              
├─Scenes
│      
├─step 1.split scenes(TransNet)
│  │  put config&inference dictionary here
│  │  
│  ├─configs
│  │      transnetv1.gin
│  │      transnetv2-realtrans.gin
│  │      transnetv2.gin
│  │      
│  └─inference
│      │  Dockerfile
│      │  README.md
│      │  transnetv2.py
│      │  __init__.py
│      │  
│      └─transnetv2-weights
│          │  saved_model.pb
│          │  
│          └─variables
│                  variables.data-00000-of-00001
│                  variables.index
│                  
├─step 2.splitVideo(opencv)
│      Splitter.py
│      
├─step 3.preliminaryFrame
│      libopencv_core4120.dll
│      libopencv_imgcodecs4120.dll
│      libopencv_imgproc4120.dll
│      libopencv_videoio4120.dll
│      opencv_videoio_ffmpeg4120_64.dll
│      pf.py
│      preliminaryFrame.cpp
│      preliminaryFrame.exe
│      
├─step 4.precisionFrame
│      classify.exe
│      classify.py
│      libopencv_core4120.dll
│      libopencv_imgcodecs4120.dll
│      libopencv_imgproc4120.dll
│      
├─step 5.generatePrompt
│      common.dll
│      ggml-base.dll
│      ggml-cpu-x64.dll
│      ggml-cuda.dll
│      ggml.dll
│      inferprompt.cpp
│      inferprompt.exe
│      llama.dll
│      mtmd.dll
│          
└─videoSegments

```