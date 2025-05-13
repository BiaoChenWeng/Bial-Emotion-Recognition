# Bial Emotion Recognition

## Authors

- Biao Chen Weng
- Álvaro Llera Calderón

## Overview

Bial Emotion Recognition uses convolutional neural networks (CNNs) trained on facial expression datasets to analyze real-time  video input or pre-recorded video files. 
The system performs the followig : 
 * Detect faces in video using OpenCV
 * Classifies facial expressions
 * Display the result
 * The result is save in a CSV file: TimeStamp , predicted emotion label , probabilites for all emotions classes.

## Models
| Model | Link |
| --- | --- |
| trpakov | https://huggingface.co/trpakov/vit-face-expression |
| deepface | https://github.com/serengil/deepface |
| atulapra | https://github.com/atulapra/Emotion-detection |
|EMO-AffectNetModel|https://github.com/ElenaRyumina/EMO-AffectNetModel|
## Installation

We recommend using a virtual environment to avoid interference with other libraries you may have installed on your computer.

To do this, you need to install the library that allows you to create virtual environments. The command to install it is the following (you only need to download it on your computer once):

~~~
pip install virtualenv
~~~

Once installed, you can download the repository or clone it into an empty folder using:

~~~
git clone https://github.com/BiaoChenWeng/Bial-Emotion-Recognition.git
~~~

With the repository ready, open a terminal in the `code` folder and run the following commands to create and activate the environment:

~~~
py -3 -m venv env
env/Scripts/activate
~~~

To download all the necessary libraries, you can either install them individually by checking the `requirements.txt` file or run this command in the terminal:

~~~
pip install deepface torch onnxruntime transformers mediapipe torchvision
~~~

You now have everything you need to run the program.

## Execution

The program comes ready with 4 emotion detection models and 2 face detection models, all set up and ready to use at any time.

Running the code is as simple as using the following command:

~~~
python .\main.py [options]
~~~

The options include:
- -m model : used to choose one of the 3 emotion detection models. By default, it uses Trpakov. (Check the EMOTION_MODELS array in main.py to see which values to use for the model).
- -i video_path : used to select a video to process. (Valid formats: .mp4 and .mkv). By default, it uses the computer camera.
- -c model : used to choose one of the 2 face detection models. By default, it uses CV2. (Check the FR_MODELS array in main.py to see which values to use for the model).
- -d : disable on-screen emotion display.
