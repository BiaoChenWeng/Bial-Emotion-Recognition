## Bial Emotion Recognition
An AI-powered tool that detects facial expressions from a live camera or video .
Built with PyTorch and Keras, it  identify emotions such as happiness, sadness, anger, surprise, disgust and fear.

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


## Usage

```bash
$ python main.py 
```

