
from predictionMode import batchPrediction,streamPrediction
import emotionRecognitionModel
import argparse
import facialRecognitionModel

argp = argparse.ArgumentParser()    
 
argp.add_argument("-m","--model",help="Choose a model to predict emotion the default model is : trpakov",default="t")    
argp.add_argument("-i","--input",help="Choose a input so the model can evaluate",default=0)
argp.add_argument("-c","--FR_Model",help="Choose a facial recognition model default = mediapipe",default="c")
argp.add_argument("-d","--display",action="store_true",help= "Display the emotion on the screen",default= True)
arg = argp.parse_args()

EMOTION_MODELS = {
    "deepface":emotionRecognitionModel.deepFace(),
    "d": emotionRecognitionModel.deepFace(),
    "otaha":emotionRecognitionModel.otaha(),
    "o":emotionRecognitionModel.otaha(),
    "atulapra": emotionRecognitionModel.atulapra(),
    "a": emotionRecognitionModel.atulapra(),
    "t":emotionRecognitionModel.trpakov(),
    "trpakov": emotionRecognitionModel.trpakov(),
    "e": emotionRecognitionModel.Em()
}

FR_MODELS={
    "cv2": facialRecognitionModel.cv2Classfier(),
    "c":facialRecognitionModel.cv2Classfier(),
    "m":facialRecognitionModel.mediaPipe(),
    "mediapipe":facialRecognitionModel.mediaPipe()
}



if arg.model.lower() in EMOTION_MODELS: 
    model = EMOTION_MODELS[arg.model.lower()]
else:
    print("ERModel {}, don't found".format(arg.model.lower()))
    exit()
if arg.FR_Model.lower() in FR_MODELS: 
    frModel = FR_MODELS[arg.FR_Model.lower()]
else:
    print("FRModel {}, don't found".format(arg.FR_Model.lower()))
    exit()


model.loadModel()
frModel.loadModel()


if(arg.display):
    streamPrediction(model,frModel,arg.input)
else :
    batchPrediction(model,frModel,arg.input)
