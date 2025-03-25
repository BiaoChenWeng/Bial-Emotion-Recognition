
from stream import predecir
import model
import argparse
import facialRecognitionModel

argp = argparse.ArgumentParser()    
 
argp.add_argument("-m","--model",help="Choose a model to predict emotion the default model is : deepFace",default="deepface")    
argp.add_argument("-i","--input",help="Choose a input so the model can evaluate",default=0)
argp.add_argument("-c","--FR_Model",help="Choose a facial recognition model default = mediapipe",default="m")
argp.add_argument("-s","--saveVideo",action="store_true",help= "Save the video if the flag is activate")
argp.add_argument("-d","--displayEmotion",action="store_false",help= "Display the emotion on your screen")
arg = argp.parse_args()

EMOTION_MODELS = {
    "deepface":model.deepFace(),
    "d": model.deepFace(),
    "otaha":model.otaha(),
    "o":model.otaha(),
    "atulapra": model.atulapra(),
    "a": model.atulapra(),
    "t":model.trpakov(),
    "trpakov": model.trpakov(),
    "e": model.Em()
}

FR_MODELS={
    "cv2": facialRecognitionModel.cv2Classfier(),
    "c":facialRecognitionModel.cv2Classfier(),
    "m":facialRecognitionModel.mediaPipe()
}


modelo = EMOTION_MODELS[arg.model.lower()]
frModel = FR_MODELS[arg.FR_Model.lower()]
modelo.loadModel()
frModel.loadModel()
saveVideo = arg.saveVideo
displayEmotion = arg.displayEmotion
predecir(modelo,frModel,arg.input,saveVideo,displayEmotion)