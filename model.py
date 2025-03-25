
import cv2
import numpy as np
import os 
import proyect_utils
from abc import ABC, abstractmethod
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    Flatten,
    Dense,
    Dropout,
)
import torch
import onnxruntime  
class Model(ABC):
    def __init__(self):
        self.default_path = os.path.expanduser("~")        
       
    def getEmotion(self): # return the array of emotions
        if isinstance(self.emocion,dict):
            return self.emocion.values()
        return self.emocion
    

    @abstractmethod
    def loadModel(self): # initialize the model 
        pass
   
    @abstractmethod
    def predict(self,input): #predict the input
        pass
    @abstractmethod
    def decode(self,prediction): # return the prediction value
        pass
    @abstractmethod
    def loadEmotion(): #init array emotion
        pass


class deepFace(Model):
    def __init__(self):
        self.modelName = "deepFace"
        super().__init__()
    
    def loadModel(self):#carga las capas  se puede guardar en json
        url = "https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5"
        model = Sequential()

        # 1st convolution layer
        model.add(Conv2D(64, (5, 5), activation="relu", input_shape=(48, 48, 1)))
        model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

        # 2nd convolution layer
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(Conv2D(64, (3, 3), activation="relu"))
        model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

        # 3rd convolution layer
        model.add(Conv2D(128, (3, 3), activation="relu"))
        model.add(Conv2D(128, (3, 3), activation="relu"))
        model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

        model.add(Flatten())

        # fully connected neural networks
        model.add(Dense(1024, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(1024, activation="relu"))
        model.add(Dropout(0.2))

        model.add(Dense(7, activation="softmax"))
        
        model_file = proyect_utils.getFilePath(f"{self.modelName}_weight.h5",url)
        model.load_weights(model_file)
        self.loadEmotion()
        self.model = model
    
    def predict(self,input):
        gray = input
        if len(input.shape) == 3:
            gray= cv2.cvtColor(input,cv2.COLOR_BGR2GRAY) 
        cropped_img = np.expand_dims(cv2.resize(gray, (48,48)), 0)               
        return self.model.predict(cropped_img)
    def decode(self,prediction):
        return self.emocion[int(np.argmax(prediction))]
    
    def printSummary(self):
        self.model.summary()
    
    def loadEmotion(self):
        self.emocion = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"] 
    


class atulapra(Model):
    
    def __init__(self):
        self.modelName = "atulapra"
        super().__init__()
    
    def loadModel(self):#carga las capas  se puede guardar en json
        url ="https://drive.google.com/uc?id=1FUn0XNOzf-nQV7QjbBPA6-8GLoHNNgv-"
        model = Sequential()

        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))
        
        model_file = proyect_utils.getFilePath(f"{self.modelName}_weight.h5",url)
        model.load_weights(model_file)
        
        self.model = model
        self.loadEmotion()
    
    def predict(self,input):
        gray = input
        if len(input.shape) == 3:
            gray= cv2.cvtColor(input,cv2.COLOR_BGR2GRAY) 
        cropped_img = np.expand_dims(cv2.resize(gray, (48,48)), 0)               
        return self.model.predict(cropped_img)
    def decode(self,prediction):
        return self.emocion[int(np.argmax(prediction))]
    
    def printSummary(self):
        self.model.summary()
    
    def loadEmotion(self):
        self.emocion = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"] 

class otaha(Model):  
    def __init__(self):
        self.modelName = "otaha"
        super().__init__()
    
    def loadModel(self):#carga las capas  se puede guardar en json
        from keras.models import load_model
        url ="https://drive.google.com/uc?id=1QksjWlwyRDM3--tZBSlHWIIGTRogc0uK"
     
        model_file = proyect_utils.getFilePath(f"{self.modelName}_weight.h5",url)
        self.model = load_model(model_file, compile=False)
        self.loadEmotion()
    
    def predict(self,input):
        gray = input
        if len(input.shape) == 3:
            gray= cv2.cvtColor(input,cv2.COLOR_BGR2GRAY) 
        gray = cv2.resize(gray, (64,64))
        gray = gray.astype("float")/255.0
        cropped_img = np.expand_dims(gray, 0)     
        return self.model.predict(cropped_img)
    
    def decode(self,prediction):
        return self.emocion[int(np.argmax(prediction))]
    
    def printSummary(self):
        self.model.summary()
    
    def loadEmotion(self):
        self.emocion = ["angry" ,"disgust","scared", "happy", "sad", "surprised","neutral"]

class trpakov(Model):

    def __init__(self):
        self.modelName = "trpakov"
        super().__init__()
    
    def loadModel(self):
        from transformers import AutoImageProcessor , AutoModelForImageClassification
        self.extractor = AutoImageProcessor.from_pretrained("trpakov/vit-face-expression")
        self.model = AutoModelForImageClassification.from_pretrained(
            "trpakov/vit-face-expression"
        )
        self.loadEmotion()
        
    def predict(self,input):
        im = input
        if len(input.shape)!=3:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        inputs = self.extractor(images=im, return_tensors="pt")

        outputs = self.model(**inputs)

        probabilities = torch.nn.functional.softmax(outputs.logits,dim=-1)
        probabilities = probabilities.detach().numpy().tolist()[0]
        return probabilities
    
    def decode(self,prediction):
        return self.emocion[int(np.argmax(prediction))]
    
    def loadEmotion(self):
        from transformers import AutoConfig # lazy load , only load if it call the function
        self.emocion = AutoConfig.from_pretrained("trpakov/vit-face-expression").id2label


class Em(Model): # se le cambia el nombre despues
    def __init__(self):
        self.modelName = "em"
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def loadModel(self):
        url = "https://drive.google.com/uc?id=1zwH8S4VoghbJDTbfnLlE3C8ZS7RF_APj"
        model_file = proyect_utils.getFilePath(f"{self.modelName}.onnx",url)
        model = onnxruntime.InferenceSession(model_file)
        self.model = model
        self.loadEmotion()
    
    def predict(self,input):
        import torchvision.transforms as transforms
        from PIL import Image
        img = Image.fromarray(input)
        img = img.convert("RGB")
        img = img.resize((224, 224), Image.Resampling.NEAREST)
        img = transforms.PILToTensor()(img)
        img = img.to(torch.float32)
        img = torch.flip(img, dims=(0,))
        img[0, :, :] -= 91.4953
        img[1, :, :] -= 103.8827
        img[2, :, :] -= 131.0912
        img = torch.unsqueeze(img, 0).to(self.device)
       
        outputs =  self.model.run(None,{self.model.get_inputs()[0].name:img.to(self.device).numpy()})
        probabilities = outputs[0]
        return probabilities
    
    def decode(self,prediction):
        return self.emocion[int(np.argmax(prediction))]
    def loadEmotion(self):
        self.emocion = ['Neutral', 'Happiness','Sadness','Surprise','Fear','Disgust','Anger']