from abc import ABC, abstractmethod
from proyect_utils import getFilePath
import cv2
import math
import numpy as np
class facialRecognitionModel(ABC) : 
    @abstractmethod
    def loadModel(self):  
        pass   
    @abstractmethod
    def getFaces(self):  
        pass  

class cv2Classfier(facialRecognitionModel):
    def loadModel(self):
        filePath = getFilePath("haarcascade_frontalface_default.xml","https://drive.google.com/uc?id=1Pe38Uc_dQAlNJ9aNPxOMbGH3zPcznxs4")
        self.model = cv2.CascadeClassifier(filePath)
    
    def getFaces(self,frame):
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        result = self.model.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)  
        faces = []
        for (x,y,w,h) in result: 
            faces.append(faceInfo(gray[y:y + h, x:x + w],x,y))
        return faces

class mediaPipe(facialRecognitionModel):
    def loadModel(self):
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        self.model = mp_face_mesh.FaceMesh(
            max_num_faces = 5,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
    
    def getFaces(self,frame):
        h,w, _ = frame.shape
        frame_copy = frame.copy()
        frame_copy.flags.writeable = False
        frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        results = self.model.process(frame_copy)
        frame_copy.flags.writeable = True
        faces = []
        value = {}
        #cambio al mediapipe
        results.multi_face_landmarks = results.multi_face_landmarks or []
        #
        for fl in results.multi_face_landmarks:
            for idx, landmark in enumerate(fl.landmark):
                landmark_px = self.norm_coordinates(landmark.x, landmark.y, w, h)
                if landmark_px:
                    value[idx] = landmark_px
                    
            startX = np.min(np.asarray(list(value.values()))[:,0])
            startY = np.min(np.asarray(list(value.values()))[:,1])
            endX = np.max(np.asarray(list(value.values()))[:,0])
            endY = np.max(np.asarray(list(value.values()))[:,1])

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            faces.append(faceInfo(frame_copy[startX:endY,startY:endX],startX,startY))
        return faces

    def norm_coordinates(self,normalized_x, normalized_y, image_width, image_height):
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px, y_px


class faceInfo: 
    def __init__(self,face,x,y):
        self.x = x
        self.y = y
        self.face = face
    def getFace(self):
        return self.face
    def getX(self):
        return self.x
    def getY(self):
        return self.y