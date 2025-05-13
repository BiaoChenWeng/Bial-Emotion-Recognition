from abc import ABC, abstractmethod
from proyect_utils import getFilePath
import cv2

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
            if(w!= 0 and h != 0 ):
                faces.append(faceInfo(gray[y:y + h, x:x + w],x,y,w,h))
        return faces

class mediaPipe(facialRecognitionModel):
    def loadModel(self):
        import mediapipe as mp

        mp_face = mp.solutions.face_detection
        self.model = mp_face.FaceDetection(
            model_selection=1, 
            min_detection_confidence=0.5)
    
    def getFaces(self,frame):
        h,w, _ = frame.shape
        frame_copy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model.process(frame_copy)
        faces = []
        if not results.detections:
            return faces
        for detect in results.detections:
            bbox = detect.location_data.relative_bounding_box
            x, y = int(bbox.xmin * w), int(bbox.ymin * h)
            width, height = int(bbox.width * w), int(bbox.height * h)

            faces.append(faceInfo(frame_copy[y:y + height, x:x + width],x,y,width,height))
        return faces



class faceInfo: 
    def __init__(self,face,x,y,w,h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.face = face
    def getFace(self):
        return self.face
    def getX(self):
        return self.x
    def getY(self):
        return self.y
    def getW(self):
        return self.w
    def getH(self):
        return self.h