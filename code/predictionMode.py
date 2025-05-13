
import cv2
import outputFormat
from emotionRecognitionModel import Model

from facialRecognitionModel import facialRecognitionModel

def getVideoInput(source):
    cv2.ocl.setUseOpenCL(False)# prevents openCL usage and unnecessary logging messages
    cap = cv2.VideoCapture(source if isinstance(source,str) else int(source))
    if not cap.isOpened():
        ValueError("I can't be opened {}".format(source))  
    return cap

def putEmotionOnFrame(frame, emotion,x,y,w,h):
     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
     cv2.putText(frame, str(emotion), (x + 20, y - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)


def streamPrediction(model:Model,FR_Model:facialRecognitionModel,source = 0 ):
    
    cap = getVideoInput(source)
    output= "neutral"
    windowSize = 3
    delay = 3
    counter =windowSize
    lastValue = output
    newValue = output
    outputData = outputFormat.outputManagement(model.getEmotion())
    try : 
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            faces = FR_Model.getFaces(frame)
            num_faces = len(faces)

            if num_faces == 1:
                face = faces[0]
                input_face = face.getFace()
                pred = model.predict(input_face)
                output = model.decode(pred).lower()

                if output == lastValue or (counter >= windowSize and output == newValue ):
                    newValue = lastValue = output
                    outputData.preparar_output(pred, output)
                elif newValue == output:
                    output = lastValue
                else:
                    newValue = output
                    output = lastValue
                    counter = 0
                putEmotionOnFrame(frame,output,face.getX() , face.getY(),face.getW(),face.getH())  

            elif counter >= delay:
                counter =0
                for face in faces:
                    pred = model.predict(face.getFace())  
                    output = model.decode(pred).lower()
                    outputData.preparar_output(pred, output)
                    putEmotionOnFrame(frame,output,face.getX() , face.getY(),face.getW(),face.getH()) 

            counter = counter + 1
        
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    except Exception as e : 
        print("Error: {}".format(e))
    except KeyboardInterrupt as k : 
        print("END")    
    outputData.sacar_output()
    cap.release()
    cv2.destroyAllWindows()


def prepareTimeStamp(timestamp_ms):
    
    hours = int(timestamp_ms // (3600 * 1000))
    minutes = int((timestamp_ms % (3600 * 1000)) // (60 * 1000))
    seconds = int((timestamp_ms % (60 * 1000)) // 1000)
    milliseconds = int(timestamp_ms % 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02}:{milliseconds:03}"



def batchPrediction(model:Model,FR_Model:facialRecognitionModel,source):
    cap = getVideoInput(source)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    output= "neutral"
    windowSize = 3
    delay = 3
    counter =windowSize
    lastValue = output
    newValue = output
    totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(totalFrames)
    outputData = outputFormat.outputManagement(model.getEmotion())
    pbar = 0 
    try : 
        while True:

            ret, frame = cap.read()
            if not ret:
                break
            faces = FR_Model.getFaces(frame)
            num_faces = len(faces)

            if num_faces == 1:
                face = faces[0]
                input_face = face.getFace()
                pred = model.predict(input_face)
                output = model.decode(pred).lower()

                if output == lastValue or (counter >= windowSize and output == newValue ):
                    newValue = lastValue = output
                    outputData.preparar_output(pred, output,prepareTimeStamp(cap.get(cv2.CAP_PROP_POS_MSEC)))
                elif newValue != output:
                    newValue = output
                    output = lastValue
                    counter = 0

            elif counter >= delay:
                counter =0
                for face in faces:
                    pred = model.predict(face.getFace())  
                    
                    output = model.decode(pred).lower()
                    outputData.preparar_output(pred, output,prepareTimeStamp(cap.get(cv2.CAP_PROP_POS_MSEC)))

            counter = counter + 1
            pbar= pbar+1
            print(f"{(pbar*100)/totalFrames:.2f}")
        
    except Exception as e : 
        print("Error: {}".format(e))
    except KeyboardInterrupt as k : 
        print("END")    
    outputData.sacar_output()
    cap.release()
    
    cv2.destroyAllWindows()


