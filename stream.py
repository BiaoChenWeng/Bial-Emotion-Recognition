
import cv2
import salida
from model import Model

from facialRecognitionModel import facialRecognitionModel
def predecir(model:Model,FR_Model:facialRecognitionModel,source = 0,saveVideo =False , displayEmotion= True):
    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)
    cap = cv2.VideoCapture(source if isinstance(source,str) else int(source))
    if not cap.isOpened():
        ValueError("I can't be opened {}".format(source))
    

    
    output= "neutral"
    windowSize = 3
    delay = 3
    counter =windowSize
    lastValue = output
    newValue = output
    outputData = salida.outputManagement(model.getEmotion())
    try : 
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # faces = FR_Model.getFaces(frame)
            # num_faces = len(faces)

            # if num_faces == 1:
            #     face = faces[0]
            #     input_face = face.getFace()
            #     pred = model.predict(input_face)
            #     output = model.decode(pred).lower()

            #     if output == lastValue or counter >= windowSize:
            #         newValue = lastValue = output
            #         counter += 1
            #         outputData.preparar_output(pred, output)
            #     elif newValue == output:
            #         counter += 1
            #         output = lastValue
            #     else:
            #         newValue = output
            #         counter = 1

            # elif num_faces > 1 and counter >= delay:
            #     counter = (counter + 1) % delay
            #     pred = model.predict(faces[0].getFace())  # Usa la primera cara detectada
            #     output = model.decode(pred).lower()
            #     outputData.preparar_output(pred, output)

            # for face in faces:
            #     cv2.putText(frame, str(output), (face.getX() + 20, face.getY() - 60),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            #if saveVideo or displayEmotion:
            #    resizedFrame = cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC)

            if saveVideo:
                outputData.saveFrame(frame)

            if displayEmotion:
                cv2.imshow('Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
    except Exception as e : 
        print("Error: {}".format(e))
    except KeyboardInterrupt as k : 
        print("END")    
    outputData.sacar_output()
    outputData.generateVideo()
    cap.release()
    cv2.destroyAllWindows()

        
        # while True: 
        #     ret,frame = cap.read()
        #     if not ret :
        #         break
        #     #frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #     faces = FR_Model.getFaces(frame)
            
        #     for face in faces: 
        #         #cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        #         input = face.getFace()
        #         if len(faces) ==1: # it only detected a face
        #             pred = model.predict(input)
        #             output = model.decode(pred).lower()
        #             if output == lastValue or counter >=windowSize: # if the emotion didnt change or the new emotion is stable
        #                 newValue= lastValue = output
        #                 counter+=1
        #                 outputData.preparar_output(pred, output) 
        #             elif newValue == output: # the new emotion is equal to the prediction 
        #                 counter+=1
        #                 output = lastValue# give the last prediction in case the model is unstable
        #             else : # reset the counter the new emotion was unstable
        #                 newValue = output 
        #                 counter = 1
        #         elif counter >= delay:  # in case it have more than one face we put a delay in the prediction 
        #             counter= (counter+1)%delay
        #             pred = model.predict(input)
        #             output = model.decode(pred).lower()
        #             outputData.preparar_output(pred, output)
        #         cv2.putText(frame, str(output),(face.getX()+20, face.getY()-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            
        #     #if saveVideo or displayEmotion:
        #     #resizedFrame= cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC)
        #     if saveVideo:
        #         outputData.saveFrame(frame)
        #     if displayEmotion:     
        #         cv2.imshow('Video', frame)
        #         if cv2.waitKey(1) & 0xFF == ord('q'):
        #             break