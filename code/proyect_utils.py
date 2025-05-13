import os 
import gdown

proyect_name = ".BER"

def createFolder():
    proyect_home_path = getHome()
    weigths_path = getWeightPath()
    if not os.path.exists(proyect_home_path):
        os.makedirs(proyect_home_path,exist_ok=True)
    if not os.path.exists(weigths_path):
        os.makedirs(weigths_path,exist_ok=True)

def getHome():
    return os.path.join(os.path.expanduser('~'),proyect_name)
def getWeightPath():
    return os.path.join(getHome(),"weights")



def getFilePath(modelName, file_url):    
    if not modelName:
        return 
    
    path = getWeightPath()
    model_file = os.path.normpath(os.path.join(path,modelName))
    if not os.path.isfile(model_file):
        createFolder()
        gdown.download(file_url,model_file,quiet=False)
   
    return model_file

