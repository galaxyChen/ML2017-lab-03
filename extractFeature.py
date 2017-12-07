from feature import NPDFeature
from PIL import Image
import numpy as np
import pickle
import os


def save(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)

    
def extract():
    if not os.path.exists('./feature_face'):
        features_face = []
        print("extra face feature")
        print("")
        for i in range(0,500):
            name = str(i)
            while len(name)<3:
                name = '0'+name
            img = Image.open('./datasets/original/face/face_%s.jpg'%name).convert('L').resize((24,24))
            f = NPDFeature(np.array(img))
            feature = f.extract()
            features_face.append(feature)
    
        
        save(features_face,'./feature_face')    
    
    if not os.path.exists('./feature_nonface'):
        features_nonface = []
        print("extra nonface feature")
        print("")
        for i in range(0,500):
            name = str(i)
            while len(name)<3:
                name = '0'+name
            img = Image.open('./datasets/original/nonface/nonface_%s.jpg'%name).convert('L').resize((24,24))
            f = NPDFeature(np.array(img))
            feature = f.extract()
            features_nonface.append(feature)
    
        
        save(features_nonface,'./feature_nonface')
    print("extract feature done!")



