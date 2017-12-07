import pickle
import extractFeature
import numpy as np
import ensemble as ada
from sklearn.tree import DecisionTreeClassifier

def load(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    # extract feature and save features to file
    extractFeature.extract()
    #load features
    print("loading features")
    face_data = np.matrix(load('./feature_face'))
    nonface_data = np.matrix(load('./feature_nonface'))
    
    print("divide training set and test set")
    train_data = np.concatenate((face_data[0:400],nonface_data[0:400]),axis=0)
    y = np.matrix(([1]*400)+([-1]*400)).T
    train_data = np.concatenate((y,train_data),axis=1)
    np.random.shuffle(train_data)
    train_set = train_data[:,1:]
    train_y = train_data[:,0]
    
    test_data = np.concatenate((face_data[400:500],nonface_data[400:500]),axis=0)
    test_y = np.matrix(([1]*100)+([-1]*100)).T
    test_data = np.concatenate((test_y,test_data),axis=1)
    np.random.shuffle(test_data)
    test_set = test_data[:,1:]
    test_y = test_data[:,0]

    print("ready to train")
    model = DecisionTreeClassifier
    AdaBoost=ada.AdaBoostClassifier(model,1)
    AdaBoost.fit(train_set,train_y)
    pred=AdaBoost.predict(test_set)
    print("test set accuracy:")
    print((pred==test_y.A1).sum()/len(pred))