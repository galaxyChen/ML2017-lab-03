import pickle
import extractFeature
import numpy as np
import ensemble as ada
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def load(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
    
def save(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    # extractFeature是自己写的函数，在extractFeature文件中
    extractFeature.extract()
    #读取上一步生成的特征文件
    print("loading features")
    face_data = np.matrix(load('./feature_face'))
    nonface_data = np.matrix(load('./feature_nonface'))
    
    print("divide training set and test set")
    # 选取400个有脸和400个无脸作为训练集
    train_data = np.concatenate((face_data[0:400],nonface_data[0:400]),axis=0)
    y = np.matrix(([1]*400)+([-1]*400)).T
    # 下面这几行是拿来打乱数据的，意义不大，可以删除
    train_data = np.concatenate((y,train_data),axis=1)
    np.random.shuffle(train_data)
    train_set = train_data[:,1:]
    train_y = train_data[:,0]
    # 最后的100个有脸和没脸的数据作为测试集
    test_data = np.concatenate((face_data[400:500],nonface_data[400:500]),axis=0)
    test_y = np.matrix(([1]*100)+([-1]*100)).T
    # 下面这几行是拿来打乱数据的，意义不大，可以删除
    test_data = np.concatenate((test_y,test_data),axis=1)
    np.random.shuffle(test_data)
    test_set = test_data[:,1:]
    test_y = test_data[:,0]
    # 开始训练
    print("ready to train")
    # 基学习器是决策树
    model = DecisionTreeClassifier
    # 基学习器的数量
    num_of_model = 10
    # 声明一个adaboost并进行训练，同时预测测试集
    AdaBoost=ada.AdaBoostClassifier(model,num_of_model)
    acc_history=AdaBoost.fitAndPredict(train_set,train_y,test_set,test_y)
    
    # 最后测试测试集的正确率
    pred=AdaBoost.predict(test_set)
    print("test set accuracy:")
    print((pred==test_y.A1).sum()/len(pred))
    # 生成题目需要的report.txt文件，这个文件要print出来才能看，直接看是很糟糕的排版
    report = classification_report(test_y.A1, pred)
    print(report)
    save(report,'./report.txt')
    # 打印准确率随着学习器数量变化的曲线
    plt.plot(np.arange(num_of_model),acc_history,label="test acc")
    plt.legend(loc=1)
    plt.xlabel('number of base classifier')
    plt.ylabel('test accuracy')