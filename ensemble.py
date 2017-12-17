import pickle
import numpy as np

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.model = weak_classifier
        self.M = n_weakers_limit
        pass

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Returns:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        print("begin to train")
        n = X.shape[0]
        w = [np.ones(n)]*(self.M+1)
        alpha = [0]*(self.M+1)
        w[0] = w[0]/n
        model_list = []
        for m in range(0,self.M):
            model = self.model(max_depth=6,min_samples_split=10,min_samples_leaf=10,random_state=1010)
            print("begin to learn the %d  base classifier"%(m+1))
            model.fit(X,y,sample_weight=w[m])
            h = model.predict(X)
            hm = h==y.A1
            indicator = np.array([1 if not x else 0 for x in hm])
            em = (w[m]*indicator).sum()
            if em>0.5:
                break
            if em==0:
                break
            alpha[m] = 0.5*np.log((1-em)/em)
            zm = (w[m]*np.exp(-alpha[m]*y.A1*h)).sum()
            w[m+1] = w[m]/zm*np.exp(-alpha[m]*y.A1*h)
            print(" base classifier accuracy:%f"%((h==y.A1).sum()/len(h)))
            print("")
            model_list.append(model)
            
        self.alpha = alpha
        self.model_list = model_list

    def fitAndPredict(self,X,y,val_x,val_y):
        # 这个是fit的升级版，能够输出每一轮基学习器学习完之后的准确率
        print("")
        print("begin to train")
        # 样本数量
        n = X.shape[0]
        # 参数，初始化第一个参数为1/n
        w = [np.ones(n)]*(self.M+1)
        w[0] = w[0]/n
        # 参数，初始化为0
        alpha = [0]*(self.M+1)
        # 保存每一轮的学习器
        model_list = []
        # 保存每一轮的测试集正确率
        acc_history = []
        for m in range(0,self.M):
            # 初始化一个决策树，参数随便选的，默认参数会达到100%的训练正确率
            model = self.model(max_depth=6,min_samples_split=10,min_samples_leaf=10,random_state=1010)
            print("begin to learn the %d  base classifier"%(m+1))
            # 训练模型
            model.fit(X,y,sample_weight=w[m])
            h = model.predict(X)
            hm = h==y.A1
            indicator = np.array([1 if not x else 0 for x in hm])
            # indicator是课件中的指示函数
            # em是课件中的Epsilon m
            em = (w[m]*indicator).sum()
            # 错误率大于0.5，随机结果都比这个好，抛弃训练器并停止训练
            if em>0.5:
                break
            # 错误率0，停止训练
            if em==0:
                break
            # 课件上面的更新公式
            alpha[m] = 0.5*np.log((1-em)/em)
            zm = (w[m]*np.exp(-alpha[m]*y.A1*h)).sum()
            w[m+1] = w[m]/zm*np.exp(-alpha[m]*y.A1*h)
            print("base classifier accuracy:%f"%((h==y.A1).sum()/len(h)))
            # 保存模型
            model_list.append(model)            
            self.alpha = alpha
            self.model_list = model_list
            # 对测试集进行预测并保存正确率
            pred=self.predict(val_x)
            print("test set accuracy:")
            acc = (pred==val_y.A1).sum()/len(pred)
            print(acc)
            acc_history.append(acc)
            print("")

        return acc_history


    def predict_scores(self, X):
        # 这部分遍历每一个基学习器，分别得出一个预测结果，与对应的基学习器权重相乘并求和得到最后的预测结果
        h = np.zeros(X.shape[0])
        for i in range(len(self.model_list)):
            hi = self.model_list[i].predict(X)
            h = h + self.alpha[i]*hi
        return h

    def predict(self, X, threshold=0):
        # 这部分先调用了predict_socre函数，返回的结果中大于阈值的标为正例
        h = self.predict_scores(X)
        h = [1 if x>threshold else -1 for x in h]
        return h
        

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
