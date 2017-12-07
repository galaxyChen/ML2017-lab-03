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
            print("begin to learn the %d base learner"%(m+1))
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
            print("base learner accuracy:%f"%((h==y.A1).sum()/len(h)))
            model_list.append(model)
            
        self.alpha = alpha
        self.model_list = model_list

        return X,y
            


    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        h = np.zeros(X.shape[0])
        for i in range(len(self.model_list)):
            hi = self.model_list[i].predict(X)
            h = h + self.alpha[i]*hi
        return h

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        h = self.predict_scores(X)
        h = [1 if x>0 else -1 for x in h]
        return h
        

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
