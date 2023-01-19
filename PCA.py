import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
import glob
from sklearn.svm import OneClassSVM

N_COMPONENTS = 2 #must be between 0 and datasize 
IMG_SIZE = (512, 512) 

def pca_result(files):
    pca = PCA(n_components=N_COMPONENTS)
    data = np.zeros((len(files), IMG_SIZE[0]**2))
    
    for idx, path in enumerate(files):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, IMG_SIZE) 
        data[idx, :] = img.ravel() / 255.
        
    result = pca.fit_transform(StandardScaler().fit_transform(data))
    return result

def SVM(train_feature, test_feature):
    svm = OneClassSVM(kernel='rbf', nu=0.2, gamma=1e-04)
    svm.fit(train_feature)  
    pred = svm.predict(test_feature)
    return pred

def Plot(train_feature, test_feature, pred):
    correct = test_feature[np.where(pred==1, True, False)]
    incorrect = test_feature[np.where(pred==-1, True, False)]
    
    fig = plt.figure()
    plt.scatter(train_feature[:,0], train_feature[:,1], c="grey")
    plt.scatter(correct[:,0], correct[:,1], c="blue")
    plt.scatter(incorrect[:,0], incorrect[:,1], c="red")    
    plt.show()
    
    
def main():
    train_files = glob.glob(".\\train_img\\*.jpeg")
    test_files = glob.glob(".\\test_img\\*.jpeg")
    
    train_feature = pca_result(train_files)
    test_feature = pca_result(test_files)
    
    pred = SVM(train_feature, test_feature)
    
    if N_COMPONENTS==2 :  Plot(train_feature, test_feature, pred)
    incorrect = test_feature[np.where(pred==-1, True, False)]
    acc = (len(incorrect)/len(test_feature)) * 100
    print("正解率:{}%".format(acc))
    
if __name__ == "__main__":
    main()
