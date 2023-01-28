#pip install umap-learn scikit-learn
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
import glob
from sklearn.svm import OneClassSVM
import umap

IMG_SIZE = (126, 126)
RANDOM_STATE = 0
train_path = ".\\forUMAP\\train_img\\*"
test_path = ".\\forUMAP\\test_img\\*.jpeg"


def load_data(train_path, test_path):
    
    label = []
    train_data = np.array([[0 for x in range(IMG_SIZE[0]**2)]])
    for idx, path in enumerate(glob.glob(train_path)):
        path += "\\*.jpeg"
        f = glob.glob(path)
        
        l = [idx+1 for i in range(len(f))]
        label += l
        d = np.zeros((len(f), IMG_SIZE[0]**2))
        for i, p in enumerate(f):
            img = cv2.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, IMG_SIZE) 
            d[i, :] = img.ravel() / 255.
        train_data = np.concatenate([train_data, d], 0)
    train_data = train_data[1:, :]
    train_label = np.array(label)
    
    test_file = glob.glob(test_path)
    test_data = np.zeros((len(test_file), IMG_SIZE[0]**2))
    for idx, path in enumerate(test_file):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, IMG_SIZE) 
        test_data[idx, :] = img.ravel() / 255.
    test_label = [0 for _ in range(len(test_file))]
    test_label = np.array(test_label)
    
    return train_data, train_label, test_data, test_label

def Embedding(train_data, test_data):
    data = np.concatenate([train_data, test_data], 0)
    mapper = umap.UMAP(random_state=RANDOM_STATE)
    embedding = mapper.fit_transform(data)
    train_embed = embedding[:len(train_data)]
    test_embed = embedding[len(train_data):]
    return train_embed, test_embed

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
    train_data, train_label, test_data, test_label = load_data(train_path, test_path)
    train_embed, test_embed = Embedding(train_data, test_data)
    print(train_embed.shape)
    print(test_embed.shape)
    
    
if __name__ == "__main__":
    main()
    
