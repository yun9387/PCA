import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
import glob
from sklearn.svm import OneClassSVM
import random
import datetime

IMG_SIZE = (126, 126) 

def pca_inv(train_files, test_files, n_comp):
    files = train_files + test_files
    pca = PCA(n_components=n_comp)
    data = np.zeros((len(files), IMG_SIZE[0]**2))
    
    for idx, path in enumerate(files):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, IMG_SIZE) 
        data[idx, :] = img.ravel() / 255.
    
    result = pca.fit_transform(StandardScaler().fit_transform(data))
    
    result_inv = pca.inverse_transform(result)
    
    train_result = result_inv[:len(train_files)]
    test_result = result_inv[len(train_files):]
    return train_result, test_result

def Plot(train_files, test_files, component=9):
    idx = random.randint(0, len(test_files))
    idx=5
    s = int(np.ceil(np.sqrt(component))) 
    fig = plt.figure()
    for i in range(component):
        n_comp = i+1
        _, test_data= pca_inv(train_files, test_files, n_comp)
        img = test_data[idx].reshape(IMG_SIZE[0], IMG_SIZE[1]) * 255
        ax = fig.add_subplot(s, s, n_comp)
        ax.imshow(img)
        title = "N_COMPONENTS = " + str(n_comp)
        ax.set_title(title)
    dt = datetime.datetime.now()
    time = dt.strftime("%Y-%m-%d %H %M %S")
    t1 = ".\\inverse_result\\result " + time + ".png"
    plt.savefig(t1)
    plt.show()
        
def main():
    train_files = glob.glob(".\\train_img\\*.jpeg")
    test_files = glob.glob(".\\test_img\\*.jpeg")
    _, test_data= pca_inv(train_files, test_files,6)

    Plot(train_files, test_files, 6)
    
   
    
if __name__ == "__main__":
    main()
    
