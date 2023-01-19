from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.pyplot as plt
import glob

file = glob.glob("C:\\anaconda3\\envs\\test\\img\\*.jpeg")
print(file)

def get_pca(f):
    data = []
    pca = PCA(n_components=2)
    for idx, file in enumerate(f):
        #画像データの取得
        img = Image.open(file).convert('L')
        
        data.append(np.array(img)/255.)
        pca.fit(np.array(img)/255.)
        print(pca.transform(np.array(img)/255.))
        
get_pca(file)
        
        
        
        

#print(img.shape)
#data.append(np.array(img).flatten()/255.) 