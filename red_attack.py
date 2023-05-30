


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from PIL import Image
import os

# Reading the input images and putting them into a numpy array
data=[]
labels=[]

height = 30
width = 30
channels = 3
classes = 43
n_inputs = height * width * channels

for i in range(classes):
    path = "dataset/train/{0}/".format(i)
    print(path)
    Class = os.listdir(path)
    for a in Class:
        try:
            image = cv2.imread(path+a)
            image_from_array = Image.fromarray(image, 'RGB')
            size_image = image_from_array.resize((height, width))
            data.append(np.array(size_image))
            labels.append(i)
        except AttributeError:
            print(" ")
            
Cells = np.array(data)
labels = np.array(labels)

# Randomize the order of the input images
s = np.arange(Cells.shape[0])
np.random.seed(43)
np.random.shuffle(s)
Cells = Cells[s]
labels = labels[s]

from tensorflow.keras.models import load_model
model = load_model("gtsrb_model_final.h5")
print(path)

# Predicting with the test data
y_test = pd.read_csv("dataset/Test.csv")
labels = y_test['Path'].to_numpy()
y_test = y_test['ClassId'].values

data = []

for f in labels:
    image = cv2.imread("dataset/test/" + f.replace('Test/', ''))
    image_from_array = Image.fromarray(image, 'RGB')
    size_image = image_from_array.resize((height, width))
    data.append(np.array(size_image))

X_test = np.array(data)
X_test = X_test.astype('float32') / 255 
pred = model.predict_classes(X_test)

predict_x = model.predict(X_test) 
pred = np.argmax(predict_x, axis=1)
print(pred)

# Accuracy with the test data
from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred)

def max_diff(img1,img2):
    img = img1 - img2
    return np.amax(img)

def pred(image):
    data = []
    data.append(image)
    X_test = np.array(data)
    X_test = X_test.astype('float32')/255 
    X_test = X_test.reshape(1,30,30,3)
    #print(X_test.shape)
    predict_x=model.predict(X_test) 
    pred_target_image=np.argmax(predict_x,axis=1)
    #pred_target_image=pred_target_image[0]
    
    #pred = model.predict_classes(X_test)
    return pred_target_image[0]

def boundary_estimation(source, target, dmin):
    Ii = ((source + target)/2.0)
    k = pred(Ii)
    delta = max_diff(source, Ii)
    Ia2 = source
    Ib2 = target
    p = Ib2
    while (delta > dmin):
        if (pred(Ia2) != k):
            Ib2 = Ii
        else:
            Ia2 = Ii
        Ii = ((Ia2+Ib2)/2.0)
        k = pred(Ii)
        delta = max_diff(Ia2,Ii)  
    return Ii

def go_out(source,iout,alpha):
    i_diff = iout - source
    pred_source = pred(source)
    inew = iout
    while (pred(inew)==pred_source):
        inew = inew + alpha*(i_diff)
        
    return inew

source_image_path = "C:/Users/legra/Downloads/dataset/test/00001.png";
target_image_path = "C:/Users/legra/Downloads/dataset/test/00009.png";
print("SOURCE IMAGE:\n")
print(source_image)

img = (np.asarray(Image.open(source_image_path)))
image_from_array = Image.fromarray(img, 'RGB')
img1 = image_from_array.resize((height, width))
img1=np.array(img1)
img1=img1.reshape(30,30,3)
source_image = np.array(img1)

img = (np.asarray(Image.open(target_image_path)))
image_from_array = Image.fromarray(img, 'RGB')
img2 = image_from_array.resize((height, width))
img2=np.array(img2)
img2=img2.reshape(30,30,3)
target_image = np.array(img2)

i = boundary_estimation(source_image,target_image,1.0)

print (pred(i)) 
print (pred(source_image)) #Class of source image
print (pred(target_image)) #Class of target image

ii = go_out(source_image,i,0.01)

pred(ii)

Image.fromarray(i.astype('uint8')).show()

Image.fromarray(ii.astype('uint8')).show()

def array_diff(d1):
    sumd1 = 0.0
    for i in range(0,3):
        for j in range(0,30):
            for k in range(0,30):
                d1[j][k][i] = d1[j][k][i]*d1[j][k][i]
                sumd1 = sumd1 + d1[j][k][i]
    return (sumd1)

def gradient_estimation(source, target, adversarial, n, theta):
    Ia = source
    Ib = target
    Ii = adversarial
    Io = np.zeros((2700))
    X = np.random.randint(0,2700, size=n)
    for i in X:
        Io[i] = 255
    Io = Io.reshape((30,30,3))
#     print(Io*theta)
    Ii2 = Ii + theta*Io
    Ii2_new = boundary_estimation(Ia, Ii2, 1.0)
    Ii2_new = go_out(source,Ii2_new,0.01)
    diff2 = Ii2_new - Ia
    diff1 = Ii - Ia
    d2 = array_diff(diff2)
    d1 = array_diff(diff1)
    if (d2 > d1):
        return (-1, Ii2_new)
    elif (d1 > d2):
        return (1, Ii2_new)
    else:
        return (0,Ii2_new)

def efficient_update(source, target, adversarial, I2, g, j):
    Ia = source
    Ib = target
    Ii = adversarial
    Ii2 = I2
    delta = g*(Ii2 - Ii)
    l = j
    Inew = Ii + l*delta
    
    diff1 = Inew - Ia
    diff2 = Ii - Ia
    d1 = array_diff(diff1)
    d2 = array_diff(diff2)
    ii = 0
    it = 0
    while(d1 > d2):
        l = (l/2.0)
        Inew = Ii + l*delta
        if(pred(Inew)==pred(source)):
            Inew = go_out(source,Inew,0.01)
        it = it + 1
        d1 = array_diff(Inew-Ia)
        if(it>100):
            break
    if (d1 > d2):
        print(ii)
        ii = ii + 1
        Inew = Ii
    return Inew

def iteration(itr, source, target, n, theta, j, dmin):
    targett = target
    sourcee = source
    for i in range(itr):
        print ("\n Iteration: ",i)
        adversarial_image = boundary_estimation(sourcee, targett, dmin)
        adversarial_image = go_out(sourcee,adversarial_image,0.01)
        (g, Iii2) = gradient_estimation(sourcee, targett, targett, n, theta)
        targett = efficient_update(sourcee, targett, adversarial_image, Iii2, g, j)
        if (pred(targett) == pred(source)):
            j = j/2.0
        fin = targett
        if(pred(targett)==pred(sourcee)):
            fin = go_out(sourcee,targett,0.01)
        if(array_diff(fin-sourcee)<array_diff(adversarial_image-sourcee)):
            targett = fin
            #print("uopp")
    
    return fin

final = iteration(1000,source_image,target_image,5,0.196,5.0,1.0)

pred(final)

Image.fromarray(source_image.astype('uint8')).save('original_image.png')
Image.fromarray(final.astype('uint8')).save('perturbed_image.png')
# s = measure.compare_ssim(arr[1],arr[0])

original = cv2.imread("original_image.png")
perturb = cv2.imread("perturbed_image.png")

#s = measure.compare_ssim(original,perturb,multichannel=True)
#print(s)
print(perturb)
print(original)







