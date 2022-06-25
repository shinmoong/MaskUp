import os, re, glob
import cv2
import numpy as np
import shutil
from keras.models import load_model


def Dataization(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (32, 32))

#    img = cv2.resize(img, (100, 100))
#    dst = img[50:82, 34:66].copy()
#    img = cv2.resize(img, (50, 50))
#    dst = img[18:50, 9:41].copy()
    return (img / 256)
#    return (dst / 256)

categories = ['with_mask', 'badly', 'no_mask']
num_classes = len(categories)
src = []
name = []
test = []
Y = []
group_dir = 'test_data_200'
acc = [0.0, 0.0, 0.0]

for idex, categorie in enumerate(categories):
    label = [0 for i in range(num_classes)]
    label[idex] = 1
    image_dir = os.path.join(group_dir, categorie)
    for top, dir, f in os.walk(image_dir):
        for filename in f:
            if filename.find('.jpg') is not -1:
                src.append(os.path.join(image_dir, filename))
                name.append(filename)
                test.append(Dataization(os.path.join(image_dir, filename)))
                Y.append(label)
#
test = np.array(test)
print(test.shape)
#여기 밑에 학습 모델 이름 바꾸기!
model = load_model('MaskModel_resize_32.h5')
predict = model.predict(test)
print(predict.shape)
print("ImageName : , Predict : [with_mask, badly, no_mask]")

for i in range(len(test)):
#    print(name[i] + " : , Predict : " + str(predict[i]))
    if predict[i][0] > predict[i][1] and predict[i][0] > predict[i][2]:
        if Y[i][0] == 1:
            acc[0] += 1.0
    elif predict[i][1] > predict[i][0] and predict[i][1] > predict[i][2]:
        if Y[i][1] == 1:
            acc[1] += 1.0
    else:
        if Y[i][2] == 1:
            acc[2] += 1.0

print("acc = with_mask: " + str(acc[0]/625.0*100) + "%, badly: " + str(acc[1]/625.0*100) + "%, no_mask: " + str(acc[2]/625.0*100) + "%")
print("acc:" + str((acc[0]+acc[1]+acc[2])/1875.0*100))