from keras.models import Sequential
from keras.layers import Dropout, Activation, Dense
from keras.layers import Flatten, Convolution2D, MaxPooling2D, Conv2D, BatchNormalization
from keras.models import load_model
from keras import regularizers
import tensorflow
import cv2
import numpy as np
import os, re, glob
import matplotlib.pyplot as plt


def Dataization(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (32, 32))
    return (img / 256)


groups_folder_path = 'train_data_200'
categories = ['with_mask', 'badly', 'no_mask']

num_classes = len(categories)
n = 0

X = []
Y = []
src = []
name = []
test = []
test_Y = []
group_dir = 'validation_data_200'
acc = [0.0, 0.0, 0.0]

for idex, categorie in enumerate(categories):
    label = [0 for i in range(num_classes)]
    label[idex] = 1
    image_dir = os.path.join(groups_folder_path, categorie)
    test_image_dir = os.path.join(group_dir, categorie)
    for top, dir, f in os.walk(image_dir):
        for filename in f:
            img = cv2.imread(os.path.join(image_dir, filename))
            img = cv2.resize(img, (32, 32))

            #    img = cv2.resize(img, (100, 100))
            #    dst = img[50:82, 34:66].copy()

#            img = cv2.resize(img, (50, 50))
#            dst = img[18:50, 9:41].copy()

            X.append(img / 256)
#            X.append(dst / 256)
            Y.append(label)
    for top, dir, f in os.walk(test_image_dir):
        for filename in f:
            if filename.find('.jpg') is not -1:
                src.append(os.path.join(test_image_dir, filename))
                name.append(filename)
                test.append(Dataization(os.path.join(test_image_dir, filename)))
                test_Y.append(label)

Xtr = np.array(X)
Ytr = np.array(Y)
test = np.array(test)
test_Y = np.array(test_Y)
X_train, Y_train = Xtr,Ytr
print(X_train.shape)
print(Y_train.shape)
print(test.shape)
print(test_Y.shape)

weight_decay = 1e-4
model = Sequential()
model.add(Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32,32,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3,3), padding='same', activation='relu'))

model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

hist=model.fit(X_train, Y_train, batch_size=64, nb_epoch=40, verbose=1, validation_data=(test, test_Y))

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

score = model.evaluate(test, test_Y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


model.save('MaskModel_resize_32.h5')