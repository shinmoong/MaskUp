import cv2
import numpy as np
import os

facenet = cv2.dnn.readNet('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')
with_dir = os.path.join("raw_train_data", "with_mask")
without_dir = os.path.join("withmask_1108")
print('total training with_mask images:', len(os.listdir(with_dir)))
print('total training badly images:', len(os.listdir(without_dir)))
withimgnum = len(os.listdir(with_dir))
withoutimgnum = len(os.listdir(without_dir))
with_files = os.listdir(with_dir)
without_files = os.listdir(without_dir)
print(withimgnum)
print(withoutimgnum)

for k in range(2624, 3000): # Change range (the number of images)
    count=k
    img = cv2.imread(os.path.join(without_dir, without_files[k]))
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, scalefactor=1., size=(305,305), mean=(104., 177., 123.)) # 얼굴이라고 생각하는 것들을 찾음
    facenet.setInput(blob)
    dets = facenet.forward()

    for i in range(dets.shape[2]):
        confidence = dets[0, 0, i, 2]
        if confidence < 0.5:
            continue
        x1 = int(dets[0, 0, i, 3] * w)
        y1 = int(dets[0, 0, i, 4] * h)
        x2 = int(dets[0, 0, i, 5] * w)
        y2 = int(dets[0, 0, i, 6] * h)

        if (x2 >= w or y2 >= h):
            continue

        face = img[y1:y2, x1:x2]
    face = cv2.resize(face, (200, 200))
# 여기 밑에꺼 수정!
    file_name_path = os.path.join("train_data_200", "with_mask", str(count) + '.jpg')
    # file_name_path = os.path.join("augument_face", str(count) + '.jpg')
    cv2.imwrite(file_name_path, face)
    print(count)

print("CopyComplete")