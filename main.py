import os
import numpy as np
from PIL import Image, ImageFile
import dlib
import glob
import cv2

faces_folder_path = os.path.join("use_image")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_distance_from_point_to_line(point, line_point1, line_point2):
    distance = np.abs((line_point2[1] - line_point1[1]) * point[0] +
                      (line_point1[0] - line_point2[0]) * point[1] +
                      (line_point2[0] - line_point1[0]) * line_point1[1] +
                      (line_point1[1] - line_point2[1]) * line_point1[0]) / \
               np.sqrt((line_point2[1] - line_point1[1]) * (line_point2[1] - line_point1[1]) +
                       (line_point1[0] - line_point2[0]) * (line_point1[0] - line_point2[0]))
    return int(distance)


for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    if count < 7500:
        count += 1
        continue

    img = cv2.imread(f, 1)
    rows, cols = img.shape[:2]
    dets = detector(img, 1)
    for k, d in enumerate(dets):
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        # 3은 왼쪽 턱, 8은 아래 턱, 13은 오른쪽 턱, 29는 코 중앙, 34는 코끝, 52는 윗입술 중앙
        for i in range(68):
            landmark_x = shape.part(i).x
            landmark_y = shape.part(i).y

            if i == 3:
                left = np.array([landmark_x, landmark_y])
            elif i == 8:
                chin = np.array([landmark_x, landmark_y])
            elif i == 13:
                right = np.array([landmark_x, landmark_y])
            elif i == 27:
                nose = np.array([landmark_x, landmark_y])

    img2 = Image.open(f)
    mask_img = Image.open('white_mask.png')
    width = mask_img.width
    height = mask_img.height
    width_ratio = 1.9
    new_height = int(np.linalg.norm(left - right)*1.9)

    # left
    mask_left_img = mask_img.crop((0, 0, width // 2, height))
    mask_left_width = get_distance_from_point_to_line(left, nose, chin)
    mask_left_width = int(mask_left_width * width_ratio)
    mask_left_img = mask_left_img.resize((mask_left_width, new_height))

    # right
    mask_right_img = mask_img.crop((width // 2, 0, width, height))
    mask_right_width = get_distance_from_point_to_line(right, nose, chin)
    mask_right_width = int(mask_right_width * width_ratio)
    mask_right_img = mask_right_img.resize((mask_right_width, new_height))

    # merge mask
    size = (mask_left_img.width + mask_right_img.width, new_height)
    mask_img = Image.new('RGBA', size)
    mask_img.paste(mask_left_img, (0, 0), mask_left_img)
    mask_img.paste(mask_right_img, (mask_left_img.width, 0), mask_right_img)

    # rotate mask
    angle = np.arctan2(chin[1] - nose[1], chin[0] - nose[0])
    rotated_mask_img = mask_img.rotate(angle, expand=True)

    # calculate mask location
    center_x = (nose[0] + chin[0]) // 2
    center_y = (nose[1] + chin[1]) // 2

    offset = mask_img.width // 2 - mask_left_img.width
    radian = angle * np.pi / 180
    box_x = center_x + int(offset * np.cos(radian)) - rotated_mask_img.width // 2
    box_y = center_y + int(offset * np.sin(radian)) - rotated_mask_img.height // 2

    img2.paste(rotated_mask_img, (box_x, box_y), rotated_mask_img)

    img2.save(os.path.join(str(count) + ".jpg"))
    count += 1
    print(count)

    if count == 7504:
        break

#    if count == 6610:
#        break