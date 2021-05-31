from face_recognition import face_locations
import matplotlib.pyplot as plt
import cv2
import numpy as np

def _extract_bbox(image):
    """
    Trích xuất ra tọa độ của face từ ảnh input
    input:
    image: ảnh input theo kênh RGB. 
    single: Lấy ra 1 face trên 1 bức ảnh nếu True hoặc nhiều faces nếu False. Mặc định True.
    return:
    bbox: Tọa độ của bbox: <start_Y>, <start_X>, <end_Y>, <end_X>
    """
    bboxs = face_locations(image)

    if len(bboxs)==0:
        return None
    else:
        #trả về một list các face
        return bboxs

def draw_labels_and_boxes(image, all_box):
    print(all_box)
    for bbox in all_box:
        x_max = round(bbox[1])
        y_max = round(bbox[2])
        x_min = round(bbox[3])
        y_min = round(bbox[0])

        image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 255), thickness=2)

    return image

def face_detec(img):
    bboxs = _extract_bbox(img)
    return draw_labels_and_boxes(img,bboxs)





