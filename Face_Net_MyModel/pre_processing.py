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


def _extract_face(image, bboxs, face_scale_thres = (30, 30)):
    """
    input:
        image: ma trận RGB ảnh đầu vào
        bbox: tọa độ của ảnh input
        face_scale_thres: ngưỡng kích thước (h, w) của face. Nếu nhỏ hơn ngưỡng này thì loại bỏ face
    return:
        face: ma trận RGB ảnh khuôn mặt được trích xuất từ image input.
    """
    faces = []
    for bbox in bboxs :  
        face = None
        h, w = image.shape[:2]
        try:
            (startY, startX, endY, endX) = bbox
        except:
            return None

        minX, maxX = min(startX, endX), max(startX, endX)
        minY, maxY = min(startY, endY), max(startY, endY)
        face = image[minY:maxY, minX:maxX].copy()
        # extract the face ROI and grab the ROI dimensions
        (fH, fW) = face.shape[:2]

        # ensure the face width and height are sufficiently large
        if fW < face_scale_thres[0] or fH < face_scale_thres[1]:
            face = None
        else:
            faces.append(face)
        
    return faces


def pre_processing(img):
    all_face = []
    #gọi lần lượt các hàm tiền xử lý
    bboxs = _extract_bbox(img)

    faces = _extract_face(img,bboxs,face_scale_thres=(30,30))

    for face in faces : 
        face = cv2.resize(face,(160,160))
        all_face.append(face)
    return all_face



