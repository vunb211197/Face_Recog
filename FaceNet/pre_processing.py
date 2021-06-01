from face_recognition import face_locations
import matplotlib.pyplot as plt
import cv2

def _extract_bbox(image, single = True):
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
    if single:
        bbox = bboxs[0]
        return bbox
    else:
        return bboxs


def _extract_face(image, bbox, face_scale_thres = (30, 30)):
    """
    input:
        image: ma trận RGB ảnh đầu vào
        bbox: tọa độ của ảnh input
        face_scale_thres: ngưỡng kích thước (h, w) của face. Nếu nhỏ hơn ngưỡng này thì loại bỏ face
    return:
        face: ma trận RGB ảnh khuôn mặt được trích xuất từ image input.
    """
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
        return None
    else:
        return face

def _blobImage(image, out_size = (96, 96), scaleFactor = 1.0, mean = (104.0, 177.0, 123.0)):
    """
    input:
        image: ma trận RGB của ảnh input
        out_size: kích thước ảnh blob
    return:
        imageBlob: ảnh blob
    """
    # Chuyển sang blobImage để tránh ảnh bị nhiễu sáng
    imageBlob = cv2.dnn.blobFromImage(image, scalefactor=scaleFactor,size=out_size, mean=mean,swapRB=False,crop=False)
    return imageBlob

def pre_processing(img):
    #gọi lần lượt các hàm tiền xử lý
    bbox = _extract_bbox(img, single = True)
    face = _extract_face(img,bbox,face_scale_thres=(30,30))
    img = _blobImage(face, out_size = (96, 96), scaleFactor=1/255.0, mean=(0, 0, 0))
    
    return img



