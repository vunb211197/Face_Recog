from keras.utils import np_utils
import numpy as np
import cv2
import os
from pre_processing import pre_processing

data_path= 'data'
root_path='data_face'

def get_data():
    #khởi tạo các list để chứa labels và data
    # lặp các thư mục trong data_path
    for dir in os.listdir(data_path):
        class_dir = os.path.join(data_path, dir)
        for image in os.listdir(class_dir):
            try :
                #lấy được ảnh từ đường dẫn
                pixel = cv2.imread(os.path.join(class_dir,image))
                #tiền xử lý ảnh 
                face = pre_processing(pixel)
                face_path= os.path.join(root_path,dir)
                if not os.path.exists(face_path): 
                    os.makedirs(face_path) 
                cv2.imwrite(os.path.join(face_path,image), face) 
                
            except :
                continue

def get_labes():
    Labels=[]
    for dir in os.listdir(data_path):
        Labels.append(dir)
    return Labels
