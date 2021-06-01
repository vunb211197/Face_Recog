from keras.utils import np_utils
import numpy as np
import cv2
import os
from pre_processing import pre_processing

data_path= 'C:\\Users\\nguyen ba vu\\Desktop\\Face_Recog\\FaceNet\\data'

def get_data():
    #khởi tạo các list để chứa labels và data
    pixels = []
    labels = []
    # lặp các thư mục trong data_path
    for dir in os.listdir(data_path):
        class_dir = os.path.join(data_path, dir)
        for image in os.listdir(class_dir):
            try :
                #lấy được ảnh từ đường dẫn
                pixel = cv2.imread(os.path.join(class_dir,image))
                #tiền xử lý ảnh trước khi đưa vào model
                face_blob = pre_processing(pixel)
                #thêm vào file train
                pixels.append(face_blob)
                #các case của nhãn
                if dir == "baejun":
                    labels.append(0)
                elif dir == "chipu":
                    labels.append(1)
                elif dir == "khanh":
                    labels.append(2)
                elif dir == "sontung":
                    labels.append(3)
                elif dir == "thaotam":
                    labels.append(4)
            except :
                continue
    #trả về pixels và labels
    return pixels,labels
def split_train_val_test_data(pixels,labels):
    # Chuẩn hoá dữ liệu pixels và labels

    #đưa dữ liệu về mảng np array
    pixels = np.array(pixels)

    #đưa nhãn về dạng one-hot vector
    labels = np_utils.to_categorical(labels)

    # Nhào trộn dữ liệu ngẫu nhiên để cho công bằng
    randomize = np.arange(len(pixels))
    np.random.shuffle(randomize)

    X = pixels[randomize]
    y = labels[randomize]

    # Chia dữ liệu theo tỷ lệ 80% train và 20% còn lại cho test
    train_size = int(X.shape[0] * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]



    #trả về các tập thu được 
    return X_train, y_train, X_test, y_test
