import os
import cv2
import numpy as np
from keras.utils import np_utils
from pretrain_facenet import _load_model
from face_embedding import _embedding_faces


def split_train_val_test_data():
    count = -1 
    data_path='data_face'
    #khởi tạo các list để chứa labels và data
    pixels = []
    labels = []
    # lặp các thư mục trong data_path
    for dir in os.listdir(data_path):
        count = count + 1
        class_dir = os.path.join(data_path, dir)
        for image in os.listdir(class_dir):
                #lấy được ảnh từ đường dẫn
                pixel = cv2.imread(os.path.join(class_dir,image))
                #thêm vào file train
                pixels.append(pixel)
                #các case của nhãn
                labels.append(count)
    pixels = np.asarray(pixels)
    labels = np.asarray(labels)
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



