from sklearn.svm import SVC
from make_data import split_train_val_test_data
from face_embedding import _embedding_faces
from pretrain_facenet import _load_model
from sklearn.metrics import accuracy_score
from pre_processing import pre_processing,_extract_bbox
import pickle
from create_data_face import get_labes
import cv2
import numpy as np


def _make_model(update):
    filename = 'my_model.sav'

    #trường hợp cần train lại do có data mới , dữ liệu được làm lại, cần train lại model
    if update :
        X_train, y_train, X_test, y_test = split_train_val_test_data()
        facenet_model = _load_model()
        X_train = _embedding_faces(facenet_model,X_train)
        model = SVC(kernel='linear',probability=True)
        model.fit(X_train, y_train)
        # lưu model vào đĩa
        pickle.dump(model, open(filename, 'wb'))
        return model
    #nếu không cần train lại thì lấy được model
    else : 
        #load model
        model = pickle.load(open(filename, 'rb'))
        return model

def predict(img,update):
    all_text = []
    all_box = []
    img1 = img
    #lấy được model 
    my_model = _make_model(update)
    
    #tiền xử lý dữ liệu
    allfaces = pre_processing(img)

    for face in allfaces : 
        #đưa vào embedding
        face = _embedding_faces(_load_model(),face.reshape(1,160,160,3))

        #lây được ra số dự đoán và phân phối
        y_class = my_model.predict(face)

        y_prob = my_model.predict_proba(face)

        class_index = y_class[0]
        class_probability = y_prob[0,class_index] * 100

        
        Labels = get_labes()

        class_name= Labels[class_index]
        class_probability=str('%.2f' % class_probability)
        #lấy được text
        text = class_name + "-" +class_probability+"%"
        #lấy được bbox
        bbox = _extract_bbox(img1)
        #thêm vào lists
        all_text.append(text)


    return draw_labels_and_boxes(img1,all_text,bbox)

def draw_labels_and_boxes(image, all_text, all_box):
    for i,bbox in enumerate(all_box):
        x_max = round(bbox[1])
        y_max = round(bbox[2])
        x_min = round(bbox[3])
        y_min = round(bbox[0])

        image = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 255), thickness=2)
        image = cv2.putText(image, all_text[i], (x_min - 20, y_min-10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 200), thickness=2)

    return image
