from make_data import get_data,split_train_val_test_data
from embedding import _embedding_faces
from model import EMBEDDING_FL,_load_torch
from pre_processing import pre_processing
from similarity import _most_similarity
import numpy as np

#hàm để đinh hình các bước dự đoán 1 ảnh đầu vào xem là ai
def _predict(img):
    # B1: load labels và pixel của data có sẵn 
    pixels,labels = get_data()
    # lấy được các thông số X_train,Y_train,X_test,Y_test
    (X_train, y_train, X_test, y_test) = split_train_val_test_data(pixels,labels)
    print("Y train la",y_train)
    # B2 : Lấy được model pretrain 
    model = _load_torch(EMBEDDING_FL)
    # B3 : Embedding các labels đó bằng cách nhét nó vào model và forward
    emb_vecs = _embedding_faces(model,X_train)
    #B4: preprocessing ảnh để đưa vào model embedding
    face_blob = pre_processing(img)
    #B5 : Đưa vào model embedding
    emb_vec = _embedding_faces(model,np.reshape(face_blob, (1,1,3,96,96)))
    #B6 : So sáng các embeeding với  nhau :
    label = _most_similarity(emb_vecs,emb_vec[0],y_train)
    #Trả về label có so sánh cosine gần nhất 
    return label


