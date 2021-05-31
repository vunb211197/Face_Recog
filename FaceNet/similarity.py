from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

#hàm để so sánh sử dụng phương pháp learning similarity để tìm kiếm ảnh đồng nhất 
def _most_similarity(embed_vecs, vec, labels):
    image = ['Baejun','Chipu','Khanh','Son Tung','Thao Tam']
    s2= np.array(vec).reshape(128,).reshape(1,-1)
    sub=[]
    #sử dụng hàm cosine của sklearn
    for i in embed_vecs : 
        s1=np.array(i).reshape(128,).reshape(1,-1)
        a= cosine_similarity(s1,s2)
        sub.append(a)
    b=np.argmin(sub)
    print(b)
    c=labels[b]
    d=np.where(c==1)[0][0]
    print(image[d])