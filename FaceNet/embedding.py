


def _embedding_faces(encoder, faces):
    #khởi tạo list chứa embedding đầu tiên là rỗng
    emb_vecs = []
    for i in range(faces.shape[0]):
        # set input cho model
        encoder.setInput(faces[i])
        #forward để tính được đầu ra model là vector
        vec = encoder.forward()
        #thêm vào list vector embedding
        emb_vecs.append(vec)
    #trả về list vector embedding
    return emb_vecs