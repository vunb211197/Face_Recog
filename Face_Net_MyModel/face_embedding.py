import numpy as np
from pretrain_facenet import _load_model


def _embedding_faces(model, all_face_pixels):
    embedding_train=[]
    for face_pixels in all_face_pixels : 
        # scale pixel values
        face_pixels = face_pixels.astype('float32')
        # standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # transform face into one sample
        samples = np.expand_dims(face_pixels, axis=0)
        # make prediction to get embedding
        yhat = model.predict(samples)
        #add to list
        embedding_train.append(yhat[0])
    return np.asarray(embedding_train)