
# example of loading the keras facenet model
from keras.models import load_model
# load the model
def _load_model():
    model = load_model('facenet_keras.h5')
    return model
