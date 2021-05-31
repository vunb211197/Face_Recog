import cv2
import os
path = ""
EMBEDDING_FL = os.path.join(path, "nn4.small2.v1.t7")

def _load_torch(model_path_fl):
  """
  model_path_fl: Link file chứa weigth của model
  """
  model = cv2.dnn.readNetFromTorch(model_path_fl)
  return model

#load được model
model = _load_torch(EMBEDDING_FL)