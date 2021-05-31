import argparse
import cv2
import time
import numpy as np
from predict import _predict

def getArgument():
    arg = argparse.ArgumentParser()
    # định nghĩa một tham số cần parse
    arg.add_argument('-i', '--image_path',
                     help='link to image')
    # Giúp chúng ta convert các tham số nhận được thành một object và gán nó thành một thuộc tính của một namespace.
    return arg.parse_args()


# start time
start = time.time()

arg = getArgument()

# B1: Lấy ra được ảnh từ đường dẫn
img = cv2.imread(arg.image_path)
# B2: Cho dự đoán
labels_predict = _predict(img)
# hiển thị ảnh cùng với labels dự đoán
cv2.imshow(labels_predict, img)
cv2.waitKey(0)
cv2.destroyAllWindows()

end = time.time()
# in ra thời gian thực hiện model
print('Model process on %.2f s' % (end - start))
