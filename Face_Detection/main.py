import argparse
import cv2
from svc import predict

def getArgument():
    arg = argparse.ArgumentParser()
    # định nghĩa một tham số cần parse
    arg.add_argument('-i', '--image_path',help='link to image')
    arg.add_argument('-u', '--update',help='True or false',default=False)
    # Giúp chúng ta convert các tham số nhận được thành một object và gán nó thành một thuộc tính của một namespace.
    return arg.parse_args()


arg = getArgument()

# đọc được các thuộc tính  từ đường dẫn

img = cv2.imread(arg.image_path)
update = arg.update


#predict hình ảnh
img = predict(img,update)

#show ảnh
cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


