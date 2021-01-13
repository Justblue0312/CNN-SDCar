"""
-This module gets an image through the webcam using the opencv package
-Display can be turned on or off
-Image size can be defined
"""

# import cv2

# cap = cv2.VideoCapture(0)
# size = [480, 240]

# def getImg(display= False):
#     _, img = cap.read()
#     img = cv2.resize(img,(size[0],size[1]))
#     if display:
#         cv2.imshow('IMG',img)
#     return img

# if __name__ == '__main__':
#     while True:
#         img = getImg(True)

import cv2


def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    while True:
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()


def main():
    show_webcam(mirror=True)


if __name__ == '__main__':
    main()