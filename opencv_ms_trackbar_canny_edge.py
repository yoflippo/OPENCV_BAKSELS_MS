import cv2
import opencv_ms_helper
import numpy as np

h = opencv_ms_helper.opencv_ms_helper(cv2, np)
windowname = "trackbar Canny Edge"
tbName1 = "threshold1"
tbName2 = "threshold2"
tbName3 = "blurfactor"


def funcCan(thresh1=0):
    thresh1 = cv2.getTrackbarPos(tbName1, windowname)
    thresh2 = cv2.getTrackbarPos(tbName2, windowname)
    blurfactor = cv2.getTrackbarPos(tbName3, windowname)

    if blurfactor % 2 == 0:
        blurfactor += 1
    pic = cv2.GaussianBlur(img, (blurfactor, blurfactor), 0)
    edge = cv2.Canny(pic, thresh1, thresh2)

    print(thresh1, thresh2, blurfactor)
    cv2.imshow(windowname, 255-edge)


if __name__ == '__main__':
    original = cv2.imread("./Images/lena.jpg", 1)
    img = original.copy()
    img = h.scaleImage(img, 0.8)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    windowname = 'canny'
    cv2.namedWindow(windowname)
    cv2.resizeWindow(windowname, 500, 500)

    thresh1 = 10
    thresh2 = 1
    cv2.createTrackbar(tbName1, windowname, thresh1, 255, funcCan)
    cv2.createTrackbar(tbName2, windowname, thresh2, 255, funcCan)
    cv2.createTrackbar(tbName3, windowname, 1, 51, funcCan)
    funcCan(0)

    cv2.imshow('Frame', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()


# def funcCan(thresh1=0):
#     thresh1 = cv2.getTrackbarPos('thresh1', 'canny')
#     thresh2 = cv2.getTrackbarPos('thresh2', 'canny')
#     edge = cv2.Canny(img, thresh1, thresh2)
#     cv2.imshow('canny', edge)


# if __name__ == '__main__':

#     original = cv2.imread("./Images/ms.jpg", 0)
#     img = original.copy()
#     img = h.scaleImage(img, 0.5)
#     img = cv2.GaussianBlur(img, (5, 5), 0)
#     cv2.namedWindow('canny')

#     thresh1 = 10
#     thresh2 = 1
#     cv2.createTrackbar('thresh1', 'canny', thresh1, 255, funcCan)
#     cv2.createTrackbar('thresh2', 'canny', thresh2, 255, funcCan)
#     funcCan(0)

#     cv2.imshow('Frame', img)
#     cv2.waitKey(0)

# cv2.destroyAllWindows()
