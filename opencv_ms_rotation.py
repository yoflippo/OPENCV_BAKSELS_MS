import cv2

pic = cv2.imread("./Images/resized_cropped_region_2x.png")
# print(pic)
cols = pic.shape[1]
rows = pic.shape[0]
# print(rows, cols)


def rotatems(pic, cols, rows, numberoftimes=1):
    center = (cols/2, rows/2)
    angle = 90
    rotate = pic
    M = cv2.getRotationMatrix2D(center, angle, 1)
    for x in range(numberoftimes):
        rotate = cv2.warpAffine(rotate, M, (cols, rows),
                                flags=cv2.INTER_LINEAR)
    print(rotate.shape)
    return rotate


rotate = rotatems(pic, cols, rows, 4)


cv2.imshow('normal', pic)
cv2.imshow('rotated', rotate)
cv2.waitKey(0)
cv2.destroyAllWindows()
