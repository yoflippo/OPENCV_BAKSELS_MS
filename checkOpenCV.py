import cv2 as cv
print(cv.__version__)
img = cv.imread('./Images/resized_cropped_region_2x.png', 0)
# img = cv.imread('./Images/img_coins_0.jpg')
print(img)
winname = 'nameofWindow'
cv.namedWindow(winname, cv.WINDOW_NORMAL)
cv.resizeWindow(winname, (800, 600))
cv.imshow(winname, img)
cv.waitKey(0)
cv.destroyAllWindows()
