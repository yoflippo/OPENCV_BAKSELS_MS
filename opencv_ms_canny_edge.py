import cv2
import opencv_ms_helper
import numpy as np

h = opencv_ms_helper.opencv_ms_helper(cv2, np)

pic1 = cv2.imread("./Images/lena_baw.jpg", 0)
if pic1 is not None:
    print("succesfully read picture")
else:
    print("DID NOT READ PICTURE")

edges = cv2.Canny(pic1, 100, 100)

cv2.imshow("Canny Edge Detection", h.shiftAndAdd(pic1, edges, factor=0.5))
cv2.waitKey(0)
cv2.destroyAllWindows()
