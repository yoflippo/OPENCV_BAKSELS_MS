import cv2
import numpy as np
from matplotlib import pyplot as plt
import add2pictures2one as ad
import differenceBetweenPictures as dp

pic1 = cv2.imread("./Images/canada.jpg",
                  cv2.IMREAD_UNCHANGED)  # color

pic2 = cv2.GaussianBlur(pic1, (5, 5), 1)

picdiff = dp.getDifferenceBetweenPictures(pic1, pic2)


cv2.imshow('difference between picture and its blurred version', picdiff)
cv2.waitKey(0)
cv2.destroyAllWindows()
