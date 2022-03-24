import cv2
import numpy as np
from matplotlib import pyplot as plt
import add2pictures2one as ad

pic1 = cv2.imread("./Images/canada.jpg",
                  cv2.IMREAD_UNCHANGED)  # color
# pic1 = pic1[:, :, ::-1] ## only need when pyplot is used

pic2 = cv2.imread("./Images/colorful_umbrella.jpg",
                  cv2.IMREAD_UNCHANGED)  # color

cv2.imshow('pic1, pic2 and combined', ad.shiftAndAdd(pic1, pic2, 0.5))
cv2.waitKey(0)
cv2.destroyAllWindows()
