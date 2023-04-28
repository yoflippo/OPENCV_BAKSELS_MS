import cv2
import numpy as np
import opencv_ms_helper
from matplotlib import pyplot as plt
# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html#goals

h = opencv_ms_helper.opencv_ms_helper(cv2, np)
path1 = "./Images/mario1.png"
pic1 = cv2.imread(path1, cv2.IMREAD_UNCHANGED)
path2 = "./Images/mario1_coin.png"
pic2 = cv2.imread(path2, cv2.IMREAD_UNCHANGED)

if pic1 is not None:
    print("succesfully read picture")
else:
    print("DID NOT READ PICTURE")

pic1_gray = cv2.cvtColor(pic1, cv2.COLOR_RGB2GRAY)
pic2_gray = cv2.cvtColor(pic2, cv2.COLOR_RGB2GRAY)

# All the 6 methods for comparison in a list
methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

for meth in methods:
    img = pic1.copy()
    h, w = pic2.shape[:2]
    method = eval(meth)

    # Apply template Matching
    res = cv2.matchTemplate(img, pic2, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img, top_left, bottom_right, 255, 2)

    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)

    plt.show()


# cv2.imshow("mario", pic1)
# cv2.imshow("mario coin", pic2)

# cv2.imshow("mario gray", pic1_gray)
# cv2.imshow("mario coin gray", pic2_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
