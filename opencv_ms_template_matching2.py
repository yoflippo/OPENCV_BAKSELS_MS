import cv2
import numpy as np
import opencv_ms_helper
# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html#goals

mshelp = opencv_ms_helper.opencv_ms_helper(cv2, np)
path1 = "./Images/mario2.png"
pic1 = cv2.imread(path1, cv2.IMREAD_UNCHANGED)
path2 = "./Images/mario2_coin.png"
template = cv2.imread(path2, cv2.IMREAD_UNCHANGED)

if pic1 is not None:
    print("succesfully read picture")
else:
    print("DID NOT READ PICTURE")

pic1_gray = cv2.cvtColor(pic1, cv2.COLOR_RGB2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)

# Apply template Matching
res = cv2.matchTemplate(pic1_gray, template_gray, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where(res >= threshold)
h, w = template_gray.shape[:2]
for pt in zip(*loc[::-1]):
    cv2.rectangle(pic1, pt, (pt[0]+w, pt[1]+h), (0, 0, 255), 1)

mshelp.shiftAndAddHorizontal(pic1, res)
cv2.imshow("mario original with detected features", pic1)
cv2.imshow("mario result", res)
cv2.waitKey(0)
cv2.destroyAllWindows()
