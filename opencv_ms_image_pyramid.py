import cv2
import numpy as np
import opencv_ms_helper
# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html#goals

mshelp = opencv_ms_helper.opencv_ms_helper(cv2, np)
path1 = "./Images/lena.jpg"
pic1 = cv2.imread(path1, cv2.IMREAD_UNCHANGED)

# Make Gray
# pic1 = cv2.cvtColor(pic1, cv2.COLOR_RGB2GRAY)

if pic1 is not None:
    print("succesfully read picture")
else:
    print("DID NOT READ PICTURE")

layer = pic1.copy()
gp = [layer]
final = pic1.copy()

for i in range(6):
    layer = cv2.pyrDown(layer)
    gp.append(layer)
    final = mshelp.shiftAndAddHorizontal(final, layer)

final = mshelp.scaleImage(final, 0.5)
cv2.imshow("combined", final)

cv2.waitKey(0)
cv2.destroyAllWindows()
