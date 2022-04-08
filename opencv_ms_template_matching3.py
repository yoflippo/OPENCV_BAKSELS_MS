import cv2
import numpy as np
import opencv_ms_helper
# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html#goals

mshelp = opencv_ms_helper.opencv_ms_helper(cv2, np)
path1 = "./Images/mario2.png"
pic1 = cv2.imread(path1, cv2.IMREAD_UNCHANGED)
path2 = "./Images/mario2_coin.png"
template = cv2.imread(path2, cv2.IMREAD_UNCHANGED)

pic1_gray = cv2.cvtColor(pic1, cv2.COLOR_RGB2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)

# cv2.imshow("test gray to color", cv2.cvtColor(pic1_gray, cv2.COLOR_GRAY2RGB))
# cv2.imshow("just gray", pic1_gray)
if pic1 is not None:
    print("succesfully read picture")
else:
    print("DID NOT READ PICTURE")

templatePyramids = mshelp.makePyramids(template_gray, 3, 1)
templatePyramidscolor = mshelp.makePyramids(template, 3, 1)
blFoundSomething = False
for i in range(len(templatePyramids)):
    template2 = templatePyramids[i]
    picout = pic1.copy()
    # Apply template Matching, BE AWARE that res = -[-1...1]
    res = cv2.matchTemplate(pic1_gray, template2, cv2.TM_CCOEFF_NORMED)
    result = ((res*255)+255)/2
    resultabs = abs(res)*255

    threshold = 0.9
    loc = np.where(res >= threshold)
    h, w = template2.shape[:2]

    blFound = False
    for pt in zip(*loc[::-1]):
        blFound = True
        cv2.rectangle(picout, pt, (pt[0]+w, pt[1]+h), (0, 0, 255), 1)
        blFoundSomething = blFound
    if blFound:
        cv2.imshow(str(i),
                   mshelp.shiftAndAddHorizontal3(picout, result, resultabs))

if not blFoundSomething:
    print("Did not find anything")
# cv2.imshow("check1", picout)
# cv2.imshow("check2", result)
# cv2.imshow("check3", template2)
# mshelp.shiftAndAddHorizontal(result, picout)

# # TODO HAVE A LOOK AT HOW TO PROPERLY ADD A GRAY TO COLOR IMAGE
cv2.waitKey(0)
cv2.destroyAllWindows()
