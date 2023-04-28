import cv2
import numpy as np
# from matplotlib import pyplot as plt
import opencv_ms_helper

h = opencv_ms_helper.opencv_ms_helper(cv2, np)


def _getPictures():
    path1 = "./Images/lena.jpg"
    pic1 = cv2.imread(path1,
                      cv2.IMREAD_UNCHANGED)  # color
    path2 = "./Images/colorful_umbrella.jpg"
    pic2 = cv2.imread(path2,
                      cv2.IMREAD_UNCHANGED)  # color
    return pic1, pic2, path1, path2


def _showResult(pic, title=""):
    cv2.imshow(title, pic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ShiftAndAdd_Test():
    pic1, pic2, _, _ = _getPictures()
    # pic1 = pic1[:, :, ::-1] ## only need when pyplot is used
    _showResult(h.shiftAndAdd(pic1, pic2, 1), 'shift and add picture')


def ShiftAndAddHorizontal_Test():
    pic1, pic2, _, _ = _getPictures()
    # pic1 = pic1[:, :, ::-1] ## only need when pyplot is used
    _showResult(h.shiftAndAddHorizontal(
        pic1, pic2), 'shift and add picture')


def getDifferenceBetweenPictures_Test():
    pic1, pic2, _, _ = _getPictures()
    pic2 = cv2.GaussianBlur(pic1, (5, 5), 1)
    picdiff = h.getDifferenceBetweenPictures(pic1, pic2)
    _showResult(picdiff, 'difference between picture and its blurred version')


def makeBinaryPicture_Test():
    pic1, pic2, _, _ = _getPictures()
    _showResult(h.makeBinaryPicture(pic1), 'black and white')


def makeBinaryPictureAndSave_Test():
    pic1, _, path1, _ = _getPictures()
    _showResult(h.makeBinaryPictureAndSave(pic1, path1), 'black and white')


ShiftAndAddHorizontal_Test()
# ShiftAndAdd_Test()
# getDifferenceBetweenPictures_Test()
# makeBinaryPicture_Test()
# makeBinaryPictureAndSave_Test()
