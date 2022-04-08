import os
from shutil import SpecialFileError


class opencv_ms_helper:
    # def _init_(self):
    def __init__(self, opencvref, numpyref, blUseOptimizedCV2=False):
        self.cv2 = opencvref
        self.np = numpyref
        # opencvref.setUseOptimized(blUseOptimizedCV2)

    def _cutPicturesToSimilarSize(self, pic1, pic2):
        sz1 = pic1.shape
        sz2 = pic2.shape
        minx = min(sz1[0], sz2[0])
        miny = min(sz1[1], sz2[1])
        return pic1[0:minx, 0:miny], pic2[0:minx, 0:miny]

    def _getScaledHeightAndWidth(self, pic, factor):
        h, w = pic.shape[:2]
        h *= factor
        h = int(h)
        w *= factor
        w = int(w)
        return h, w

    def _makeSameSizeByScaling(self, pic1, pic2, factor=0.5):
        h1, w1 = self._getScaledHeightAndWidth(pic1, factor)
        h2, w2 = self._getScaledHeightAndWidth(pic2, factor)

        def findSmallestFactor(h1, h2, w1, w2):
            factor1 = max(h1, h2)/min(h1, h2)
            factor2 = max(w1, w2)/min(w1, w2)
            if factor1 > 1:
                factor1 = 1/factor1
            if factor2 > 1:
                factor2 = 1/factor2
            return min(factor1, factor2)

        def orderPictureFromLargeToSmall(pic1, pic2):
            h1, w1 = pic1.shape[:2]
            h2, w2 = pic2.shape[:2]
            if h1*w1 > h2*w2:
                return pic1, pic2
            else:
                return pic2, pic1

        picbig, _ = orderPictureFromLargeToSmall(pic1, pic2)
        (h, w) = self._getScaledHeightAndWidth(
            picbig, findSmallestFactor(h1, h2, w1, w2))

        if self.np.all(self.np.equal(picbig.shape, pic1.shape)):
            pic1out = self.cv2.resize(
                pic1, (w, h), interpolation=self.cv2.INTER_CUBIC)
            pic2out = pic2
        else:
            pic2out = self.cv2.resize(
                pic2, (w, h), interpolation=self.cv2.INTER_CUBIC)
            pic1out = pic1

        return pic1out, pic2out

    def _scaleFigures(self, pic1, pic2, factor=1):
        h, w = self._getScaledHeightAndWidth(pic1, factor)
        pic1out = self.cv2.resize(
            pic1, (w, h), interpolation=self.cv2.INTER_CUBIC)

        h, w = self._getScaledHeightAndWidth(pic2, factor)
        pic2out = self.cv2.resize(
            pic2, (w, h), interpolation=self.cv2.INTER_CUBIC)

        return pic1out, pic2out

    def _shiftAndAdd(self, pic1, pic2):
        if len(pic1.shape) <= 2:
            pic1 = self.cv2.cvtColor(pic1, self.cv2.COLOR_GRAY2RGB)
        if len(pic2.shape) <= 2:
            pic2 = self.cv2.cvtColor(pic2, self.cv2.COLOR_GRAY2RGB)

        rows1, cols1 = pic1.shape[:2]
        rows2, cols2 = pic2.shape[:2]
        M = self.np.float32([[1, 0, cols1], [0, 1, 0]])
        pic_shifted = self.cv2.warpAffine(
            pic2, M, (cols1+cols2, max(rows1, rows2)))

        # add picture to larger picture
        pic_shifted[0:rows1, 0:cols1] = pic1
        return pic_shifted

    def shiftAndAdd(self, pic1, pic2, factor=1, rescale=False):
        # https://docs.opencv.org/4.x/d0/d86/tutorial_py_image_arithmetics.html
        if rescale:
            return self._shiftAndAdd(*self._scaleFigures(
                *self._makeSameSizeByScaling(pic1, pic2), factor))
        else:
            return self._shiftAndAdd(*self._scaleFigures(pic1, pic2, factor))

    def getDifferenceBetweenPictures(self, pic1, pic2):
        if not(self.np.all(self.np.equal(pic1.shape, pic2.shape))):
            print('Pictures must by same shape, picture1 is',
                  pic1.shape, 'picture2 is ', pic2.shape)
        else:
            pic_diff = pic1.astype('float32')-pic2.astype('float32')
            self.np.abs(pic_diff)
            return pic_diff.astype('uint8')

    def makeBinaryPicture(self, pic):
        grayImage = self.cv2.cvtColor(pic, self.cv2.COLOR_BGR2GRAY)
        _, bwImage = self.cv2.threshold(
            grayImage, 127, 255, self.cv2.THRESH_BINARY)
        return bwImage

    def addPostFixToImageName(self, picLocation, postfix=""):
        pathname, extension = os.path.splitext(picLocation)
        return pathname + postfix + extension

    def makeBinaryPictureAndSave(self, pic, picLocation):
        pic_bw = self.makeBinaryPicture(pic)
        picloc = self.addPostFixToImageName(picLocation, "_baw")
        self.cv2.imwrite(picloc, pic_bw)
        return pic_bw

    def shiftAndAddHorizontal(self, pic1, pic2):
        if len(pic1.shape) <= 2:
            pic1 = self.cv2.cvtColor(pic1, self.cv2.COLOR_GRAY2BGR)
        if len(pic2.shape) <= 2:
            pic2 = self.cv2.cvtColor(pic2, self.cv2.COLOR_GRAY2BGR)

        h1, w1 = pic1.shape[:2]
        h2, w2 = pic2.shape[:2]
        pic_out = self.np.zeros((max(h1, h2), w1+w2, 3), dtype=self.np.uint8)
        pic_out[:, :] = (0, 0, 0)
        # [:,:,:3] to ignore a transparency channel
        pic_out[:h1, :w1, :3] = pic1[:, :, :3]
        pic_out[:h2, w1:(w1+w2), :3] = pic2[:, :, :3]
        # self.cv2.imshow("show from within shiftAndAddHorizontal", pic_out)
        return pic_out

    def shiftAndAddHorizontal3(self, pic1, pic2, pic3, factor=1.0):
        pic1 = self.scaleImage(pic1, factor)
        pic2 = self.scaleImage(pic2, factor)
        pic3 = self.scaleImage(pic3, factor)
        out = self.shiftAndAddHorizontal(pic1, pic2)
        return self.shiftAndAddHorizontal(out, pic3)

    def scaleImage(self, pic, factor=1.0):
        if factor < 1.0:
            h = int(pic.shape[0]/2)
            w = int(pic.shape[1]/2)
            return self.cv2.resize(pic, (w, h), interpolation=self.cv2.INTER_CUBIC)
        else:
            return pic

    def makePyramids(self, pic, numberDown=2, numberUp=0):
        layerup = pic.copy()
        layerdown = pic.copy()
        gp = [layerdown]

        for i in range(numberDown):
            layerdown = self.cv2.pyrDown(layerdown)
            gp.append(layerdown)
        for i in range(numberUp):
            layerup = self.cv2.pyrUp(layerup)
            gp.append(layerup)
        return gp
