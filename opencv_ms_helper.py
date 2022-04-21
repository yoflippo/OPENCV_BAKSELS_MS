import os


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

    def __testAndGiveThreeDim(self, pic):
        if len(pic.shape) <= 2:
            pic = self.cv2.cvtColor(pic, self.cv2.COLOR_GRAY2BGR)
        return pic

    def __giveGrayImageThreeDim(self, pic1, pic2):
        return (self.__testAndGiveThreeDim(pic1),
                self.__testAndGiveThreeDim(pic2))

    def shiftAndAddVertical(self, pic1, pic2):
        pic1, pic2 = self.__giveGrayImageThreeDim(pic1, pic2)
        h1, w1 = pic1.shape[:2]
        h2, w2 = pic2.shape[:2]
        pic_out = self.np.zeros((h1+h2, max(w1, w2), 3), dtype=self.np.uint8)
        pic_out[:, :] = (0, 0, 0)
        # [:,:,:3] to ignore a transparency channel
        pic_out[:h1, :w1, :3] = pic1[:, :, :3]
        # pic_out[:h2, w1:(w1+w2), :3] = pic2[:, :, :3]
        pic_out[h1:(h1+h2), :w2, :3] = pic2[:, :, :3]
        # self.cv2.imshow("show from within shiftAndAddHorizontal", pic_out)
        return pic_out

    def shiftAndAddHorizontal(self, pic1, pic2):
        pic1, pic2 = self.__giveGrayImageThreeDim(pic1, pic2)

        h1, w1 = pic1.shape[:2]
        h2, w2 = pic2.shape[:2]
        pic_out = self.np.zeros((max(h1, h2), w1+w2, 3), dtype=self.np.uint8)
        pic_out[:, :] = (0, 0, 0)
        # [:,:,:3] to ignore a transparency channel
        pic_out[:h1, :w1, :3] = pic1[:, :, :3]
        pic_out[:h2, w1:(w1+w2), :3] = pic2[:, :, :3]
        # self.cv2.imshow("show from within shiftAndAddHorizontal", pic_out)
        return pic_out

    def combineFourImagesInQuadrant(self, pic_lh, pic_rh, pic_ll, pic_rl,
                                    factor=1.0):
        picup = self.shiftAndAddHorizontal(pic_lh, pic_rh)
        picdo = self.shiftAndAddHorizontal(pic_ll, pic_rl)
        picout = self.shiftAndAddVertical(picup, picdo)
        return self.scaleImage(picout, factor)

    def shiftAndAddHorizontal3(self, pic1, pic2, pic3, factor=1.0):
        if factor < 1.0 or factor > 1.0:
            pic1 = self.scaleImage(pic1, factor)
            pic2 = self.scaleImage(pic2, factor)
            pic3 = self.scaleImage(pic3, factor)
        out = self.shiftAndAddHorizontal(pic1, pic2)
        return self.shiftAndAddHorizontal(out, pic3)

    def scaleImage(self, pic, factor=1.0):
        h = int(pic.shape[0]*factor)
        w = int(pic.shape[1]*factor)
        return self.cv2.resize(pic, (w, h), interpolation=self.cv2.INTER_CUBIC)

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

    def shiftBinaryUDLR(self, pic_binary):
        def addOrImage(picbig, sh, eh, sw, ew, picsmall):
            picbig[sh:eh, sw:ew] = self.cv2.bitwise_or(
                picbig[sh:eh, sw:ew], picsmall)
            return picbig

        if len(pic_binary.shape) <= 2:
            h, w = pic_binary.shape[:2]
            pic_out = self.np.zeros((h+2, w+2), dtype=self.np.uint8)
            pic_out[:h, :w] = pic_binary
            pic_out = addOrImage(pic_out, 1, h+1, 1, w+1, pic_binary)
            pic_out = addOrImage(pic_out, 0, h+0, 1, w+1, pic_binary)
            pic_out = addOrImage(pic_out, 1, h+1, 0, w+0, pic_binary)
        return pic_out[:h, :w]

    def applyGaussianBlur(self, image, size=3):
        if size % 2 == 0:
            size + 1
        M = (size, size)
        return self.cv2.GaussianBlur(image, M, 0)

    def sharpenImageBasedOnGaussianBlur(self, pic, blurfactor=11):
        if blurfactor % 2 == 0:
            blurfactor = blurfactor + 1
        blurred = self.applyGaussianBlur(pic, blurfactor)
        return self.cv2.addWeighted(pic, 1.5, blurred, -0.5, 0)

    def applyTextToImage(self, image, txt):
        pic = self.__testAndGiveThreeDim(image)
        return self.cv2.putText(pic, text=txt, org=(30, 50),
                                fontFace=self.cv2.FONT_HERSHEY_PLAIN,
                                fontScale=3,
                                color=(0, 255, 0),
                                thickness=3)

    def applyNormalizationToFloatImage(self, pic):
        return self.cv2.normalize(pic, None, 255, 0, self.cv2.NORM_MINMAX,
                                  self.cv2.CV_8UC1)

    def gradientColor(self, pic, threshbinarylow=50):
        pic1 = self.applyGaussianBlur(pic, 3)
        sobelx = self.cv2.Sobel(pic1, self.cv2.CV_32F, 1, 0)
        sobely = self.cv2.Sobel(pic1, self.cv2.CV_32F, 0, 1)

        orien = self.cv2.phase(sobelx, sobely, angleInDegrees=True)
        mag = self.cv2.magnitude(sobelx, sobely)

        _, mask = self.cv2.threshold(
            mag, threshbinarylow, 255, self.cv2.THRESH_BINARY)

        red = self.np.array([0, 0, 255])
        cyan = self.np.array([255, 255, 0])
        green = self.np.array([0, 255, 0])
        yellow = self.np.array([0, 255, 255])
        black = self.np.array([0, 0, 0])

        orientcolor = self.np.zeros(
            (orien.shape[0], orien.shape[1], 3), dtype=self.np.uint8)
        orientcolor = self.np.add(self.np.where((mask == 255) & (
            orien < 90), red, black), orientcolor)
        orientcolor = self.np.add(
            self.np.where((mask == 255) &
                          (orien > 90) & (orien < 180),
                          cyan, black), orientcolor)
        orientcolor = self.np.add(
            self.np.where((mask == 255) & (orien > 180)
                          & (orien < 270), green, black),
            orientcolor)
        orientcolor = self.np.add(self.np.where((mask == 255) & (
            orien > 270), yellow, black), orientcolor)
        # Apply normalization because this is an 8 bit color image
        # which cv2.imshow() does not like
        return self.applyNormalizationToFloatImage(orientcolor)
