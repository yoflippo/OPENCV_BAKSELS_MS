import cv2
import opencv_ms_helper
import numpy as np

h = opencv_ms_helper.opencv_ms_helper(cv2, np)

pic1 = cv2.imread("./Images/einstein.jpg", 1)
pic1 = h.scaleImage(pic1, 0.7)

if pic1 is not None:
    print("succesfully read picture")
else:
    print("DID NOT READ PICTURE")

pic1 = h.applyGaussianBlur(pic1, 11)
_, mag, orien = h.gradientColor(pic1)
mag = h.applyNormalizationToFloatImage(mag)
orien = h.applyNormalizationToFloatImage(orien)
oriencolor, _, _ = h.gradientColor(pic1)

sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
sobel_y = np.transpose(sobel_x)

sobelx = cv2.filter2D(pic1, cv2.CV_64F, sobel_x)
sobely = cv2.filter2D(pic1, cv2.CV_64F, sobel_y)
sobelx = h.applyNormalizationToFloatImage(sobelx)
sobely = h.applyNormalizationToFloatImage(sobely)

# sobelx = cv2.Sobel(pic1, cv2.CV_32F, 1, 0)
sobelx = h.applyTextToImage(sobelx, "Sobel X")
# sobely = cv2.Sobel(pic1, cv2.CV_32F, 0, 1)
sobely = h.applyTextToImage(sobely, "Sobel Y")

pic1 = h.applyTextToImage(pic1, "Normal")
mag = h.applyTextToImage(mag, "magnitude")
orien = h.applyTextToImage(oriencolor, "orientation (in color)")

cv2.imshow("normal, sobelx, sobely", h.shiftAndAddHorizontal3(
    pic1, sobelx, sobely))

cv2.imshow("normal, magnitude, orientation", h.shiftAndAddHorizontal3(
    pic1, mag, orien))

cv2.waitKey(0)
cv2.destroyAllWindows()
