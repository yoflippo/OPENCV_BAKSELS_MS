import cv2
import opencv_ms_helper
import numpy as np
h = opencv_ms_helper.opencv_ms_helper(cv2, np)

windowname = 'blobstorage'
threshold1Name = 'minarea'
threshold2Name = 'maxarea'
threshold3Name = 'minthresh'
threshold4Name = 'maxthresh'
threshold5Name = 'disblobs'
threshold6Name = 'minrep'


def funcCan(w):
    tMinArea = cv2.getTrackbarPos(threshold1Name, windowname)
    tMaxArea = cv2.getTrackbarPos(threshold2Name, windowname)
    tmin = cv2.getTrackbarPos(threshold3Name, windowname)
    tmax = cv2.getTrackbarPos(threshold4Name, windowname)
    disb = cv2.getTrackbarPos(threshold5Name, windowname)
    minrep = cv2.getTrackbarPos(threshold6Name, windowname)

    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = tmin
    params.maxThreshold = tmax
    params.filterByArea = True
    params.minArea = tMinArea
    params.maxArea = tMaxArea
    params.filterByCircularity = True
    params.minCircularity = 0.3
    params.filterByConvexity = True
    params.minConvexity = 0.5
    params.filterByInertia = True
    params.minInertiaRatio = 0.2
    params.minDistBetweenBlobs = disb
    params.minRepeatability = minrep

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)

    blank = np.zeros((1, 1))
    blobs = cv2.drawKeypoints(img, keypoints, blank, (0, 0, 255),
                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    text = "Number of Circular Blobs: " + str(len(keypoints))
    cv2.putText(blobs, text, (20, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
    cv2.imshow(windowname, blobs)


if __name__ == '__main__':

    original = cv2.imread("./Images/img_coins_0.jpg", 0)
    img = original.copy()
    img = h.scaleImage(img, 0.2)

    cv2.namedWindow(windowname)
    cv2.resizeWindow(windowname, 700, 400)

    cv2.createTrackbar(threshold1Name, windowname, 1, 255, funcCan)
    cv2.createTrackbar(threshold2Name, windowname, 127, 255*16, funcCan)
    cv2.createTrackbar(threshold3Name, windowname, 1, 255, funcCan)
    cv2.createTrackbar(threshold4Name, windowname, 200, 255, funcCan)
    cv2.createTrackbar(threshold5Name, windowname, 10, 255, funcCan)
    cv2.createTrackbar(threshold6Name, windowname, 10, 255, funcCan)
    funcCan(windowname)

    cv2.waitKey(0)

cv2.destroyAllWindows()
