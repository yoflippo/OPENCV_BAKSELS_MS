import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):
    # Take each frame
    _, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)

    cv2.putText(frame, str(fps), (7, 70), cv2.FONT_HERSHEY_COMPLEX,
                3, (100, 255, 0), 3, cv2.LINE_AA)

    orange = np.uint8([[[0, 215, 255]]])
    orange_hsv = cv2.cvtColor(orange, cv2.COLOR_BGR2HSV)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_color = np.array([10, 80, 200])
    upper_color = np.array([60, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
