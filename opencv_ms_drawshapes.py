import numpy as np
import cv2

black_picture = np.zeros((500, 500, 3), dtype='uint8')

cv2.rectangle(black_picture, (0, 0), (500, 250),
              (255, 200, 98), 3, lineType=1, shift=0)

cv2.line(black_picture, (350, 350), (500, 350), (0, 0, 255), lineType=8)

color_circle = (255, 255, 0)
cv2.circle(black_picture, (250, 250), 150, color_circle, lineType=8)

font = cv2.FONT_HERSHEY_DUPLEX
cv2.putText(black_picture, 'thisisatext', (100, 100),
            font, 2, color_circle, 4, cv2.LINE_8)

cv2.imshow('dark', black_picture)
cv2.waitKey(0)
cv2.destroyAllWindows()
