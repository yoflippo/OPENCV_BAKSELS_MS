import cv2
import numpy as np
from matplotlib import pyplot as plt

pic = cv2.imread("./Images/colorful_umbrella.jpg",
                 cv2.IMREAD_UNCHANGED)  # color
# pic = cv2.imread("./Images/New_Zealand_Lake.jpg", 0) #gray
# cv2.imshow('normal', pic)
pic = pic[:, :, ::-1]


def makeSubplot(pic, title='normal', x=2, y=2, i=1):
    plt.subplot(x, y, i),
    plt.imshow(pic),
    plt.title(title),
    plt.xticks([]), plt.yticks([])


makeSubplot(pic, 'normal', 2, 2, 1)

piclowercontrast = (pic/2).astype('uint8')
makeSubplot(piclowercontrast, 'lower contrast', 2, 2, 2)

to_add = 100
# piclighter = pic
# piclighter[piclighter > (255-to_add)] = 255 - to_add
# piclighter += to_add
# makeSubplot(piclighter, 'lighter picture', 2, 2, 3)
piclighter = np.array(pic, dtype=np.uint64) + to_add
np.clip(piclighter, 0, 255, out=piclighter)
makeSubplot(piclighter.astype('uint8'), 'lighter picture', 2, 2, 3)

picinverted = 255-pic
makeSubplot(picinverted, 'inverted picture', 2, 2, 4)

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
