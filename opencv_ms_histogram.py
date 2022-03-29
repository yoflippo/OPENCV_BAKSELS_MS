import cv2
from matplotlib import pyplot as plt

picloc = "./Images/colorful_umbrella.jpg"
picgray = cv2.imread(picloc, 0)
pic = cv2.imread(picloc, cv2.IMREAD_UNCHANGED)

# cv2.imshow("original", pic)


def getColorchannel(picture, keepchannel=1):
    pic = picture.copy()  # otherwise original picture will be messed up
    if keepchannel < 3:
        nameOfChannel = ['Blue', 'Green', 'Red']
        channels = [0, 1, 2]
        channels.remove(keepchannel)
        for ch in channels:
            pic[:, :, ch] = 0
    else:
        print('number of channels ranges from 0..2')
    return nameOfChannel[keepchannel], pic


def setPltLabels(plt, blIsHistogram=False):
    plt.xticks(color='w')
    plt.yticks(color='w')
    if blIsHistogram:
        plt.xlim([0, 256])


# plot original
plt.subplot(5, 2, 1)
plt.imshow(pic[:, :, ::-1])
setPltLabels(plt)

# plot histogram of original
plt.subplot(5, 2, 2)
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([pic], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    setPltLabels(plt, True)

# repeat plotting and histogram per colorchannel
for i in range(0, 3):
    plt.subplot(5, 2, 3+(2*i))
    name, channel = getColorchannel(pic, i)
    plt.imshow(channel[:, :, ::-1])
    plt.title(name)
    setPltLabels(plt)
    plt.subplot(5, 2, 4+(2*i))
    histr = cv2.calcHist([pic], [i], None, [256], [0, 256])
    plt.plot(histr, color=name)
    setPltLabels(plt, True)

# plot graychannel
plt.subplot(5, 2, 9)
plt.imshow(picgray, 'gray')
plt.title('gray')
setPltLabels(plt)
plt.subplot(5, 2, 10)
histr = cv2.calcHist(picgray, [0], None, [256], [0, 256])
plt.plot(histr)
setPltLabels(plt, True)


plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
