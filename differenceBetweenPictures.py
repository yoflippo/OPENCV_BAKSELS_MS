import numpy as np


def getDifferenceBetweenPictures(pic1, pic2):
    if not(np.all(np.equal(pic1.shape, pic2.shape))):
        print('Pictures must by same shape, picture1 is',
              pic1.shape, 'picture2 is ', pic2.shape)
    else:
        pic_diff = pic1.astype('float32')-pic2.astype('float32')
        np.abs(pic_diff)
        return pic_diff.astype('uint8')
