import cv2
import numpy as np

def autocrop(image, resize,threshold=0):
    """Crops any edges below or equal to threshold

    Crops blank image to 1x1.

    Returns cropped image.

    """
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]
    
    image = cv2.resize(image, resize, interpolation = cv2.INTER_AREA)

    return image

# aloe = cv2.imread("test3.jpg")

# img = autocrop(aloe, (64,64))

# # from matplotlib import pyplot as plt
# # plt.imshow(img, interpolation='nearest')
# # plt.show()

# cv2.imshow("cropped", img)
# cv2.waitKey(2000)
# cv2.destroyAllWindows()

# #img = cv2.resize(img, (256,256), interpolation = cv2.INTER_AREA)

# cv2.imshow("resized", img)
# cv2.waitKey(2000)
# cv2.destroyAllWindows()

# cv2.imwrite("resulttest3.jpg", img)
