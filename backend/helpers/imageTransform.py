import cv2

def transformBlackWhite(data):
    originalImage = cv2.imread('/Users/phillip/Desktop/happy.jpeg')
    grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Original image', originalImage)
    cv2.imshow('Gray image', grayImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return grayImage

def transformDimensions(data):
    originalImage = cv2.imread('/Users/phillip/Desktop/happy.jpeg')
    width = 48
    height = 48
    dim = (width, height)
    resized = cv2.resize(originalImage, dim, interpolation=cv2.INTER_AREA)
    print('Resized Dimensions : ', resized.shape)
    cv2.imshow("Resized image", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


print(transformBlackWhite(''))
