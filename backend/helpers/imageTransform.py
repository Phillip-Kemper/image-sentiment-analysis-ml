import cv2

def transformBlackWhite(data):
    originalImage = cv2.imread('/Users/phillip/Desktop/happy.jpeg')
    grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Original image', originalImage)
    cv2.imshow('Gray image', grayImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return grayImage




print(transformBlackWhite(''))
