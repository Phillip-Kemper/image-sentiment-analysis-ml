import cv2

def transformBlackWhite(data):
    grayImage = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    return grayImage

    #cv2.imshow('Original image', originalImage)
    #cv2.imshow('Gray image', grayImage)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return grayImage

def transformDimensions(data):
    width = 48  # as needed for fer-2013 dataset
    height = 48 #           "
    dim = (width, height)
    resized = cv2.resize(data, dim, interpolation=cv2.INTER_AREA)
    #print('Resized Dimensions : ', resized.shape)
    #cv2.imshow("Resized image", resized)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return resized



print(transformDimensions(''))
