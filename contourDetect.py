import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def detectContours(img):
    img = cv.GaussianBlur(img, (5,5), 0)
    edges = cv.Canny(img, 100, 200)

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()


    # ret, thresh = cv.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # cv.drawContours(img, [cnt], 0, (0,255,0),3)
    cv.imshow("demo", img)
    cv.waitKey()

if __name__ == "__main__":
    im = cv.imread(
        "/media/zw/DL/ly/data/data_20191022/ori_frame/20191007185014_泰安宁阳河道采砂热成像/20191007185014_泰安宁阳河道采砂热成像_image_00152.jpg")
    detectContours(im)