import cv2
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':
    f = cv2.imread('../../video/road-photo.jpg')
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    lines_img = np.copy(f)
    edges = cv2.Canny(gray,100,600,apertureSize=3)

    lines = cv2.HoughLines(edges,1,3.14159/180,250)
    print(lines)
    for rho,theta in lines[:,0]:
        cos = np.cos(theta)
        sin = np.sin(theta)
        x0 = cos*rho
        y0 = sin*rho
        x1 = int(x0 + 10000*(-sin))
        y1 = int(y0 + 10000*cos)
        x2 = int(x0 - 10000*(-sin))
        y2 = int(y0 - 10000*cos)

        cv2.line(lines_img,(x1,y1),(x2,y2),(0,255,255),5)
    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.title('original image')
    plt.imshow(f)

    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.title('canny image')
    plt.imshow(edges, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.title('lines detected')
    plt.imshow(lines_img)
    plt.show()

