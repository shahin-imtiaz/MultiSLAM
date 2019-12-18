# Incomplete at the moment.
# Enhancement still in progress.

import cv2
from matplotlib import pyplot as plt

class ImageStitcher:
    def __init__(self, left_img, right_img):
        this.left = left_img
        this.right = right_img

    def stitch():
        sift = cv2.xfeatures2d.SIFT_create()
        left_kp, left_des = sift.detectAndCompute(this.left, None)
        right_kp, right_des = sift.detectAndCompute(this.right, None)
        matches = cv2.BFMatcher().knnMatch(
            queryDescriptors=right_des,
            trainDescriptors=left_des,
            k=2
        )

        to_pts = []
        from_pts = []
        for m, n in matches:
            if m.distance < 0.65 * n.distance:
                from_pts.append(left_kp[m.trainIdx].pt)
                to_pts.append(right_kp[m.queryIdx].pt)

       H = cv2.findHomography(np.float32(from_pts), np.float32(to_pts), cv2.RANSAC, 2.0)[0]

       left_height, left_width = this.left.shape[:2]
       warped = np.zeros((left_height, int(left_width*1.75), 3), dtype=left_img.dtype)
       for c in range(3):
           warped[:,:,c] = cv2.warpPerspective(this.left[:,:,c], H, (warped.shape[1], warped.shape[0]), flags=cv2.INTER_NEAREST)
       left_mask = np.ones(warped.shape[:2], warped.dtype) * 255

       return warped

if __name__ == '__main__':
    # use some images from kitti for demonstration
    left = cv2.imread('../../videos/cam1.jpg')
    right = cv2.imread('../../videos/cam2.jpg')
    
    out = ImageStitcher(left, right).stitch()
    cv2.imshow(out)
