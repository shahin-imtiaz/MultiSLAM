# https://docs.opencv.org/3.4/da/df5/tutorial_py_sift_intro.html

import numpy as np
import cv2

class PointCloud:
    def __init__(self, kpNum=1000):
        self.kpNum = kpNum
        self.sift = cv2.xfeatures2d.SIFT_create(self.kpNum)
        self.FLANN_INDEX_KDTREE = 1
        self.index_params = dict(algorithm = self.FLANN_INDEX_KDTREE, trees = 5)
        self.search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(self.index_params,self.search_params)
        self.match_thresh = 5

    def getTrainDescriptors(self):
        return self.flann.getTrainDescriptors()

    def isApart(self, kp, kpList, thresh=50):
        cur_pt = np.array(kp.pt)

        for i in kpList:
            i_pt = np.array(i.pt)
            if np.linalg.norm(cur_pt - i_pt) < thresh:
                return False
        return True

    def estimate(self, img, kpNum=30):
        kp, des = self.sift.detectAndCompute(img,None)

        # sort keypoints by response
        kp, des = zip(*(sorted(zip(kp, des), key=lambda x: x[0].response, reverse=True)))
        
        # space apart keypoints
        kp_apart = []
        des_apart = []
        for i in range(len(kp)):
            if self.isApart(kp[i], kp_apart):
                kp_apart.append(kp[i])
                des_apart.append(des[i])

        if kpNum <= len(kp_apart):
            kp = kp_apart[:kpNum]
            des = des_apart[:kpNum]
        else:
            kp = kp_apart
            des = des_apart

        # matches = self.flann.radiusMatch(np.array(des), maxDistance=50)
        matches = self.flann.knnMatch(np.array(des), k=1)
        print([x[0].imgIdx for x in matches])
        matches = [x for x in matches if x[0].distance < 200]
        print(len(matches))
        # if len(matches) > 0:
        #     for i in range(len(matches)):
        #         print(matches[i][0].distance)

        self.flann.add([np.array(des)])
        self.flann.train()
        return (kp, matches)
        # return cv2.drawKeypoints(img,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


if __name__ == "__main__":
    pc = PointCloud()
    img1 = cv2.imread("input1.jpg")
    img2 = cv2.imread("input2.png")

    imgL = cv2.imread("imgL.png")
    imgR = cv2.imread("imgR.png")
    # cv2.imwrite("out1.png", pc.estimate(img1.copy()))
    # cv2.imwrite("out2.png", pc.estimate(img1.copy()))
    # zkp1, zm1 = pc.estimate(img1.copy())
    # zkp2, zm2 = pc.estimate(img1.copy())
    # cv2.imwrite("out.png", cv2.drawMatchesKnn(img1,zkp1,img1,zkp2,zm2,None))
    # pc.estimate(img1.copy())
    # pc.estimate(img2.copy())
    zkp1, zm1 = pc.estimate(imgL.copy())
    zkp2, zm2 = pc.estimate(imgR.copy())
    cv2.imwrite("out.png", cv2.drawMatchesKnn(imgR,zkp2,imgL,zkp1,zm2,None))