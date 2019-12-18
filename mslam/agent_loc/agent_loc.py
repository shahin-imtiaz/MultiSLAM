import cv2
import numpy as np

'''
Agent Location
Utilizes the OpenCV library: https://opencv.org/
Computes the movement of the agent based on frame by frame differences
'''
class AgentLocate:
    # Initialize SIFT and Flann Matcher
    def __init__(self, kpNum=1000, enableGlobal=False, match_threshold=200, debugging=False):
        self.debugging = debugging
        self.kpNum = kpNum  # Max keypoints to detect
        self.match_threshold = match_threshold # Distance threshold
        self.enableGlobal = enableGlobal

        self.sift = cv2.xfeatures2d.SIFT_create(self.kpNum)
        self.FLANN_INDEX_KDTREE = 1
        self.index_params = dict(algorithm = self.FLANN_INDEX_KDTREE, trees = 5)
        self.search_params = dict(checks=50)
        self.flannFrameByFrame = cv2.FlannBasedMatcher(self.index_params,self.search_params)
        self.prevFrameKP = None
        
        self.flannGlobal = None
        if self.enableGlobal:
            self.flannGlobal = cv2.FlannBasedMatcher(self.index_params,self.search_params)
        
        self.smooth_history = 8
        self.smooth_x_magnitude = []
        self.smooth_y_magnitude = []
        self.smooth_translate = []
        self.smooth_rotate = []
        self.letter_to_move = {
            's': 0,
            'f': 1,
            'r': 2,
            'l': 3,
        }
        self.move_to_letter = {
            0: 's',
            1: 'f', 
            2: 'r', 
            3: 'l', 
        }

    # Return the descriptors saved in the given flann matcher
    def getTrainDescriptors(self, matcher):
        if matcher == "global" and self.enableGlobal:
            return self.flannGlobal.getTrainDescriptors()
        elif matcher == "frameByframe":
            return self.flannFrameByFrame.getTrainDescriptors()
        return None

    # Returns True iff a given keypoint is a euclidean distance of 20 away from ALL other saved keypoints
    # And it is not in the middle region
    def pruneKP(self, kp, kpList, middle, thresh=30):
        cur_pt = np.array(kp.pt)
        if np.linalg.norm(cur_pt - middle) < 100:
                return False
        for i in kpList:
            i_pt = np.array(i.pt)
            if np.linalg.norm(cur_pt - i_pt) < thresh:
                return False
        return True

    def smoothMove(self, mov):
        if len(self.smooth_translate) < self.smooth_history:
            self.smooth_translate.append(mov[0])
            self.smooth_rotate.append(mov[1])
            self.smooth_x_magnitude.append(mov[2])
            self.smooth_y_magnitude.append(mov[3])
            out = [max(set(self.smooth_translate), key=self.smooth_translate.count),
                    max(set(self.smooth_rotate), key=self.smooth_rotate.count),
                    max(set(self.smooth_x_magnitude), key=self.smooth_x_magnitude.count),
                    max(set(self.smooth_y_magnitude), key=self.smooth_y_magnitude.count)]
            if self.debugging:
                if out[2] is None or out[3] is None:
                    print(out, 'mag=', None)
                else:
                    print(out, 'mag=', np.hypot(out[2], out[3]))
            return out
        
        self.smooth_translate.pop(0)
        self.smooth_rotate.pop(0)
        self.smooth_x_magnitude.pop(0)
        self.smooth_y_magnitude.pop(0)
        self.smooth_translate.append(mov[0])
        self.smooth_rotate.append(mov[1])
        self.smooth_x_magnitude.append(mov[2])
        self.smooth_y_magnitude.append(mov[3])
        out = [max(set(self.smooth_translate), key=self.smooth_translate.count),
                max(set(self.smooth_rotate), key=self.smooth_rotate.count),
                max(set(self.smooth_x_magnitude), key=self.smooth_x_magnitude.count),
                max(set(self.smooth_y_magnitude), key=self.smooth_y_magnitude.count)]
        if self.debugging:
            if out[2] is None or out[3] is None:
                print(out, 'mag=', None)
            else:
                print(out, 'mag=', np.hypot(out[2], out[3]))
        return out
        

    # Return the camera translation and rotation matrix for the movement between frames
    def getViewTransform(self, matches, kp, imgShape):
        # Divide the image into 4x4 quadrants for each axis
        quad_width = imgShape[1] / 4
        quad_height = imgShape[0] / 4
        quads_x = np.zeros((4,4))
        quads_y = np.zeros((4,4))

        # Initialize with a constant value, otherwise median of an empty list = NaN
        x_magnitude = [0.00005]
        y_magnitude = [0.00005]

        # Reduce the difference in each axis for matching points to a numerical sign and sum over differences in each quadrant
        # Place it in the quadrant belonging to the x, y location of the keypoint in the current frame.
        for i in range(len(matches)):
            x, y = kp[matches[i][0].queryIdx].pt
            xh, yh = self.prevFrameKP[matches[i][0].trainIdx].pt
            qx = int(x // quad_width)
            qy = int(y // quad_height)
            x_diff = x-xh
            y_diff = y-yh

            if qy in [0, 3]:
                x_magnitude.append(abs(x_diff))
            if qx in [0, 3]:
                y_magnitude.append(abs(y_diff))

            quads_x[qy, qx] += x_diff
            quads_y[qy, qx] += y_diff

            if x_diff < 0:
                quads_x[qy, qx] -= 1
            else:
                quads_x[qy, qx] += 1
            
            if y_diff < 0:
                quads_y[qy, qx] -= 1
            else:
                quads_y[qy, qx] += 1

        # Calculate the overall numerical sign of each quadrant
        quads_x = np.where(quads_x < 0, -1, quads_x)
        quads_x = np.where(quads_x > 0, 1, quads_x)
        quads_y = np.where(quads_y < 0, -1, quads_y)
        quads_y = np.where(quads_y > 0, 1, quads_y)

        print("QUADS X")
        print(quads_x)

        print("QUADS_Y")
        print(quads_y)

        # Transformation logic:
        # Move in the forward direction iff all conditions are met:
        #   - Majority of Y quadrants in the top most row are 0 or negative
        #   - Majority of X quadrants in the right most column are 0 or positive
        #   - Majority of Y quadrants in the bottom most row are 0 or positive
        #   - Majority of X quadrants in the left most column are 0 or negative
        # 
        # Rotate the camera right iff:
        #   - Majority of X quadrants are 0 or negative
        #
        # Rotate the camera left iff:
        #   - Majority of X quadrants are 0 or positive
        #
        # Rotate the camera up iff:
        #   - Majority of Y quadrants are 0 or positive
        #
        # Rotate the camera down iff:
        #   - Majority of Y quadrants are 0 or negative
        
        # [translation, rotation, x magnitude, y magnitude]
        mov = [None, None, np.median(x_magnitude), np.median(y_magnitude)]

        if (np.sum(quads_y[0]) == 0 and np.sum(quads_y[3]) == 0 and
            np.sum(quads_x[:,0]) == 0 and np.sum(quads_x[:,3]) == 0):
            # if self.debugging:
            #     print('Staying')
            mov[0] = 's'
        elif (np.sum(quads_y[0]) <= 0 and np.sum(quads_y[3]) >= 0 and
            np.sum(quads_x[:,0]) <= 0 and np.sum(quads_x[:,3]) >= 0):
            # if self.debugging:
            #     print('Moving Forward')
            mov[0] = 'f'
        
        # x_sum = stats.mode([quads_x[0,0], quads_x[1,0], quads_x[2,0], quads_x[3,0], quads_x[0,1], quads_x[0,2], quads_x[0,3], quads_x[1,3], quads_x[2,3], quads_x[3,3]])[0][0]
        # x_sum = np.median([quads_x[1,0], quads_x[2,0], quads_x[3,0], quads_x[1,3], quads_x[2,3], quads_x[3,3]])
        # x_sum = np.median(quads_x)
        # x_sum = np.average(quads_x)
        x_sum = np.sum(quads_x)
        # x_sum = scipy.stats.mode(quads_x, axis=None)[0][0]
        # x_sum = scipy.stats.mode(np.array([quads_x[0,0], quads_x[1,0], quads_x[2,0], quads_x[3,0], quads_x[0,1], quads_x[0,2], quads_x[0,3], quads_x[1,3], quads_x[2,3], quads_x[3,3]]), axis=None)[0][0]
        print('xsum is:', x_sum)
        if (x_sum < -2):
            # if self.debugging:
            #     print('Rotate Right', x_sum)
            mov[1] = 'r'
        elif (x_sum > 2):
            # if self.debugging:
            #     print('Rotate Left', x_sum)
            mov[1] = 'l'
        # elif (np.sum(quads_y) < 0):
        #     if self.debugging:
        #         print('Rotate Down')
        #     mov[1] = 'd'
        # elif (np.sum(quads_y) > 0):
        #     if self.debugging:
        #         print('Rotate Up')
        #     mov[1] = 'u'
        else:
            pass
            # if self.debugging:
                # print('No Rotate', x_sum)

        # No movement
        return self.smoothMove(mov)

    # Compute the change in agent location based on previous frame
    def estimate(self, img, kpNum=1000):
        kp, des = self.sift.detectAndCompute(img,None)
        if len(kp) == 0 or des is None:
            return {'frame': np.zeros(img.shape, dtype='uint8') * 255, 'transform': [None, None, None, None]}

        # Sort keypoints by response
        kp, des = zip(*(sorted(zip(kp, des), key=lambda x: x[0].response, reverse=True)))
        
        # Prune keypoints that are close together
        # And around the middle of the frame
        kp_apart = []
        des_apart = []
        middle = np.array([img.shape[1]//2, img.shape[0]//2])
        for i in range(len(kp)):
            if self.pruneKP(kp[i], kp_apart, middle):
                kp_apart.append(kp[i])
                des_apart.append(des[i])

        if kpNum <= len(kp_apart):
            kp = kp_apart[:kpNum]
            des = des_apart[:kpNum]
        else:
            kp = kp_apart
            des = des_apart

        # First frame. Add to empty training set and return blank frame
        if len(self.getTrainDescriptors('frameByframe')) == 0:
            self.flannFrameByFrame.add([np.array(des)])
            self.flannFrameByFrame.train()
            
            if self.enableGlobal:
                self.flannGlobal.add([np.array(des)])
                self.flannGlobal.train()

            self.prevFrameKP = kp
            return {'frame': np.zeros(img.shape, dtype='uint8') * 255, 'transform': [None, None, None, None]}

        # Match the current frame with the previous one
        matchesFrameByFrame = self.flannFrameByFrame.knnMatch(np.array(des), k=1)
        matchesFrameByFrame = [x for x in matchesFrameByFrame if x[0].distance < self.match_threshold]
        # Remove the previous frame's descriptors and add in the current one's to the matcher
        self.flannFrameByFrame.clear()
        self.flannFrameByFrame.add([np.array(des)])
        self.flannFrameByFrame.train()

        # Match the current frame with all previous frames and add to the descriptor collection
        if self.enableGlobal:
            matchesGlobal = self.flannGlobal.knnMatch(np.array(des), k=1)
            matchesGlobal = [x for x in matchesGlobal if x[0].distance < self.match_threshold]
            self.flannGlobal.add([np.array(des)])
            self.flannGlobal.train()

        transform = self.getViewTransform(matchesFrameByFrame, kp, img.shape)
        self.prevFrameKP = kp
        outKPFrame = cv2.drawKeypoints(img,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        return {'frame': outKPFrame, 'transform': transform}