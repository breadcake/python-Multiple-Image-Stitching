import cv2
import numpy as np
import matplotlib.pyplot as plt

class SIFTMatcher():
    def __init__(self):
        self.sift = cv2.xfeatures2d_SIFT().create()
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def match(self,srcImg,testImg,direction):
        print("Direction : ", direction)
        
        img1gray = cv2.cvtColor(srcImg, cv2.COLOR_BGR2GRAY)
        img2gray = cv2.cvtColor(testImg, cv2.COLOR_BGR2GRAY)
        # find the keypoints and descriptors with SIFT
        kp1, des1 = self.sift.detectAndCompute(img1gray, None)
        kp2, des2 = self.sift.detectAndCompute(img2gray, None)

        matches = self.flann.knnMatch(des1, des2, k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0, 0] for i in range(len(matches))]

        good = []
        pts1 = []
        pts2 = []
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                good.append(m)
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)
                matchesMask[i] = [1, 0]
                
        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=0)
        img3 = cv2.drawMatchesKnn(srcImg, kp1, testImg, kp2, matches, None, **draw_params)
        # cv2.imwrite("matches.jpg",img3)
        # plt.imshow(img3, ), plt.show()

        rows, cols = srcImg.shape[:2]
        MIN_MATCH_COUNT = 10
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good])#.reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])#.reshape(-1, 1, 2)
            # print(src_pts, dst_pts)
            
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            draw_params = dict(matchColor = (0,255,0),
                               singlePointColor = (255, 0, 0),
                               matchesMask = matchesMask, # draw only inliers
                               flags = 2)
            img4 = cv2.drawMatches(srcImg,kp1,testImg,kp2,good,None,**draw_params)
            # cv2.imwrite("matches_ransac.jpg",img4)
            # plt.imshow(img4,), plt.show()
            # print('M', M)
            return M
        else:
            print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
            matchesMask = None
        return None

