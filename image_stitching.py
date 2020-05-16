import numpy as np
import cv2
import sys
from matchers import SIFTMatcher
import time
import blend

class Stitch:
    def __init__(self, args):
        self.path = args
        fp = open(self.path, 'r')
        filenames = [each.rstrip('\r\n') for each in fp.readlines()]
        # filenames = args
        print(filenames)
        # self.images = [cv2.resize(cv2.imread(each), (480, 320)) for each in filenames]
        self.images = [cv2.imread(each) for each in filenames]
        self.count = len(self.images)
        self.left_list, self.right_list, self.center_im = [], [], None
        self.matcher_obj = SIFTMatcher()
        self.prepare_lists()

    def prepare_lists(self):
        print("Number of images : %d" % self.count)
        self.centerIdx = self.count / 2
        print("Center index image : %d" % self.centerIdx)
        self.center_im = self.images[int(self.centerIdx)]
        for i in range(self.count):
            if (i <= self.centerIdx):
                self.left_list.append(self.images[i])
            else:
                self.right_list.append(self.images[i])
        print("Image lists prepared")

    def leftshift(self):
        # self.left_list = reversed(self.left_list)
        a = self.left_list[0]
        for b in self.left_list[1:]:
            H = self.matcher_obj.match(a, b, 'left')
            # print("Homography is : ", H)
            xh = np.linalg.inv(H)
            # print("Inverse Homography :", xh)
            br = np.dot(xh, np.array([a.shape[1], a.shape[0], 1]))
            br = br /br[-1]
            tl = np.dot(xh, np.array([0, 0, 1]))
            tl = tl / tl[-1]
            bl = np.dot(xh, np.array([0, a.shape[0], 1]))
            bl = bl / bl[-1]
            tr = np.dot(xh, np.array([a.shape[1], 0, 1]))
            tr = tr / tr[-1]
            cx = int(max([0, a.shape[1], tl[0], bl[0], tr[0], br[0]]))
            cy = int(max([0, a.shape[0], tl[1], bl[1], tr[1], br[1]]))
            offset = [abs(int(min([0, a.shape[1], tl[0], bl[0], tr[0], br[0]]))),
                      abs(int(min([0, a.shape[0], tl[1], bl[1], tr[1], br[1]])))]
            dsize = (cx + offset[0], cy + offset[1])
            print("image dsize =>", dsize, "offset", offset)

            tl[0:2] += offset; bl[0:2] += offset;  tr[0:2] += offset; br[0:2] += offset
            dstpoints = np.array([tl, bl, tr, br]);
            srcpoints = np.array([[0, 0], [0, a.shape[0]], [a.shape[1], 0], [a.shape[1], a.shape[0]]])
            # print('sp',sp,'dp',dp)
            M_off = cv2.findHomography(srcpoints, dstpoints)[0]
            # print('M_off', M_off)
            warped_img2 = cv2.warpPerspective(a, M_off, dsize)
            # cv2.imshow("warped", warped_img2)
            # cv2.waitKey()
            warped_img1 = np.zeros([dsize[1], dsize[0], 3], np.uint8)
            warped_img1[offset[1]:b.shape[0] + offset[1], offset[0]:b.shape[1] + offset[0]] = b
            tmp = blend.blend_linear(warped_img1, warped_img2)
            a = tmp

        self.leftImage = tmp

    def rightshift(self):
        for each in self.right_list:
            H = self.matcher_obj.match(self.leftImage, each, 'right')
            # print("Homography :", H)
            br = np.dot(H, np.array([each.shape[1], each.shape[0], 1]))
            br = br / br[-1]
            tl = np.dot(H, np.array([0, 0, 1]))
            tl = tl / tl[-1]
            bl = np.dot(H, np.array([0, each.shape[0], 1]))
            bl = bl / bl[-1]
            tr = np.dot(H, np.array([each.shape[1], 0, 1]))
            tr = tr / tr[-1]
            cx = int(max([0, self.leftImage.shape[1], tl[0], bl[0], tr[0], br[0]]))
            cy = int(max([0, self.leftImage.shape[0], tl[1], bl[1], tr[1], br[1]]))
            offset = [abs(int(min([0, self.leftImage.shape[1], tl[0], bl[0], tr[0], br[0]]))),
                      abs(int(min([0, self.leftImage.shape[0], tl[1], bl[1], tr[1], br[1]])))]
            dsize = (cx + offset[0], cy + offset[1])
            print("image dsize =>", dsize, "offset", offset)

            tl[0:2] += offset; bl[0:2] += offset; tr[0:2] += offset; br[0:2] += offset
            dstpoints = np.array([tl, bl, tr, br]);
            srcpoints = np.array([[0, 0], [0, each.shape[0]], [each.shape[1], 0], [each.shape[1], each.shape[0]]])
            M_off = cv2.findHomography(dstpoints, srcpoints)[0]
            warped_img2 = cv2.warpPerspective(each, M_off, dsize, flags=cv2.WARP_INVERSE_MAP)
            # cv2.imshow("warped", warped_img2)
            # cv2.waitKey()
            warped_img1 = np.zeros([dsize[1], dsize[0], 3], np.uint8)
            warped_img1[offset[1]:self.leftImage.shape[0] + offset[1], offset[0]:self.leftImage.shape[1] + offset[0]] = self.leftImage
            tmp = blend.blend_linear(warped_img1, warped_img2)
            self.leftImage = tmp
            
        self.rightImage = tmp
    
    def showImage(self, string=None):
        if string == 'left':
            cv2.imshow("left image", self.leftImage)
        elif string == "right":
            cv2.imshow("right Image", self.rightImage)
        cv2.waitKey()


if __name__ == '__main__':
    try:
        args = sys.argv[1]
    except:
        args = "txtlists/files4.txt"
    finally:
        print("Parameters : ", args)
    s = Stitch(args)
    
    # images = ['images/S1.jpg', 'images/S2.jpg','images/S3.jpg','images/S5.jpg','images/S6.jpg']
    # images = ['images/trees_00{}Hill.jpg'.format(i) for i in range(0, 4)]
    # s = Stitch(images)

    s.leftshift()
    # s.showImage('left')
    s.rightshift()
    print("done")
    cv2.imwrite("results/test4.jpg", s.leftImage)
    print("image written")
    cv2.destroyAllWindows()
        
