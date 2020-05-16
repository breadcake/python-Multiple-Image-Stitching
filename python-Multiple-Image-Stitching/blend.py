import cv2
import numpy as np
import matplotlib.pyplot as plt

def blend_linear(warp_img1, warp_img2):
    img1 = warp_img1
    img2 = warp_img2

    img1mask = ((img1[:,:,0] | img1[:,:,1] | img1[:,:,2]) >0)
    img2mask = ((img2[:,:,0] | img2[:,:,1] | img2[:,:,2]) >0)

    r,c = np.nonzero(img1mask)
    out_1_center = [np.mean(r),np.mean(c)]
    
    r,c = np.nonzero(img2mask)
    out_2_center = [np.mean(r),np.mean(c)]

    vec = np.array(out_2_center) - np.array(out_1_center)
    intsct_mask = img1mask & img2mask
##    plt.gray()
##    plt.subplot(311),plt.imshow(img1mask)
##    plt.plot(out_1_center[1], out_1_center[0], 'ob')
##    plt.subplot(312),plt.imshow(img2mask)
##    plt.plot(out_2_center[1], out_2_center[0], 'ob')
##    plt.subplot(313),plt.imshow(intsct_mask)
##    plt.show()

    r,c = np.nonzero(intsct_mask)
    # def sub2ind(array_shape, rows, cols):
    #     return cols*array_shape[0] + rows
    # idx = sub2ind(img2mask[:,2], r, c)
    out_wmask = np.zeros(img2mask.shape[:2])
    proj_val = (r - out_1_center[0])*vec[0] + (c- out_1_center[1])*vec[1]
    out_wmask[r,c] = (proj_val - (min(proj_val)+(1e-3))) / \
                     ((max(proj_val)-(1e-3)) - (min(proj_val)+(1e-3)))
    
    # blending
    mask1 = img1mask & (out_wmask==0)
    mask2 = out_wmask
    mask3 = img2mask & (out_wmask==0)
##    plt.gray()
##    plt.subplot(311),plt.imshow(mask1),plt.axis('off')
##    plt.subplot(312),plt.imshow(mask2),plt.axis('off')
##    plt.subplot(313),plt.imshow(mask3),plt.axis('off')
##    plt.savefig('blend.jpg')
##    plt.show()
    
    out = np.zeros(img1.shape)
    for c in range(3):
        out[:,:,c] = img1[:,:,c]*(mask1+(1-mask2)*(mask2!=0)) + \
                     img2[:,:,c]*(mask2+mask3)
    return np.uint8(out)

if __name__=="__main__":
    img1 = cv2.imread("warped_img1.jpg")
    img2 = cv2.imread("warped_img2.jpg")
    out = blend_linear(img1, img2)
    # cv2.imwrite("result.jpg",out)
    
