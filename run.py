import cv2
import numpy as np

img = cv2.imread('image.jpg')

pts_src= [
    [2483,1729],
    [2659,1716],
    [2651,2004],
    [2455,2012]
]

ax, ay= pts_src[0]
bx, by= pts_src[1]
cx, cy= pts_src[2]
dx, dy= pts_src[3]

two_m= 275
one_m= two_m//2

pts_dst= [
    [ax, ay],
    [ax+ one_m, ay],
    [ax+ one_m, ay+ two_m],
    [ax, ay+ two_m]
]

#pts = np.array([[864, 651], [1016, 581], [1205, 667], [1058, 759]], dtype=np.float32)
pts= np.array(pts_src, dtype=np.float32)
for pt in pts:
    cv2.circle(img, tuple(pt.astype(np.int)), 1, (0,0,255), -1)

# compute IPM matrix and apply it
#ipm_pts = np.array([[448,609], [580,609], [580,741], [448,741]], dtype=np.float32)
ipm_pts= np.array(pts_dst, dtype=np.float32)
ipm_matrix = cv2.getPerspectiveTransform(pts, ipm_pts)
ipm = cv2.warpPerspective(img, ipm_matrix, img.shape[:2][::-1])

# display (or save) images
#cv2.imshow('img', img)
#cv2.imshow('ipm', ipm)
#cv2.waitKey()

cv2.imwrite('image_ipm.jpg', ipm)
