import cv2
import numpy as np

# Based on https://stackoverflow.com/a/57440865/92049

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

# Both ay-dy and by-cy are 2m apart;
# interpolate pixel distance and
# use as reference.
ref= (abs(ay-dy)+ abs(by-cy))//2
two_m= ref
one_m= two_m//2

ax_, ay_= ax, ay
bx_, by_= ax+ one_m, ay
cx_, cy_= ax+ one_m, ay+ two_m
dx_, dy_= ax, ay+ two_m

pts_dst= [
    [ax_, ay_],
    [bx_, by_],
    [cx_, cy_],
    [dx_, dy_]
]

def cross(img, x, y, s, c):
    """
    Draw cross with side length s and color c at position x/y.
    """
    s2= s//2
    cv2.line(img, (x, y-s2), (x, y+s2), c, 2)
    cv2.line(img, (x-s2, y), (x+s2, y), c, 2)

def cross_outline(img, x, y, s, c1, c2):
    cross(img, x+1, y, s, c2)
    cross(img, x-1, y, s, c2)
    cross(img, x, y+1, s, c2)
    cross(img, x, y-1, s, c2)
    cross(img, x, y, s, c1)


#pts = np.array([[864, 651], [1016, 581], [1205, 667], [1058, 759]], dtype=np.float32)
pts= np.array(pts_src, dtype=np.float32)
#for pt in pts:
#    cv2.circle(img, tuple(pt.astype(np.int)), 10, (0,0,255), -1)

# compute IPM matrix and apply it
#ipm_pts = np.array([[448,609], [580,609], [580,741], [448,741]], dtype=np.float32)
ipm_pts= np.array(pts_dst, dtype=np.float32)
ipm_matrix = cv2.getPerspectiveTransform(pts, ipm_pts)
ipm = cv2.warpPerspective(img, ipm_matrix, img.shape[:2][::-1])

for x, y in ipm_pts:
    cross_outline(ipm, int(x), int(y), 20, (0,0,255), (0,0,0))

for x, y in pts:
    cross_outline(img, int(x), int(y), 20, (0,0,255), (0,0,0))


# display (or save) images
#cv2.imshow('img', img)
#cv2.imshow('ipm', ipm)
#cv2.waitKey()


filename='input.jpg'
print(f'Writting {filename}...')
cv2.imwrite(filename, img, [int(cv2.IMWRITE_JPEG_QUALITY), 60])

filename='result.jpg'
print(f'Writting {filename}...')
cv2.imwrite(filename, ipm, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
