import cv2
import numpy as np
import matplotlib.pyplot as plt

def writeVideo(outputFile, frameList):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('testOutput.mp4',fourcc, 20.0, (1080, 1920))
    for frame in frameList:
        out.write(frame)
    out.release()


#read video to frame list with gray images
def readVideo(fileName):
    frameList = []
    cap = cv2.VideoCapture(fileName)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frameList.append(gray)
        else:
            break
    cap.release()
    print "Video read successful. #frames = ", len(frameList)
    return frameList


#homography cost for frame i, j (assuming grayscale)
def findHomographyCost(frameList,i,j):
    im1 = frameList[i]
    im2 = frameList[j]

    [h1, w1] = im1.shape
    d = np.sqrt(h1**2 + w1**2) # The length of the diagonal in pixels
    tau_c = 0.1*d
    gamma = 0.5*d

    X = getHomography(im1, im2)
    if X is None:
        print "Warning: not enough matches found between frames ", i, " and " , j
        return gamma

    H, matches1, matches2 = X

    matches1 = np.squeeze(cv2.convertPointsToHomogeneous(matches1))
    matches2Hat = np.matmul(H, matches1.T)
    matches2Hat = np.squeeze(cv2.convertPointsFromHomogeneous(matches2Hat.T))
    difference = matches2.T - matches2Hat.T
    error = np.mean(np.linalg.norm(difference, axis=0))

    # Calculating C_o from the paper
    if error >= tau_c:
      error = gamma
    else:
      center1 = np.array([h1/2, w1/2, 1])
      center1hat = np.matmul(H, center1.T).reshape(3,1)
      center1hat = np.squeeze(cv2.convertPointsFromHomogeneous(center1hat.T))
      centerCost = np.linalg.norm(center1[0:2] - center1hat)

    return error

def findVelocityCost(i,j,speedupFactor):
  tau_s = 200 # Parameter used in paper, but results aren't super sensitive to changes
  diff = np.abs(((j-i) - speedupFactor)**2)
  return min(diff, tau_s)

def findAccelerationCost(h,i,j):
    tau_a = 200 # Parameter used in paper
    diff = np.abs(((j-i)-(i-h))**2)
    return min(diff,tau_a)


def getHomography(im1, im2):
    kp1, des1 = getFeatures(im1)
    kp2, des2 = getFeatures(im2)
    src_pts, dst_pts = matchFeatures(im1, kp1, des1, im2, kp2, des2, matcher='flann')
    if src_pts is None:
        return None
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    mask = np.array(mask, dtype=bool)
    src_pts = np.array(src_pts)
    dst_pts = np.array(dst_pts)
    src_pts = src_pts[mask]
    dst_pts = dst_pts[mask]
    return M, src_pts, dst_pts

# calculates ORB keypoints and descriptors
def getFeatures(im):
    orb = cv2.ORB_create()
    kp = orb.detect(im, None)
    kp, des = orb.compute(im, kp)
    return kp, des

#matches features with option of flann or brute force
#  https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
def matchFeatures(im1, kp1, des1, im2, kp2, des2, matcher='flann', minMatchCount=10):
    if matcher == 'flann':
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                           table_number = 6, # 12
                           key_size = 12,     # 20
                           multi_probe_level = 1) #2
        search_params = dict(checks=100)
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        # ratio test (m and n are closest, 2nd closest matches?)
        good = []
        for match in matches:
            if len(match) == 2:
                m = match[0]
                n = match[1]
                if m.distance < 0.7*n.distance:
                    good.append(m)
        if len(good) > minMatchCount:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            return src_pts, dst_pts
        else:
            return None, None


    """
    if matcher == 'bf':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x:x.distance )
        matched = None
        draw_params = dict(matchColor=(0,255,0),
                           singlePointColor=(255,0,0),
                           flags = 0)
        matched = cv2.drawMatches(first,kp1,second,kp2,matches[:20], None,                         **draw_params)
        cv2.imwrite('brute_force_matches.png', matched)
    """


def main():
    frameList = readVideo('testVid.avi')
    costMatrix = np.zeros((len(frameList),len(frameList)))
    traceBack = np.zeros((len(frameList), len(frameList)))
    print costMatrix.shape
    speedupFactor = 4 # Want to speed up the video by a factor of 4
    w = 2*speedupFactor

    g = w
    lambda_s = 200 # Parameter weight for velocity cost
    lambda_a = 80 # Parameter for acceleration cost
    
    for i in range(0, g):
        for j in range(i+1, i+w):
            C_m = findHomographyCost(frameList,i,j)
            C_s = findVelocityCost(i, j, speedupFactor)
            costMatrix[i,j] = C_m + lambda_s * C_s

    for i in range(g, len(frameList)):
        for j in range(i+1, min(i+w, len(frameList))):
            C_m = findHomographyCost(frameList, i, j)
            C_s = findVelocityCost(i, j, speedupFactor)
            lambda_s = 200
             
            c = C_m + lambda_s * C_s

            # Could make this faster potentially by not using a for loop 
            D_vi = costMatrix[max(0,i-w+1):i-1, i] 
            C_a = [lambda_a * findAccelerationCost(k, i, j) for k in range(max(0, i-w+1), i-1)]
            D_vi = D_vi + C_a
            test = c + min(D_vi)

            index = len(D_vi) - np.argmin(D_vi)
            print index
            costMatrix[i,j] = c + min([costMatrix[i-k,i] +  lambda_a * findAccelerationCost(i-k, i, j) for k in range(1,w)])


    costmatrix = costmatrix * 255/np.amax(costMatrix)
    cv2.imwrite('costMatrix.png', costMatrix)

if __name__ == '__main__':
    main()
