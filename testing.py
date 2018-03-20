import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import utils
import video_stabilization as vs

# Global parameters for tuning
# ---------------------------
SPEEDUP_FACTOR = 8
COST_CHAINING = True
LAMBDA_S = 200
LAMBDA_A = 80

def findConsecutiveCenteringCosts(frameList):
    numFrames = len(frameList)
    centeringCosts = np.zeros((numFrames-1,))
    [h1, w1] = frameList[0].shape
    d = np.sqrt(h1**2 + w1**2) # The length of the diagonal in pixels
    tau_c = 0.1*d
    gamma = 0.5*d
    for i in range(numFrames-1):
        print 'progress; ', float(i)/len(frameList), '%'
        im1 = frameList[i]
        im2 = frameList[i+1]
        H, matches1, matches2 = getHomography(im1, im2)

        if H is None:
            centeringCosts[i] = gamma
        else:
            center1 = np.array([h1/2, w1/2, 1])
            center1hat = np.matmul(H, center1.T).reshape(3,1)
            center1hat = np.squeeze(cv2.convertPointsFromHomogeneous(center1hat.T))
            centerCost = np.linalg.norm(center1[0:2] - center1hat)
            centeringCosts[i] = centerCost
    return centeringCosts


#homography cost for frame i, j (assuming grayscale)
def findHomographyCost(frameList,i,j):
    im1 = frameList[i]
    im2 = frameList[j]

    [h1, w1] = im1.shape
    d = np.sqrt(h1**2 + w1**2) # The length of the diagonal in pixels
    tau_c = 0.1*d
    gamma = 0.5*d

    H, matches1, matches2 = getHomography(im1, im2)
    if H is None:
        print "Warning: not enough matches found between frames ", i, " and " , j
        return gamma

    matches1 = np.squeeze(cv2.convertPointsToHomogeneous(matches1))
    matches2Hat = np.matmul(H, matches1.T)
    matches2Hat = np.squeeze(cv2.convertPointsFromHomogeneous(matches2Hat.T))
    difference = matches2.T - matches2Hat.T
    norms = np.linalg.norm(difference, axis=0)
    error = np.mean(norms)

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

def findBlurCost(frameList, j):
# Only calculating the blurriness of the next frame
    im1 = frameList[i]
    im2 = frameList[j]
    blurriness = cv2.Laplacian(im1, im2).var() 
    return blurriness

def getHomography(im1, im2):
    kp1, des1 = getFeatures(im1)
    kp2, des2 = getFeatures(im2)
    src_pts = None
    dst_pts = None
    if kp1 and kp2:
        src_pts, dst_pts = matchFeatures(im1, kp1, des1, im2, kp2, des2, matcher='flann')
    if src_pts is None:
        return None, None, None
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
    if kp is None or len(kp) < 1:
        return None, None
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

def makeSBS(f1, f2):
    newFrames = []
    for i, frame in enumerate(f1):
        if i == len(f2):
            break
        frame2 = f2[i]
        newFrame = np.concatenate((frame, frame2))
        newFrames.append(newFrame)
    return newFrames

'''
COMPUTE_OPTIMUM_FRAMES : Calculates best frames for hyperlapse of given speed
Arguments:
    frameList - a list of ndarrays, where each ndarray is a frame in the 
	input video. 

    speedupFactor - determines how many times faster the hyperlapse video
        should be than the input video. 

Returns:
    newFrames - A list of frames containing the hyperlapse video using 
         the optimal frame selection algorithm outlined in the paper
         "Real Time Hyperlapse Creation via Optimal Frame Selection" by
         Joshi et al. 

    costMatrix - The matrix encoding the cost of traversing from frame i in
         frameList to frame j. 
'''
def compute_optimum_frames(frameList, speedupFactor):
  
    numFrames = len(frameList)
    costMatrix = np.zeros((numFrames, numFrames))
    traceBack = np.zeros((numFrames, numFrames))

    w = 2*speedupFactor

    g = 4

    centeringCosts = None
    cumSumCosts = None
    if COST_CHAINING:
        centeringCosts = findConsecutiveCenteringCosts(frameList)
        cumSumCosts = np.cumsum(centeringCosts)
        cumSumCosts = np.insert(cumSumCosts, 0, 0.0, axis=0) #add a zero to the beginning of cumsum to make indexing easier

    homographyCostMat = np.zeros((len(frameList),len(frameList)))

    for i in range(0, g):
#        print 'progress; ', float(i)/len(frameList), '%'
        for j in range(i+1, i+w):
            C_m = findHomographyCost(frameList,i,j)
            C_s = findVelocityCost(i, j, speedupFactor)
            costMatrix[i,j] = C_m + LAMBDA_S * C_s
            homographyCostMat[i,j] = C_m

    for i in range(g, len(frameList)):
#        print 'progress; ', float(i)/len(frameList), '%'
        for j in range(i+1, min(i+w, len(frameList))):
            C_m = 0

            if (COST_CHAINING):
                C_m = cumSumCosts[j] - cumSumCosts[i]
            else:
                C_m = findHomographyCost(frameList, i, j)

            C_s = findVelocityCost(i, j, speedupFactor)
            c = C_m + LAMBDA_S * C_s
            # Could make this faster potentially by not using a for loop
            D_vi = costMatrix[max(0,i-w+1):i-1, i]
            C_a = [LAMBDA_A * findAccelerationCost(k, i, j) for k in range(max(0, i-w+1), i-1)]
            D_vi = D_vi + C_a
            costMatrix[i,j] = c + min(D_vi)

            index = i - len(D_vi) + np.argmin(D_vi)
            traceBack[i,j] = index
            homographyCostMat[i,j] = C_m

    min_val = np.inf
    s,d = (numFrames-g, numFrames-g+1)
    for i in range(numFrames-g, numFrames):
      for j in range(i+1, i+w):
        if j < numFrames:
          if costMatrix[i,j] < min_val:
            min_val = costMatrix[i,j]
            s,d = (i, j)

    p = [d]
    while s > g:
      p = [s] + p
      b = traceBack[int(s), int(d)]
      d = s
      s = b

    p = [s] + p
    # Now we have the list of frames

    newFrames = []
    for i in range(0,len(p)):
      newFrames.append(frameList[int(p[i])])
  
    return newFrames, costMatrix

def main():

    ### SET INPUT/OUTPUT LOCATIONS ###

    inputFile = 'Input/andreamble.mov'
    outputDirectory = 'Output/'

    ### -------------------------- ###

    frameList = utils.readVideo(inputFile)
    newFrames, costMatrix = compute_optimum_frames(frameList, 4)

#    newFrames = vs.stabilize(newFrames) #imported video_stabilization.py as vs
    utils.writeVideo(outputDirectory + 'output.mp4', newFrames)

    naiveFrames = frameList[0:len(frameList):SPEEDUP_FACTOR]
    utils.writeVideo(outputDirectory + 'naive_output.mp4', naiveFrames)
    
    newStdDev = utils.computeStdDev(newFrames, 90, 5)
    naiveStdDev = utils.computeStdDev(naiveFrames, 90, 5)

    cv2.imwrite('new_stddev.png', newStdDev)
    cv2.imwrite('naive_stddev.png', naiveStdDev)

    sbsFrames = makeSBS(newFrames, naiveFrames)
    utils.writeVideo(outputDirectory + 'sbs_output.mp4', sbsFrames)
 
#    scipy.io.savemat('costMatrix.mat', dict(costMatrix=costMatrix, homographyCostMat=homographyCostMat))
#    scipy.io.savemat('costMatrix.mat', dict(costMatrix=costMatrix, homographyCostMat=homographyCostMat))

if __name__ == '__main__':
    main()
