import cv2
import numpy as np

def writeVideo(outputFile, frameList):
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	out = cv2.VideoWriter('testOutput.mp4',fourcc, 20.0, (1080, 1920))
	for frame in frameList:
		out.write(frame)
	out.release()



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
	return framelist





def findHomographyCost(H, kp1, kp2):
	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

	return True
	




  orb = cv2.ORB_create()
  first = cv2.cvtColor(first, cv2.COLOR_RGB2GRAY)
  kp1 = orb.detect(first, None)
  kp1, des1 = orb.compute(first, kp1)
  img = None # Weird, but it complains when you don't specify outImage in drawKeypoints
  img = cv2.drawKeypoints(first, kp1, outImage=img, color=(0,255,0), flags=0)
  cv2.imwrite('test_features.png', img) 

  # Repeat for features in second frame

  second = cv2.cvtColor(second, cv2.COLOR_RGB2GRAY)
  kp2 = orb.detect(second, None)
  kp2, des2 = orb.compute(second, kp2)
  img2 = None
  img2 = cv2.drawKeypoints(second, kp2, outImage=img2, color=(0,255,0), flags=0)
  cv2.imwrite('test_features2.png',img2)

  # Now we have features in two images, we can perform feature matching
  # We will use a FLANN based Matcher

  FLANN_INDEX_LSH = 6
  index_params= dict(algorithm = FLANN_INDEX_LSH,
                     table_number = 6, # 12
                     key_size = 12,     # 20
                     multi_probe_level = 1) #2
  search_params = dict(checks=100)

  flann = cv2.FlannBasedMatcher(index_params,search_params)

  matches = flann.knnMatch(des1,des2,k=2)

  # A mask that will indicate which matches are 'good' matches
  matchesMask = [[0,0] for i in xrange(len(matches))]

  # Apply ratio test to filter out 'bad' matches
  for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
  
  draw_params = dict(matchColor=(0,255,0),
                     singlePointColor=(255,0,0),
                     matchesMask=matchesMask,
                     flags = 0)

  img3 = cv2.drawMatchesKnn(first, kp1, second, kp2, matches, None, **draw_params)
  cv2.imwrite('test_matches.png',img3)



def main():
	readVideo('testVid.avi')


if __name__ == '__main__':
  main()

