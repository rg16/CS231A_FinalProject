import cv2
import numpy as np
from collections import namedtuple
import utils

TransformParam = namedtuple("TransformParam", "dx dy da")
Trajectory = namedtuple("Trajectory", "x y a") # a refers to angle

SMOOTHING_RADIUS = 30
HORIZONTAL_BORDER_CROP = 20

def stabilize(frameList):

  prevFrame = frameList[0]
  lastT = None

  prev2curr_transform = []
  for i in range(1, len(frameList)):

     currFrame = frameList[i]
     prev_corners = cv2.goodFeaturesToTrack(prevFrame, 200, 0.01, 30)
     curr_corners = None
     curr_corners, status, err = cv2.calcOpticalFlowPyrLK(prevFrame, currFrame, prev_corners, curr_corners)

     prev_corners2 = []
     curr_corners2 = []

     for i in range(0, len(status)):
       if status[i]:
         prev_corners2.append(prev_corners[i])
         curr_corners2.append(curr_corners[i])

     prev_corners2 = np.asarray(prev_corners2)
     curr_corners2 = np.asarray(curr_corners2)

     T = None
     try:
       T = cv2.estimateRigidTransform(prev_corners2, curr_corners2, False)
     except cv2.error:
       print 'Error calculating T'
     if T is None:

       if lastT is None:
         continue
       T = lastT

     else:
       lastT = T

     dx = T[0,2]
     dy = T[1,2]
     da = np.arctan2(T[1,0], T[0,0])

     prev2curr_transform.append(TransformParam(dx, dy, da))

     prevFrame = currFrame

  a = 0
  x = 0
  y = 0

  trajectory = []

  for i in range(0, len(prev2curr_transform)):

    x = x + prev2curr_transform[i].dx
    y = y + prev2curr_transform[i].dy
    a = a + prev2curr_transform[i].da

    trajectory.append(Trajectory(x,y,a))

  smoothed_trajectory = []

  for i in range(0, len(trajectory)):

    sumx = 0
    sumy = 0
    suma = 0

    count = 0

    for j in range(-SMOOTHING_RADIUS, SMOOTHING_RADIUS+1):

      if(i+j >= 0 and i+j < len(trajectory)):

        sumx = sumx + trajectory[i+j].x
        sumy = sumy + trajectory[i+j].y
        suma = suma + trajectory[i+j].a

        count = count + 1

    avga = suma / count
    avgx = sumx / count
    avgy = sumy / count

    smoothed_trajectory.append(Trajectory(avgx, avgy, avga))


  new_prev2curr_transform = []

  a = 0
  x = 0
  y = 0

  for i in range(0, len(prev2curr_transform)):

    x = x + prev2curr_transform[i].dx
    y = y + prev2curr_transform[i].dy
    a = a + prev2curr_transform[i].da

    diffx = smoothed_trajectory[i].x - x
    diffy = smoothed_trajectory[i].y - y
    diffa = smoothed_trajectory[i].a - a

    dx = prev2curr_transform[i].dx + diffx
    dy = prev2curr_transform[i].dy + diffy
    da = prev2curr_transform[i].da + diffa

    new_prev2curr_transform.append(TransformParam(dx, dy, da))


  vert_border = HORIZONTAL_BORDER_CROP * prevFrame.shape[0] / prevFrame.shape[1]
  newFrames = []
  T = np.zeros((2,3))

  for k in range(0, len(frameList)-1):

    curr = frameList[k]
    T[0,0] =  np.cos(new_prev2curr_transform[k].da)
    T[0,1] = -np.sin(new_prev2curr_transform[k].da)
    T[1,0] =  np.sin(new_prev2curr_transform[k].da)
    T[1,1] =  np.cos(new_prev2curr_transform[k].da)

    T[0,2] = new_prev2curr_transform[k].dx
    T[1,2] = new_prev2curr_transform[k].dy

    curr2 = np.zeros((curr.shape))
    curr2 = cv2.warpAffine(curr, T, (curr.shape[1], curr.shape[0]), curr2)
    curr2 = curr2[vert_border:curr2.shape[0]-vert_border, \
                  HORIZONTAL_BORDER_CROP:curr2.shape[1]-HORIZONTAL_BORDER_CROP]


    curr2 = cv2.resize(curr2, (curr.shape[1], curr.shape[0]))

    cv2.imwrite('test.png', curr2)
#    print curr2.shape
#    raise Exception('Not Implemented Error')

#    canvas = np.zeros((curr.shape[0], curr.shape[1]*2 + 10))

#    canvas[:, 0:curr.shape[1]] = curr
#    canvas[:, curr.shape[1]+10:curr.shape[1]*2+10] = curr2

    newFrames.append(curr2)

  return newFrames

if __name__ == "__main__":
  main()
