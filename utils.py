import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import skvideo.io

def writeVideo(outputFile, frameList):
  """
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(outputFile, fourcc, 20.0, (1080, 1920))
  for frame in frameList:
    out.write(frame)
  out.release()
  """
  print "writing video"
  outputdata = np.random.random(size=(5, 480, 680, 3)) * 255
  outputdata = outputdata.astype(np.uint8)

  writer = skvideo.io.FFmpegWriter("SKoutputvideo.mp4")
  for i in range(0, len(frameList)):
    writer.writeFrame(frameList[i])
  writer.close()

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
  print "Frame size = ", frameList[0].shape
  return frameList







