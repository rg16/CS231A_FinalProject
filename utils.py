import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

def writeVideo(outputFile, frameList):
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')
  out = cv2.VideoWriter(outputFile, fourcc, 20.0, (1080, 1920))
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
  print "Video read successful. #frames = ", len(frameList)
  print "Frame size = ", frameList[0].shape
  return frameList
