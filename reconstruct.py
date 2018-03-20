import argparse
import utils
import csv
import cv2

def parseArguments():
  
  parser = argparse.ArgumentParser()
  parser.add_argument('inputVideo', help='The video to be hyperlapsed')

  return parser.parse_args()


def getFrameList(input):
  
  frameList = []
  cap = cv2.VideoCapture(input)
  while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
      watermark = int(0.1 * frame.shape[0])
      frame = frame[:-watermark, :, :]
      frameList.append(frame)
    else:
      break
  cap.release()
  return frameList

def main():

  args = parseArguments()

  print args
  frameList = getFrameList(args.inputVideo)
  
  p = []
  with open('Output/p.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
      print row
  
  newFrames = []
  for i in range(0, len(p)):
    frame = frameList[p[i]]
    newFrames.append(frame)

  utils.writeVideo('test.mp4', newFrames)
  

if __name__ == '__main__':
  main()
