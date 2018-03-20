import argparse
import utils
import csv
import cv2

def parseArguments():

  parser = argparse.ArgumentParser()
  parser.add_argument('inputVideo', help='The video to be hyperlapsed')

  return parser.parse_args()


def getFrameList(input, p):

  frameList = []
  cap = cv2.VideoCapture(input)
  count = 0
  p_count = 0
  while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        if count == p[p_count]:
          watermark = int(0.1 * frame.shape[0])
          frame = frame[:-watermark, :, :]
          frameList.append(frame)
          p_count = p_count + 1
        count = count + 1
    else:
      break
  cap.release()
  return frameList

def main():

  args = parseArguments()

  p = []
  with open('Output/andreamble/p.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        for i in row:
            p.append(int(float(i)))

  frameList = getFrameList(args.inputVideo, p)

  utils.writeVideo('Output/andreamble/colortest.mp4', frameList)


if __name__ == '__main__':
  main()
