import cv2
from collections import namedtuple


TransformParam = namedtuple("TransformParam", "dx dy da")
Trajectory = namedtuple("Trajectory", "x y a") # a refers to angle


def main():

  frameList = readVideo('testVid.avi')
  

if __name__ == "__main__":
  main()
