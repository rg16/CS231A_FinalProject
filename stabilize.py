import utils
import video_stabilization as vs


def main():
  
  frameList = utils.readVideo('longTest_osl.mov')
  frameList = vs.stabilize(frameList)
  
  utils.writeVideo('longTest_osl_stabilized.mov', frameList)

if __name__ == '__main__':
  main()
