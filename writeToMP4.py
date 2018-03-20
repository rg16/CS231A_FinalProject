import utils

def main():
  frameList = utils.readVideo('naive_longtest.mov')
  utils.writeVideo('naive_longtest.mp4', frameList)

if __name__ == '__main__':
  main()
