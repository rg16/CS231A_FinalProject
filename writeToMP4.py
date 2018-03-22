import utils

def main():
  #frameList = utils.readVideo('canon_selfie.mov')
  #utils.writeVideo('canon_walk.mp4', frameList)
  utils.makeSBS('canon_walk.mp4', 'canon_walk_naive.mp4', 'canon_walk_SBS.mp4')

if __name__ == '__main__':
  main()
