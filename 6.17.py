import cv2
import numpy as np
import os
from matplotlib.image import imread

#################################
# Read the Video frame by frame #
#################################

video = cv2.VideoCapture("./original/outpy.avi")
count = 0
frames = []
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

fps = 25
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
out = cv2.VideoWriter()
success = out.open('./compressed/compressed_movie.avi',fourcc,fps,size)
while success:
    try:
          success, image = video.read()
          frames.append(image.astype("int32"))
          count += 1
    except:
        continue

  ##print('Read a new frame: ', success, image)

frames = np.array(frames)
print(frames.shape)
x,y,z,w = frames.shape

QP = 2

for i in range(1, x):

    # Array of differences
    frames[i,:,:,:] = frames[i,:,:,] - frames[i-1,:,:,:]

    #Round any possible decimals to the nearest integer then quantize it
    frames[i,:,:,:] = np.rint(np.divide(frames[i,:,:,:],QP))

#Reconstruct the frames to output the compressed video
for i in range (1, x-1):
    frames[i,:,:,:] = frames[i,:,:,:] + frames[i+1,:,:,:]
    out.write((frames[i,:,:,:]).astype('uint8'))

cv2.destroyAllWindows()
out.release()

before= os.path.getsize('./original/movie.avi')
after = os.path.getsize('./compressed/compressed_movie.mp4')
ratio = float(before/after)
print("The compress ration is: ", ratio)
print("Your video is now in the same folder as the program!")