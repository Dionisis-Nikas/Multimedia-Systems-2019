import cv2
import numpy as np
import os

####################################################################
# Neccesary initializations and video writer/reader configurations #
####################################################################

video = cv2.VideoCapture("../original_videos/movie2.avi")
count = 0
frames = []
size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

fps = 30
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
out = cv2.VideoWriter()
success = out.open('../final_videos/6.17/compressed_movie.avi',fourcc,fps,size)

#################################
# Read the Video frame by frame #
#################################
while success:
    try:
          success, image = video.read()
          frames.append(image.astype("int32"))
          count += 1
    except:
        continue


###############################################################################
# Convert list of images to numpy array for easier calculations with matrices #
###############################################################################
frames = np.array(frames)
x,y,z,w = frames.shape
print('Enter quantization parameter:')
QP = int(input())
while not(int(QP)):
    QP = int(input())
    print('Enter quantization parameter:')


for i in range(1, x):

    # Array of differences
    frames[i,:,:,:] = frames[i,:,:,] - frames[i-1,:,:,:]

    #Round the decimals to the nearest integer then quantize it with the quantization parameter from input
    frames[i,:,:,:] = np.rint(np.divide(frames[i,:,:,:],QP))

#Reconstruct the frames to output the compressed video
for i in range (1, x-1):
    frames[i,:,:,:] = frames[i,:,:,:] + frames[i+1,:,:,:]
    out.write((frames[i,:,:,:]).astype('uint8'))

cv2.destroyAllWindows()
out.release()
print(count)
before= os.path.getsize('../original_videos/movie2.avi')
after = os.path.getsize('../final_videos/6.17/compressed_movie.avi')
ratio = float(before/after)
print("Video Compression is complete")
print("The compression ratio is: ", ratio)
