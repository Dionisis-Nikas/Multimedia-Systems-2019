from scipy.stats import entropy
import numpy as np
import cv2

#####################################################################################################

def macroblocksMaker(row_size, column_size, array): #returns an array with the image divided into macroblocks
    blocks=[]
    for r in range(0,array.shape[0] - row_size+1, row_size):
        for c in range(0,array.shape[1] - column_size+1, column_size):
            macroblock = array[r:r+row_size,c:c+column_size].astype('int16')
            blocks.append(macroblock)
    blocks=np.array(blocks)
    return blocks

#####################################################################################################

def SAD(im1,im2,block):
    i = block
    blocks = []
    differentials = []
    differentials.append(sumAbsoluteDifferences(im2[i],im1[i]))
    blocks.append(i)
    if i+1<=im1.shape[1]:
        differentials.append(sumAbsoluteDifferences(im2[i],im1[i+1]))
        blocks.append(i+1)
    if i-1 >= 0:
        differentials.append(sumAbsoluteDifferences(im2[i],im1[i-1]))
        blocks.append(i - 1)
    if i + im1.shape[1] <= im1.shape[1]*im1.shape[1]:
        differentials.append(sumAbsoluteDifferences(im2[i], im1[i+ im1.shape[1]]))
        blocks.append(i + im1.shape[1])
    if i + im1.shape[1] + 1 <= im1.shape[1]*im1.shape[1]:
        differentials.append(sumAbsoluteDifferences(im2[i], im1[i+ im1.shape[1] +1]))
        blocks.append(i + im1.shape[1])
    if i + im1.shape[1] - 1 <= im1.shape[1]*im1.shape[1]:
        differentials.append(sumAbsoluteDifferences(im2[i], im1[i+ im1.shape[1]-1]))
        blocks.append(i + im1.shape[1])
    if i - im1.shape[1] >= 0:
        differentials.append(sumAbsoluteDifferences(im2[i], im1[i- im1.shape[1]]))
        blocks.append(i - im1.shape[1])
    if i - im1.shape[1] + 1 >= 0:
        differentials.append(sumAbsoluteDifferences(im2[i], im1[i- im1.shape[1]+1]))
        blocks.append(i - im1.shape[1])
    if i - im1.shape[1] - 1 >= 0:
        differentials.append(sumAbsoluteDifferences(im2[i], im1[i- im1.shape[1]-1]))
        blocks.append(i - im1.shape[1])
    index = differentials.index(min(differentials))
    min_block = blocks[index]


    return min_block

def sumAbsoluteDifferences(macro1, macro2):
    sad = 0
    n = macro1.shape[0]
    for i in range(n):
        for j in range(n):
            sad += abs(int(macro1[i, j]) - int(macro2[i, j]))
    return sad

#####################################################################################################


def round_number(size, i):
    return ((i - 1) // size + 1) * size

def fillWithBlackLines(image):
    filled_image = []
    height = image.shape[0]
    width = image.shape[1]
    black_pixels = np.array(black_pixel() * (round_number(16, width) - width))
    for row in image:
        filled_row = np.append(row, black_pixels, axis = 0)
        filled_image.append(filled_row)
    black_line = np.array(black_pixel() * round_number(16, width))
    for j in range(round_number(16, height) - height):
        filled_image.append(black_line)
    return np.array(filled_image)

def black_pixel():
    return [0]



#####################################################################################################

def level_analysis(source, target):
    source=np.array(source)
    target=np.array(target)
    global hierimg1
    global hierimg2
    #we create 2 new images based on the original with lower resolutions to start searching on smaller blocks for faster discalification of blocks
    for i in range(2):
        hierimg1 = hierimg1 + [hierarchicalDiv(source)]
        source =hierarchicalDiv(source)
        hierimg2 = hierimg2 + [hierarchicalDiv(target)]
        target =hierarchicalDiv(target)

def hierarchicalDiv(array):
    array=np.array(array)
    x, y = array.shape
    hierarchicalImage=[]
    for i in range(0, x, 2):
        for j in range(0, y, 2):
            try:
                hierarchicalImage.append(array[i][j])
            except:
                continue # even number of pixels in an array
    hierarchicalImage=np.array(hierarchicalImage)
    hierarchicalImage=np.reshape(hierarchicalImage, (int(x/2), int(y/2))) #dimensions we want
    return(hierarchicalImage)

#####################################################################################################



#####################################################################################################

def improveDeeperLevels(movement_blocks,hierimg1,hierimg2):
    l = [8,16]
    for k in range(2):
        eikona1 = macroblocksMaker(l[k],l[k],hierimg1[1-k])#we divide the smallest image into lxl blocks
        eikona2 = macroblocksMaker(l[k],l[k],hierimg2[1-k])#we divide the smallest image into lxl blocks
        to_be_checked = []
        for i in range(len(movement_blocks)):#we will only check the blocks we saw movement in the previous hierarchical step
            if estimateMotion(eikona1[movement_blocks[i]]-eikona2[movement_blocks[i]]):
                continue #if there is still movement check next block
            else:
                to_be_checked = to_be_checked + [i]#if there is no movement then save the index of the block that will later be poppes from the list
        movement_blocks = [x for x in movement_blocks if x not in to_be_checked]#pop unwanted blocks
    return(eikona1, eikona2 ,movement_blocks)

def estimateMotion(array):
    array=np.array(array)
    x, y = array.shape
    num_of_zeros = x*y - np.count_nonzero(array) #count the number of zeroes in the given image
    if (num_of_zeros >= 0.9 * x * y): #if the given array is at least 80% of zeroes then no motion is detected
        return(0)
    else:
        return(1)

#####################################################################################################

#####################################################################################################

def ImageReconstruction(x, y,eikona2):
    r=1
    #we add lines of blocks on top of each other to create the image
    for i in range(x):
        #necessary initialisation to get dimensions
        output = np.array(eikona2[i*(y)])
        #we add blocks next to eaxh other to create lines of blocks
        for j in range(y-1):
            output = np.concatenate((output,eikona2[r]), axis=1)
            r= r +1
        r = r+1
        #necessary initialisation to get dimensions
        if(i==0):
            showim = output
        else:
            showim = np.concatenate((showim,output),axis=0)
    return(showim)#return reconstructed image

#####################################################################################################


def entropy1(labels, base=None):
    value, counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)

cap = cv2.VideoCapture('../original_videos/movie.avi')
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

buf = []
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

fps = 30
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
out = cv2.VideoWriter('../final_videos/8.17/8.17b.avi', fourcc , fps , size ,False)
fc = 0
ret, r_frame = cap.read()
r_frame = cv2.cvtColor(r_frame, cv2.COLOR_BGR2GRAY)
global hierimg1
global hierimg2
while cap.isOpened():
    ret, c_frame = cap.read()
    if not ret:
        break
    if (c_frame.shape[0] % 16 != 0 or c_frame.shape[1] % 16 != 0):
            c_frame = fillWithBlackLines(c_frame)
    if (r_frame.shape[0] % 16 != 0 or r_frame.shape[1] % 16 != 0):
            r_frame = fillWithBlackLines(r_frame)
    c_frame = cv2.cvtColor(c_frame, cv2.COLOR_BGR2GRAY)
    hierimg1 = [r_frame]
    hierimg2 = [c_frame]
    level_analysis(r_frame,c_frame)
    eikona1 = macroblocksMaker(4, 4, hierimg1[2])
    eikona2 = macroblocksMaker(4, 4, hierimg2[2])
    movement_blocks = []  # whick blocks show movement
    for i in range(len(eikona1)):
        if estimateMotion(eikona1[i] - eikona2[i]):
            movement_blocks = movement_blocks + [i]  # if movement then append the index of the block
    eikona1, eikona2, movement_blocks = improveDeeperLevels(movement_blocks, hierimg1, hierimg2)
    eikona3 = eikona2
    for i in range(len(movement_blocks)):
        predicted = SAD(eikona1,eikona2,movement_blocks[i])
        eikona1[movement_blocks[i]] = eikona1[predicted]




    y = int(c_frame.shape[1] / 16)
    x = int(c_frame.shape[0] / 16)


    eikona4 = ImageReconstruction(x,y,np.uint8(np.subtract(eikona2,eikona1)))



    r_frame = c_frame
    out.write(eikona4)
    print("Frame "+str(fc +1)+" complete")
    try:
        image = eikona4

        buf.append(image.astype('int16'))
        fc = fc + 1
    except :
        continue
    r_frame = c_frame

print('The video has been created.'+ '\n entropy = '+ str(entropy1(buf)))

cap.release()
out.release()
cv2.destroyAllWindows()