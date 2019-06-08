import numpy as np
import cv2
# from PIL import Image
# import matplotlib.pyplot as plt


def blocks(windowsize_row, windowsize_col, array):  # returns an array with the image divided into blocks
    array_blocks = []
    # divide the window :)
    for k in range(0, array.shape[0] - windowsize_row+1, windowsize_row):  # for each row of length(n))
        for c in range(0, array.shape[1] - windowsize_col+1, windowsize_col):  # we get n columns of length(n
            window_block = array[k:k+windowsize_row, c:c+windowsize_col].astype('int16')  # to create a block
            array_blocks.append(window_block)
    array_blocks = np.array(array_blocks)
    return array_blocks


# def rnd_to_nxt_mul(size, i):
#     return ((i - 1) // size + 1) * size


# συνάρτηση η οποία, αν τα pixels δεν φτάνουν, τα γεμίζει με μαύρα
def add_black_pixels(image):
    tmp_image = []
    height = image.shape[0]
    width = image.shape[1]
    add_black_pixels = np.array(black_pixel() * (((width-1 // 16 + 1) * 16) - width))
    # print(add_black_pixels)
    for row in image:
        # print(row)
        tmp_row = np.append(row, add_black_pixels, axis=0)
        tmp_image.append(tmp_row)
    add_black_line = np.array(black_pixel() * ((width-1 // 16+1) * 16))
    for j in range(((width-1 // 16 + 1) * 16) - height):
        tmp_image.append(add_black_line)
    return np.array(tmp_image)


def black_pixel():
    return [0]


# γίνεται ιεραρχική αναζήτηση για πιο γρήγορα. Κανονικά είναι πάρα πολύ αργό
def hierarchical_search(array):
    array = np.array(array)  # ensure that we are handling a numpy array
    x, y = array.shape  # it's an image, not a video. only 2 coordinates
    array2 = []  # initialize the return array
    for i in range(0, x, 2):
        for j in range(0, y, 2):
            try:
                array2.append(array[i][j])  # make a single dimensional array
            except:
                continue  # in the case of an array with even number of pixels
    array2 = np.array(array2)  # convert the array to numpy array
    array2 = np.reshape(array2, (int(x/2), int(y/2)))  # reshape the array to match the dimensions we want
    return array2


def find_motion(array):
    array = np.array(array)  # ensure that we are handling a numpy array
    x, y = array.shape
    number_of_zeroes = x*y - np.count_nonzero(array)  # count the number of zeroes in the given image
    if number_of_zeroes >= 0.9 * x * y:  # if the given array is at least 80% of zeroes
        # print("array almost full of zeroes")
        return 0
    else:
        return 1


def find_differences(source, target):
    source = np.array(source)
    target = np.array(target)
    global hierimg1
    global hierimg2
    # we create 2 new images based on the original with lower resolutions to start searching on smaller blocks for faster discalification of blocks
    for i in range(2):
        hierimg1 = hierimg1 + [hierarchical_search(source)]
        source = hierarchical_search(source)
        hierimg2 = hierimg2 + [hierarchical_search(target)]
        target = hierarchical_search(target)


def image_recreate(x, y, img2):
    r = 1
    # we add lines of blocks on top of each other to create the image
    for i in range(x):
        # necessary initialisation to get dimensions
        output = np.array(img2[i * y])
        #  we add blocks next to each other to create lines of blocks
        for j in range(y-1):
            output = np.concatenate((output, img2[r]), axis=1)
            r += 1
        r += 1
        #  necessary initialisation to get dimensions
        if i == 0:
            showim = output
        else:
            showim = np.concatenate((showim, output), axis=0)
    return showim  # return reconstructed image


def initial(q):
    global hierimg1, hierimg2
    hierimg1 = [images[0]]
    hierimg2 = [images[q]]
    find_differences(images[0], images[q])  # we take 2 images to compare by reducing their resolution for quicker calculations
    # η img1 είναι η πάλια στην οποία ψάχνουμε να βρούμε σε ποια blocks έχεις κίνηση, και η img2 είναι η καινούργια χωρίς κίνηση
    img1 = blocks(4, 4, hierimg1[2])  # we divide the smallest image into 4x4 blocks
    img2 = blocks(4, 4, hierimg2[2])  # we divide the smallest image into 4x4 blocks
    blocks_move = []  # which blocks show movement
    for i in range(len(img1)):
        if find_motion(img1[i]-img2[i]):
            blocks_move = blocks_move + [i]  # if movement then append the index of the block
    return img1, img2, hierimg1, hierimg2, blocks_move


def move_in_blocks(blocks_move, hierimg1, hierimg2):
    size_blocks = [8, 16]
    for k in range(2):
        img1 = blocks(size_blocks[k], size_blocks[k], hierimg1[1-k])  # we divide the smallest image into lxl blocks // l einai (L) el
        img2 = blocks(size_blocks[k], size_blocks[k], hierimg2[1-k])  # we divide the smallest image into lxl blocks
        blocks_no_move = []
        for i in range(len(blocks_move)):  # we will only check the blocks we saw movement in the previous hierarchical step
            if find_motion(img1[blocks_move[i]]-img2[blocks_move[i]]):
                continue  # if there is still movement check next block
            else:
                blocks_no_move = blocks_no_move + [i]  # if there is no movement then save the index of the block that will later be poppes from the list
        blocks_no_move = [x for x in blocks_move if x not in blocks_no_move]  # pop unwanted blocks
    return img1, img2, blocks_no_move


def main(images):
    # if the resolution cannot be divided perfectly by 16x16 blocks we add black pixels
    if images[0].shape[0] % 16 != 0 or images[0].shape[1] % 16 != 0:
        for i in range(len(images)):
            images[i] = add_black_pixels(images[i])
    images = np.array(images)
    for q in range(1, len(images)):
        # initialize arrays to get the required dimensions
        img1, img2, hierimg1, hierimg2, blocks_move = initial(q)
        # calculate whick blocks are to be replaced
        img1, img2, blocks_move = move_in_blocks(blocks_move, hierimg1, hierimg2)

        # replace movement-blocks with backround-blocks
        for i in range(len(blocks_move)):
            img2[blocks_move[i]] = img1[blocks_move[i]]

        y = int(images[0].shape[1]/16)
        x = int(images[0].shape[0]/16)
        # reconstuct image based on the original resolution
        images[q] = image_recreate(x, y, img2)
        print('frame ' + str(q) + ' Done')
    return images


# necessary initializations
hierimg1 = []
hierimg2 = []
img1 = []
img2 = []
video = cv2.VideoCapture('video.mp4')
success, image = video.read()
images = []
first = []
success = True
number_of_frames = 0


# read the video, convert each frame to grayscale. save the background as an individual frame.
while success:  # store the video as list
    try:
        success, image = video.read()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert image to grayscale
        if number_of_frames == 0:
            first.append(image)  # first frame. only the background of the video
        images.append(image.astype('int16'))  # see "continue" at line 16
        number_of_frames = number_of_frames + 1
    except :
        continue  # it stores the dtype for some reason and we don't want that


# get the final images
images = np.array(main(images), dtype=np.uint8)
# start a video file to write on
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('edit_video.avi', fourcc, 30.0, (images[0].shape[1], images[0].shape[0]), False)
print('Creating video file...')
# write all the images in the video file
for i in range(len(images)):
    out.write(images[i])
print('The video "output.avi" ha been added/updated in your run folder.')
# release
out.release()
cv2.destroyAllWindows()
