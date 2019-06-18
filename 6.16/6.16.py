from PIL import Image
import numpy as np


# IMAGE AFTER QUANTIZATION (DIVISION)
#                4     pixel_array
def quantization(int1, quant_array):
    for i in range(height):
        quant_array[i][:] = [x / int1 for x in quant_array[i]]         # division all elements in array for quantization
    print('\n\n\n\n  After the quantization\n', quant_array)         # print the quantum array
    im = Image.fromarray(quant_array)   # retrieve data from quantum array back to image

    im.show()
    return quant_array


# RECOVER IMAGE FROM QUANTIZATION
#
def de_quantization(int2, recover_img):
    for i in range(height):
        recover_img[i][:] = [x * int2 for x in recover_img[i]]

    im = Image.fromarray(recover_img)

    #im.show()
    return recover_img


def encoder(enco_array):  # encode array
    encoded = ' ' + str(enco_array[0, 0][0]) + ';' + str(enco_array[0, 0][1]) + ';' + str(enco_array[0, 0][2]) + ''
    i = 0
    j = 1
    while i < height:
        if i == 0:
            before = enco_array[i, j-1]  # previous pixel
        else:
            before = enco_array[i-1, j]

        while j < width:
            present = enco_array[i, j]  # current pixel
            if not (before == present).all():
                encoded += '-'+str(i) + ';' + str(j) + '|' + str(present[0]) + ';' + str(present[1]) + ';' + str(present[2])
            before = present
            j += 1
        j = 0
        i += 1
    encoded += '-' + str(height - 1) + ';' + str(width - 1)
    return encoded


def decoder(encoded):
    row_end = False
    decode = np.zeros((height, width, 3), dtype=np.uint8)
    encoded_array = encoded.split('|')
    present = 0
    pre_col = 0  # previous column
    for i in encoded_array:
        i = i.split('-')
        range = i[1].split(';')  # get the range of the RGB value
        col = int(range[0])  # get the column that the next value starts
        end = int(range[1])  # get the row index of the next value- end of current line
        rgb = i[0].split(';')  # get the rgb value

        # check if there is value on the next column
        if present <= width and col > pre_col:
            col = pre_col
            end = width
            row_end = True

        while present < end:
            decode[col][present] = np.array(rgb)
            if row_end and range[1] != 0 and present == width-1:
                present = 0
                end = int(range[1])
                col = int(range[0])
                row_end = False
            else:
                present += 1
            pre_col = col

    return decode


# ------------------------------------------------------------------ #

# Load image

image = Image.open('image.jpeg', 'r')
height = image.height
width = image.width
pixel_array = np.array(image)

image.show()
# IMAGE BEFORE QUANTIZATION
print('Before quantization\n', pixel_array, '\n\n\n')      # εκτυπώνω

# QUANTIZATION OF IMAGE
quant_array = quantization(10, pixel_array)

# ENCODE THE QUANTIZE IMAGE
result_encoder = encoder(quant_array)

# DECODE THE PREVIOUS ENCODING
result_decoder = decoder(result_encoder)

# MAKE THE IMAGE FROM THE DECODING
newImg = de_quantization(10, result_decoder)  # η συνάρτηση de_quantization έχει την τιμή της επαναφοράς μετά την κβάντιση (παρατηρείται μικρή διαφορά)

# CREATE AND SHOW THE IMAGE FROM THE DE-QUANTIZATION
img = Image.fromarray(newImg)
img.save("reconstruct_image.jpg")


img.show()

with open('apotelesmata.txt', 'w') as text:
    print(result_encoder, file=text)
