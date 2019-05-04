from skimage import color
from skimage import io
from skimage import img_as_ubyte
import  numpy as np
import cv2
from scipy import signal

def apply_smoothing_filter (array , fsize ) :
    avg_abs_before = np.mean(array)
    filter = []
    for i in range(1, fsize):
        filter.append(1 / fsize)

    new_values = np.convolve(array, filter, 'same')
    avg_abs_after = np.mean(np.abs(new_values))

    output = new_values*(avg_abs_before/avg_abs_after)
    return output

def get_written_area(binary_image):

    #get contours of the image
    img2, contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    contours = np.array(contours)

    # get the largest 3 contours in the image (supposed to be the three lines)
    contour_lengthes = np.array([contour.shape[0] for contour in contours])
    largest_contour_indecies = np.argsort(contour_lengthes)[-4:-1]

    # select 2 lines we want to cut the image at
    vertical_indecies = [contours[i][0, 0, 1] for i in largest_contour_indecies]
    edge_contour_indecies = largest_contour_indecies[np.argsort(vertical_indecies)[-2:]]

    # get cutting indecies
    upper_index = np.max(contours[edge_contour_indecies[0]][:, 0, 1])
    lower_index = np.min(contours[edge_contour_indecies[1]][:, 0, 1])

    return binary_image[upper_index:lower_index]



def to_printable_image(binary_image):
    return (1-binary_image.astype('int'))*255


def group_list (list , min_dist):
    list = np.array(list)
    g_start_index = 0
    g_end_index = 1
    output = []
    for i in range(1 , len(list)):
        if(list[i] - list[i-1] > min_dist):
            output.append(int(np.mean(list[g_start_index:g_end_index])))
            g_start_index = i
        g_end_index += 1
    output.append(int(np.mean(list[g_start_index:g_end_index])))
    return np.array(output)



def get_lines(binary_written_area):

    #invert the image so that every black pixel is considered 1 else 0
    binary_written_area_inverted = 1- (binary_written_area / np.max(binary_written_area))
    horizontal_projection = np.sum(binary_written_area_inverted, axis=1)

    #remove empty space after text
    SAFTY_PADDING = 50
    horizontal_projection_indecies = np.array(range(horizontal_projection.size))
    needed_areas = np.where(horizontal_projection > np.max(horizontal_projection) * 0.05)
    end_index = horizontal_projection_indecies[needed_areas][-1]
    if(end_index + SAFTY_PADDING< horizontal_projection_indecies[-1]):
        end_index+= SAFTY_PADDING
    else :
        end_index = horizontal_projection_indecies[-1]

    horizontal_projection = horizontal_projection[:end_index]
    binary_written_area = binary_written_area[:end_index]
    binary_written_area_inverted = binary_written_area_inverted[:end_index]

    io.imsave("data/lines/allLines.jpg" , binary_written_area)

    #do some smoothing on the horizontal projection curve to get local minimums
    filter_size = int(len(horizontal_projection)/30)
    smoothed_horizontal_projection = apply_smoothing_filter(horizontal_projection , filter_size)

    #get local minimums in the smoothed curve (horizontal projection)
    minimums_indecies = signal.argrelextrema(smoothed_horizontal_projection, np.less_equal, order=30)[0]

    #group nearby indecies together to be represented by one ponit
    minimums_indecies = group_list(minimums_indecies , 100)


    #get the lines in the image based on minimum indecies
    lines = []
    lines_pixcels = []
    for i in range(1 , len(minimums_indecies)):
        outputImage = binary_written_area[minimums_indecies[i-1]:minimums_indecies[i]]
        binary_line = binary_written_area_inverted[minimums_indecies[i-1]:minimums_indecies[i]]
        lines_pixcels.append(np.sum(binary_line))
        lines.append(outputImage )

    #remove empty lines and lines that are less than half full
    mediean_line_pixcels  = np.median(np.array(lines_pixcels))
    output_lines = []
    for i ,line in enumerate(lines):
        if (lines_pixcels[i] > mediean_line_pixcels * 0.3):
            #add white wrapper to lines for further processing to work well
            PADDING_PIXCELS  = 20
            HPP = int(PADDING_PIXCELS/2)
            line_rows = line.shape[0]
            line_cols = line.shape[1]

            new_line = np.ones((line_rows+PADDING_PIXCELS , line_cols+PADDING_PIXCELS)).astype("uint8") * 255
            new_line[HPP:line_rows+HPP , HPP:line_cols+HPP] = line
            output_lines.append(new_line)

    return output_lines

    ################# plot the horizontal projection ##################

    # y = smoothed_horizontal_projection[minimums_indecies]
    # pixel_indecies = range(0, horizontal_projection.shape[0])
    # # imgplot = plt.imshow(binary_written_area, cmap="gray")
    #
    # for i, line in enumerate(output_lines):
    #     io.imsave("data/lines/line" + str(i) + ".jpg", line)
    #
    # plt.subplot(1, 2, 1)
    # plt.plot(pixel_indecies, horizontal_projection)
    # plt.scatter(minimums_indecies, y, c='r')
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(pixel_indecies, smoothed_horizontal_projection)
    # plt.scatter(minimums_indecies, y, c='r')
    # plt.show()

def remove_black_line(binary_image):
    binary_image_inverted = 1 - binary_image / 255
    vertical_projection = np.sum(binary_image_inverted, axis=0)
    vertical_projection_midiean = np.median(vertical_projection)
    binary_image[:,vertical_projection > vertical_projection_midiean * 5] = 255
    return binary_image

def do_segmentation (image_path):
    img = io.imread(image_path)
    gray_image = img_as_ubyte(color.rgb2gray(img))
    ret, binary_image = cv2.threshold(gray_image, 200, 255, 0)
    binary_image = remove_black_line(binary_image)
    binary_written_area = get_written_area(binary_image)
    return  get_lines(binary_written_area)



