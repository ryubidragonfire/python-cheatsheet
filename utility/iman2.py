####################
def morph(bw, horizontal_size, vertical_size, iterations):
    import numpy as np
    
    # Create the images that will use to extract the horizontal and vertical lines
    horizontal = np.copy(bw)
    vertical = np.copy(bw)
    
    horizontal = morph_horizontal(horizontal, horizontal_size, iterations)
    vertical = morph_vertical(vertical, vertical_size, iterations)

    horizontal_mask_invert_bool = create_horizontal_mask(horizontal)
    vertical_mask_invert_bool = create_vertical_mask(vertical)
    
    new = bw * vertical_mask_invert_bool * horizontal_mask_invert_bool
    
    return(new)

def create_vertical_mask(vertical):
    import numpy as np
    
    vertical_mask_bool = vertical==255 # True
    vertical_mask_invert_bool = np.invert(vertical_mask_bool)
    return(vertical_mask_invert_bool)

def create_horizontal_mask(horizontal):
    import numpy as np
    
    horizontal_mask_bool = horizontal==255 # True
    horizontal_mask_invert_bool = np.invert(horizontal_mask_bool)
    return(horizontal_mask_invert_bool)

def morph_horizontal(horizontal, horizontal_size, iterations):
    import numpy as np
    import cv2
    
    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure, iterations)
    horizontal = cv2.dilate(horizontal, horizontalStructure, iterations)
    
    return(horizontal)

def morph_vertical(vertical, vertical_size, iterations):
    import cv2
    
    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))

    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure, iterations)
    vertical = cv2.dilate(vertical, verticalStructure, iterations)
    
    return(vertical)

##################

def plot2array(fig):
    import numpy as np
    '''
    Implement: convert a matplotlib figure to a numpy array.
    
    Argument: 
        fig = matplotlib.figure
        arr = numpy array of height, width, depth
    '''
    arr = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    arr = arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return arr

def getCoordWithManyLines(array, axis, distance, height, lower, upper, n):
    '''
    Implements locate coordinates where many lines exist in a local region by means of histogram.
    
    Argument:
        array = 2D array to be processed, e.g. a binary image array
        axis = 0 for vertical lines, 1 for horizontal lines
        distance = distance between peaks of an histogram
        height = minimum height of peaks required of an histogram
        lower = lower limit of spacing between peaks
        upper = upper limit of spacing between peaks
        n = minimum number of consecutive lines required
    
    Return:
        locWithManyLines = return a list of a pair of indexes, indicating the start and end of area of interest 
    '''
    
    # Sum over an axis
    axis_sum = np.sum(array, axis); #print(axis_sum.shape)

    # Find peaks
    peaks, _ = find_peaks(axis_sum, distance=distance, height=height); #print(peaks)
    
    # Get the distance between peaks
    peaks_diff = np.diff(peaks); #print(peaks_diff)
    
    # Filter peaks that satisfied spacing specified by `lower` and `upper`
    peaks_diff_mask = np.greater(peaks_diff, lower) & np.less(peaks_diff, upper); #print(peaks_diff_mask)
    
    # Grouped by distance between peaks
    grouped_data = groupby(enumerate(peaks_diff_mask), key=lambda x: x[-1])

    locWithManyLines = []

    # Locate area where there are more than n number of consecutive lines
    for k, g in grouped_data:
        g = list(g)

        if k == True and len(g) > n: # more than n consecutive lines
            locWithManyLines.append((peaks[g[0][0]], peaks[g[-1][0]+1]))

    return(coordWithManyLines) 

############################################################################################################
# Get Bounding Boxes given a list of coordinated of  multiple start and end of vertical and horizontal lines
############################################################################################################

def getBBoxes(vertlines_coordinates, horzlines_coordinates):
    """
    Implements extraction of grid region give coordinates of multiple start and end of vertical and horizontal lines.
    
    Arguments:
        vertlines_coordinates, horzlines_coordinates
        
    Return:
    A list of coordinates defining the top left and botton right of a list of grid areas.
    
    """
    bbox_list = []

    for h in horzlines_coordinates:
        for v in vertlines_coordinates:
            top_left = v[0], h[0]
            bottom_right = v[1], h[1]
            bbox_list.append([top_left, bottom_right])
    
    return(bbox_list)

#######################################
# Resize Image preserving aspect ratio
#######################################

def resize_image(image, width=None, height=None, inter=None):
    import cv2
    
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    img_resized = cv2.resize(image, dim, interpolation = cv2.INTER_CUBIC)

    # return the resized image
    return img_resized

def detect_rectangles2(img, contours, w_min, w_max, h_min, h_max):
    import numpy as np
    import cv2

    img_out = np.zeros_like(img)
    img_copy = img.copy()
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        
        if (w_min<w) & (w<w_max) & (h_min<h) & (h<h_max):
            img_out = cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0,255,0), 4)
            
    return img_out 

def detect_rectangles2(img, contours, w_min, w_max, h_min, h_max, colour=None):
    import numpy as np
    import cv2

    img_out = np.zeros_like(img)
    img_copy = img.copy()
    
    rectangles = []
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        
        if colour == 'random':
            col = (np.random.randint(50, 220), np.random.randint(50, 220), np.random.randint(50, 220))
        else: 
            col = (0,255,0)

        if (w_min<w) & (w<w_max) & (h_min<h) & (h<h_max):
            rectangles.append((x,y,w,h))
            img_out = cv2.rectangle(img_copy, (x, y), (x+w, y+h), col , 5)
            
    return rectangles, img_out 


def detect_rectangles(img, contours, w_min, w_max, h_min, h_max, within=None):
    import numpy as np
    import cv2

    img_out = np.zeros_like(img)
    img_copy = img.copy()
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        
        if within == 'part1':
            [within_x_min, within_x_max, within_y_min, within_y_max] = [137-30, 157+30, 283-30, 810+30]

            if (within_x_min < x) & (x < within_x_max) & (within_y_min < y) & (y < within_y_max):

                if (w_min<w) & (w<w_max) & (h_min<h) & (h<h_max):
                    img_out = cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0,255,0), 4)

        else:
            if (w_min<w) & (w<w_max) & (h_min<h) & (h<h_max):
                img_out = cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0,255,0), 4)
                
    return img_out

def sort_contours(cnts, method="left-to-right"):
    import numpy as np
    import argparse
    import imutils
    import cv2
    
    # initialize the reverse flag and sort index    
    reverse = False
    i = 0
 
	# handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
 
	# handle if we are sorting against the y-coordinate rather than the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
 
	# construct the list of bounding boxes and sort them from top to bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b:b[1][i], reverse=reverse))
 
	# return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

def get_stats(img, display=False):
    from scipy import stats
    import math
    
    for i in range(len(img.shape)):
        s = stats.describe(img[:,:,i].flatten())

        if display == True:
            print('minmax: ', s.minmax)
            print('mean: ', s.mean)
            print('skewness: ', s.skewness)
            print('kurtosis: ', s.kurtosis)
            print('stddev: ', math.sqrt(s.variance))
    return s

def get_stats_1(img, display=False):
    from scipy import stats
    import math

    s = stats.describe(img.flatten())

    if display == True:
        print('minmax: ', s.minmax)
        print('mean: ', s.mean)
        print('skewness: ', s.skewness)
        print('kurtosis: ', s.kurtosis)
        print('stddev: ', math.sqrt(s.variance))
    return s

def threshold3(img, threshold_min, threshold_max):
    import numpy as np
    
    mask = np.zeros_like(img); 
    mask[(img < threshold_max) & (img > threshold_min)] = 1; #print(np.amax(mask), np.amin(mask)); print(sum(sum(mask)))
    img = img * mask; #print(np.amax(img), np.amin(img))
    return img, mask

def get_target_region(img, ref_shape, ref_top_left, ref_bottom_right):
    import cv2
    # ref_shape is [height, width]
    # scale factor [width, height] # img.shape is [height, width, depth]
    scale = [img.shape[1]/ref_shape[1], img.shape[0]/ref_shape[0]]; scale

    # targeted area
    top_left = [round(ref_top_left[0] * scale[0]), round(ref_top_left[1] * scale[1])]; top_left 
    bottom_right = [round(ref_bottom_right[0] * scale[0]), round(ref_bottom_right[1] * scale[1])]; bottom_right 

    img = cv2.rectangle(img, (top_left[0], top_left[1]), (bottom_right[0], bottom_right[1]), (0,255,0), 4)
    
    return img, top_left, bottom_right

def draw_target_region(img, top_left, bottom_right, region=None):
    import cv2

    if region == 'part1':
        top_left = [138, 284]
        bottom_right = [158, 811]

    img = cv2.rectangle(img, (top_left[0], top_left[1]), (bottom_right[0], bottom_right[1]), (0,255,0), 4)

    return img

def draw_region(img, rel_top_left, rel_bottom_right, pad=0):
    import cv2
    top_left = [round(img.shape[1]*rel_top_left[0]), round(img.shape[0]*rel_top_left[1])]
    bottom_right = [round(img.shape[1]*rel_bottom_right[0]), round(img.shape[0]*rel_bottom_right[1])]
    img = cv2.rectangle(img, (top_left[0]-pad, top_left[1]-pad), (bottom_right[0]+pad, bottom_right[1]+pad), (0,255,0), 4)
    return img

def plot_histogram(img, img_type=None, mode=None, img_display=False):
    import matplotlib.pyplot as plt

    # colour
    if len(img.shape) == 3:
        color = ['g','b','r']
        
        # GBR
        if img_type == 'GBR': 
            for d in range(3):
                plt.hist(img[:,:,d].flatten(), color=color[d], bins=256)

                if mode == 'separate':
                    plt.show()
            
            if mode != 'separate':
                plt.legend(['Green', 'Blue', 'Red'])
                plt.show()

        # HSV
        elif img_type == 'HSV':
            for d in range(3):
                if d == 0: # Hue
                    # hue
                    plt.hist(img[:,:,d].flatten(), color=color[0], bins=180)
                    
                else: # Saturation, Value
                    #print(np.unique(img[:,:,d]))
                    plt.hist(img[:,:,d].flatten(), color=color[d], bins=256)

                if mode == 'separate':
                    plt.show()
                    
            if mode != 'separate':
                plt.legend(['Hue', 'Saturation', 'Value'])
                plt.show() 
                
        # display layers
        if img_display == True:
            #plt.figure(figsize=(20, 10))
            #plt.axis('off')

            for d in range(3):
                #plt.subplot(1,3,d+1)
                plt.figure(figsize=(5,5))
                plt.imshow(img[:,:,d], cmap='gray')
                plt.show()
    # gray
    else:
        plt.hist(img.flatten(), bins=256); 
        
    return

def crop_target_area(img, region, offset):
    import cv2
    
    if region == 'signature':
        # template shape # [height, width]
        ref_shape = [1440, 858]

        # template coordiates # [top_left_x, top_left_y], [bottom_right_x, bottom_right_y]
        ref_top_left = [24, 1017]
        ref_bottom_right = [475, 1116]
        
    elif region == 'signhere':
        # template shape # [height, width]
        ref_shape = [1376, 863]

        # template coordiates # [top_left_x, top_left_y], [bottom_right_x, bottom_right_y]
        ref_top_left = [185, 1170]  
        ref_bottom_right = [570, 1215]
        
    # scale factor [width, height] # img.shape is [height, width, depth]
    scale = [img.shape[1]/ref_shape[1], img.shape[0]/ref_shape[0]]; scale

    # targeted area
    top_left = [round(ref_top_left[0] * scale[0]), round(ref_top_left[1] * scale[1])]; top_left 
    bottom_right = [round(ref_bottom_right[0] * scale[0]), round(ref_bottom_right[1] * scale[1])]; bottom_right 

    [x1, y1, x2, y2]  = [top_left[0], top_left[1], bottom_right[0], bottom_right[1]]; [x1, y1, x2, y2] 
    
    if offset != None:
        crop = img[y1+offset:y2-offset, x1:x2] # [y:y+h, x:x+w]
    else:
        crop = img[y1:y2, x1:x2] # [y:y+h, x:x+w]
    
    return crop

def change_colour_model(img, mode):
    import cv2
    
    if mode == 'gray':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mode == 'hsv':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    return img



if __name__ == "__main__":
    main()