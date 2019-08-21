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