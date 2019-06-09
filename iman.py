

def get_stats(img):
    from scipy import stats
    import math
    
    for i in range(len(img.shape)):
        s = stats.describe(img[:,:,i].flatten())
        print('minmax: ', s.minmax)
        print('mean: ', s.mean)
        print('skewness: ', s.skewness)
        print('kurtosis: ', s.kurtosis)
        print('stddev: ', math.sqrt(s.variance))
    return 

def threshold3(img, threshold_min, threshold_max):
    import numpy as np
    
    mask = np.zeros_like(img); 
    mask[(img < threshold_max) & (img > threshold_min)] = 1; print(np.amax(mask), np.amin(mask)); print(sum(sum(mask)))
    img = img * mask; print(np.amax(img), np.amin(img))
    return img, mask

def draw_target_area(img):
    import cv2
    # scale factor [width, height] # img.shape is [height, width, depth]
    scale = [img.shape[1]/ref_shape[1], img.shape[0]/ref_shape[0]]; scale

    # targeted area
    top_left = [round(ref_top_left[0] * scale[0]), round(ref_top_left[1] * scale[1])]; top_left 
    bottom_right = [round(ref_bottom_right[0] * scale[0]), round(ref_bottom_right[1] * scale[1])]; bottom_right 

    img = cv2.rectangle(img, (top_left[0], top_left[1]), (bottom_right[0], bottom_right[1]), (0,255,0), 4)
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
            plt.figure(figsize=(20, 10))
            plt.axis('off')

            for d in range(3):
                plt.subplot(1,3,d+1)
                plt.imshow(img[:,:,d], cmap='gray')
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