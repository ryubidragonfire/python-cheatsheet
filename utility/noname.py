
def methodone():
    print('this is methodone.')


def display_images(img_list):
    import os
    import cv2
    import matplotlib.pyplot as plt
    #%matplotlib inline
    
    _ = plt.figure(figsize=(15,20))

    for i, fname in enumerate(img_list):
        fig_col = 5
        fig_row = round(len(img_list)/fig_col) + 1
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)

        _ = plt.subplot(fig_row,fig_col,i+1)
        _ = plt.imshow(img, cmap='gray')
        _ = plt.title(os.path.basename(fname))
    return


def display_histogram(img_list, ylim):
    import os
    import cv2
    import matplotlib.pyplot as plt

    _ = plt.figure(figsize=(15,20))

    for i, fname in enumerate(img_list):
        fig_col = 5
        fig_row = round(len(img_list)/fig_col) + 1
        img = cv2.imread(fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _ = plt.subplot(fig_col,fig_row,i+1)
        _ = plt.hist(img.ravel(),256,[0,256])
        _ = plt.ylim(0,ylim)
        _ = plt.title(os.path.basename(fname))
    