def resize_image(image, width=None, height=None, inter=None):
    
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
    img_resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return img_resized


if __name__ == "__main__":
    
    import cv2
    import glob
    import os

    ## replace with your output path 
    output_path = "../data/"

    for file in glob.glob('../data/*.tiff'): # replace your input path
        print(file)

        src = cv2.imread(file); print(src.shape)
        resized = resize_image(src, width=4000, height=None, inter=cv2.INTER_CUBIC) # change width to desired number of pixels

        output_fname = os.path.join(output_path, (os.path.basename(file)[:-5] + '_resized.tiff')); print(output_fname) # change to your desired name, or keep the original name
        cv2.imwrite(output_fname, resized)