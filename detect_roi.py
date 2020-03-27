import os
import argparse
import sys
import glob
import util
import cv2
import pickle

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True, help="Path to a folder that contain imput images")
    parser.add_argument("-o", "--output_path", required=True, help="Path to write the result to")
    args = vars(parser.parse_args())
    INPUT_PATH = args['input_path']
    OUTPUT_PATH = args['output_path']

    # check for valid folder
    if os.path.exists(INPUT_PATH):
        print('input path: ', INPUT_PATH)
    else: 
        print(INPUT_PATH, ' invalid.')
        sys.exit()

    # create OUTPUT_PATH if not existed
    #if not os.path.exists(OUTPUT_PATH):
    #    try:
    #        os.mkdir(OUTPUT_PATH)
    #        os.mkdir(os.path.join(OUTPUT_PATH, 'roi1'))
    #        os.mkdir(os.path.join(OUTPUT_PATH, 'roi2'))
    #        os.mkdir(os.path.join(OUTPUT_PATH, 'roi3'))
    #    except OSError:
    #        print ("Creation of the directory %s failed" % OUTPUT_PATH)
    #    else:
    #        print ("Successfully created the directory %s " % OUTPUT_PATH)

    for fname in glob.glob(os.path.join(INPUT_PATH, '*.jpg')):
        print(fname)

        # read image
        src = cv2.imread(fname)

        # process image
        bw = util.processImage(src)
        
        # image dimension
        img_w = bw.shape[1]
        img_h = bw.shape[0]

        # TODO: put in config file
        ##### set size for filtering roi1s
        w_min_roi1 = int(img_w * 0.2)
        w_max_roi1 = img_w
        h_min_roi1 = img_w
        h_max_roi1 = int(img_h * 0.8)

        ##### set size for filtering roi2 table
        w_min_param = int(img_w * 0.8)
        w_max_param = img_w
        h_min_param = int(img_w * 0.15)
        h_max_param = img_w * 3

        ##### set size for filtering roi1 roi3/footer
        w_min_roi3 = int(img_w * 0.25)
        w_max_roi3 = img_w
        h_min_roi3 = int(img_w * 0.1)
        h_max_roi3 = img_w
        
            
        # morph image
        bw = util.morphImageThickShaveVerticalHorizontal(bw)

        # detect ROI: roi1
        roi1_coord, roi1 = util.detectROI(bw, w_min_roi1, w_max_roi1, h_min_roi1, h_max_roi1)
        
        # detect ROI: roi2 table
        roi2_coord, roi2 = util.detectROI(bw, w_min_param, w_max_param, h_min_param, h_max_param)

        # detect ROI: roi1 roi3/footer
        roi3_coord, roi3 = util.detectROI(bw, w_min_roi3, w_max_roi3, h_min_roi3, h_max_roi3)

        # write results onto processed image
        fname_out = os.path.join(OUTPUT_PATH, 'roi1', os.path.basename(fname)); print(fname_out)
        #cv2.imwrite(fname_out, roi1)
        fname_out = os.path.join(OUTPUT_PATH, 'roi2', os.path.basename(fname)); print(fname_out)
        #cv2.imwrite(fname_out, roi2)
        fname_out = os.path.join(OUTPUT_PATH, 'roi3', os.path.basename(fname)); print(fname_out)
        #cv2.imwrite(fname_out, roi3)

        # write results onto original image
        fname_out = os.path.join(OUTPUT_PATH, 'roi1', 'on_orig_' + os.path.basename(fname)); print(fname_out)
        on_orig_roi1 = util.drawRect(src, roi1_coord)
        #cv2.imwrite(fname_out, on_orig_roi1)
        fname_out = os.path.join(OUTPUT_PATH, 'roi2', 'on_orig_' + os.path.basename(fname)); print(fname_out)
        on_orig_roi2 = util.drawRect(src, roi2_coord)
        #cv2.imwrite(fname_out, on_orig_roi2)
        fname_out = os.path.join(OUTPUT_PATH, 'roi3', 'on_orig_' + os.path.basename(fname)); print(fname_out)
        on_orig_roi3 = util.drawRect(src, roi3_coord)
        #cv2.imwrite(fname_out, on_orig_roi3)

        roi_dict = {}
        roi_dict[os.path.basename(fname)] = {}
        roi_dict[os.path.basename(fname)]['roi1'] = roi1_coord
        roi_dict[os.path.basename(fname)]['roi2'] = roi2_coord
        roi_dict[os.path.basename(fname)]['roi3'] = roi3_coord

        fname_out = os.path.join(OUTPUT_PATH, os.path.splitext(os.path.basename(fname))[0]+'.pkl'); print(fname_out)
        #with open(fname_out, 'wb') as f:
        #    pickle.dump(roi_dict, f)
        
