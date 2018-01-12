def resize_images(datapath, width, height, save=False):
    print('resize_images.....')
    
    from PIL import Image
    import glob
    
    f_list = glob.glob(datapath + '*.jpg')

    for f in f_list:
        print(f)
        img = Image.open(f)
        img = img.resize((width, height), Image.ANTIALIAS) 
        
        if save == True
            img.save(f)

if __name__ == "__main__":
    main()