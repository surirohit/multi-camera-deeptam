import os
import sys
import cv2
import argparse
import shutil
import pdb


def crop_image(im):
    # pdb.set_trace()
    center = im.shape[0]/2
    crop_val = 205
    im_new = im[center-crop_val:center+crop_val,:,:]
    return im_new


def expand_dim(im):
    return im


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", '-r', required=True)
    parser.add_argument("--output_path", '-o', required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    #Copy the the dict

    #For all cameras
    cam_folders = os.listdir(args.input_path)
    for cam in cam_folders:
        cam_path_rgb = os.path.join(args.output_path,cam, 'rgb')
        cam_path_depth = os.path.join(args.output_path,cam,'depth')
        if not os.path.exists(cam_path_rgb):
            os.makedirs(cam_path_rgb)
        if not os.path.exists(cam_path_depth):
            os.makedirs(cam_path_depth)

        shutil.copy(os.path.join(args.input_path,cam,'depth.txt'),os.path.join(args.output_path,cam))
        shutil.copy(os.path.join(args.input_path,cam,'rgb.txt'),os.path.join(args.output_path,cam))

        #For all rgb
        rgb_list = os.listdir(os.path.join(args.input_path,cam, 'rgb'))
        for rgb_file in rgb_list:
            im=cv2.imread(os.path.join(args.input_path,cam, 'rgb',rgb_file))
            im_new = crop_image(im)
            im_new = expand_dim(im_new)
            cv2.imwrite(os.path.join(args.output_path,cam, 'rgb',rgb_file), im_new)
        #For all depth
        depth_list = os.listdir(os.path.join(args.input_path,cam, 'depth'))
        for depth_file in depth_list:
            im=cv2.imread(os.path.join(args.input_path,cam, 'depth',depth_file))
            im_new = crop_image(im)
            im_new = expand_dim(im_new)
            cv2.imwrite(os.path.join(args.output_path,cam, 'depth',depth_file), im_new)








if __name__ == "__main__":
    main()
