import os
import cv2
import argparse
import shutil
import yaml
import numpy as np 
from PIL import Image

CROP_SIZE = (480,360)
SCALING = 1000.0
BASELINE = 0.0311146

def get_fx(folder):
    yaml_file = os.path.join(folder,'calib.yaml')

    with open(yaml_file, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            exit(1)

    return float(data['projection_matrix']['data'][0])

def get_crop_center(folder):
    yaml_file = os.path.join(folder,'calib.yaml')

    with open(yaml_file, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            exit(1)

    crop_center = (int(data['projection_matrix']['data'][2]),
                   int(data['projection_matrix']['data'][6]))

    return crop_center

def convert_and_crop_depth_image(input_path, crop_center, f_x):
    
    disparity_img = Image.open(input_path).convert('I')

    if disparity_img.mode != "I":
        raise Exception("Depth image is not in intensity format: {0}".format(depth_path))

    disparity = np.asarray(disparity_img)

    disparity = disparity / 8.0
    
    depth = BASELINE * f_x / disparity

    depth[depth==np.Inf] = 0
    depth_np = depth_np * SCALING
    
    depth_np = depth_np.astype(np.uint16)
    
    depth_np = depth_np[crop_center[1]-CROP_SIZE[1]/2:crop_center[1]+CROP_SIZE[1]/2, \
                crop_center[0]-CROP_SIZE[0]/2:crop_center[0]+CROP_SIZE[0]/2]

    return depth_np

def crop_rgb_image(im, crop_center):
    
    im_new = im[crop_center[1]-CROP_SIZE[1]/2:crop_center[1]+CROP_SIZE[1]/2, \
                crop_center[0]-CROP_SIZE[0]/2:crop_center[0]+CROP_SIZE[0]/2, \
                :]

    return im_new

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", '-s', required=True)
    parser.add_argument("--output_path", '-o', required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    #For all cameras
    cam_folders = os.listdir(args.input_path)
    for cam in cam_folders:

        crop_center = get_crop_center(os.path.join(args.input_path,cam))
        fx = get_fx(os.path.join(args.input_path,cam))

        cam_path_rgb = os.path.join(args.output_path,cam, 'rgb')
        cam_path_depth = os.path.join(args.output_path,cam,'depth')

        if not os.path.exists(cam_path_rgb):
            os.makedirs(cam_path_rgb)
        if not os.path.exists(cam_path_depth):
            os.makedirs(cam_path_depth)

        shutil.copy(os.path.join(args.input_path,cam,'depth.txt'),os.path.join(args.output_path,cam,'depth.txt'))
        shutil.copy(os.path.join(args.input_path,cam,'rgb.txt'),os.path.join(args.output_path,cam,'rgb.txt'))
        shutil.copy(os.path.join(args.input_path,cam,'calib.yaml'),os.path.join(args.output_path,cam,'calib.yaml'))

        
        print("[INFO] Starting folder: %s"%os.path.join(args.input_path,cam, 'rgb'))
        #For all rgb
        rgb_list = os.listdir(os.path.join(args.input_path,cam, 'rgb'))
        for rgb_file in rgb_list:
            im=cv2.imread(os.path.join(args.input_path,cam, 'rgb',rgb_file))
            im_new = crop_rgb_image(im, crop_center)
            cv2.imwrite(os.path.join(args.output_path,cam, 'rgb',rgb_file), im_new)
            cv2.waitKey(1)

        print("[INFO] Starting folder: %s"%os.path.join(args.input_path,cam, 'depth'))
        #For all depth
        depth_list = os.listdir(os.path.join(args.input_path,cam, 'depth'))
        for depth_file in depth_list:
            im_new = convert_and_crop_depth_image(os.path.join(args.input_path,cam, 'depth',depth_file), crop_center, fx)
            cv2.imwrite(os.path.join(args.output_path,cam, 'depth',depth_file), im_new)
            cv2.waitKey(1)

if __name__ == "__main__":
    main()