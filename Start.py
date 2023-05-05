from __future__ import absolute_import, division, print_function
import os
from delete import delete
from transform import run
from module.CNN.test import Start_test
import argparse
import imghdr                   


if __name__  == '__main__':
    #Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', type=float, default=2.0, help='f value (controls brightness)')
    parser.add_argument('--l', type=float, default=0.5, help='l value (controls balance of attenuation constants)')
    parser.add_argument('--p', type=float, default=0.01, help='p value (controls locality of illuminant map)')
    parser.add_argument('--min-depth', type=float, default=0.0,
                        help='Minimum depth value to use in estimations (range 0-1)')
    parser.add_argument('--max-depth', type=float, default=1.0,
                        help='Replacement depth percentile value for invalid depths (range 0-1)')
    parser.add_argument('--spread-data-fraction', type=float, default=0.05,
                        help='Require data to be this fraction of depth range away from each other in attenuation estimations')
    parser.add_argument('--size', type=int, default=320, help='Size to output')
    parser.add_argument('--monodepth-add-depth', type=float, default=2.0, help='Additive value for monodepth map')
    parser.add_argument('--monodepth-multiply-depth', type=float, default=10.0,
                        help='Multiplicative value for monodepth map')
    parser.add_argument('--model-name', type=str, default="mono+stereo_1024x320",
                        help='monodepth model name')
    parser.add_argument('--output-graphs', action='store_true', help='Output graphs')
    parser.add_argument('--raw', action='store_true', help='RAW image')
    args = parser.parse_args()

    #Delete all temp file
    delete()
    print("Delete temp file Completed \n")
    
    #Check folder files
    if not os.listdir('input'):
        print("The Input Folder is empty, make sure that there are images present")
        exit()
    


    image_l = []
    image_tl = []
    #Check if inside there are images
    for file in os.listdir('input'):
        file_path = os.path.join('input', file)
        if imghdr.what(file_path) is not None and not file.startswith("._"):
            image_l.append(file)
    
    #Check if empty
    if len(image_l) == 0:
        print("Inside the Input folder there are no imges")
        exit()

   #Start Loop 
    for image_name in image_l:
        try:
            out = run(image_name,args)
            image_tl.append(out)
        except:
            print("Error Transforming the image : " + image_name+"; Skipped, Continue to next image")
            continue
    
    print("\n All Done, Check the output folder for see the results")

    Start_test('input','output')

    #Start Testing & Valuating with the images

    