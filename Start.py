from __future__ import absolute_import, division, print_function
import os
from delete import delete
from transform import run
from multiprocessing import Pool, cpu_count
from module.CNN.test import Start_test
from module.CNN.train import Start_Train
from module.Transform.monodepth2.utils import download_model_if_doesnt_exist
import argparse
from rich.progress import Progress
import ssl
import contextlib
from io import StringIO

ssl._create_default_https_context = ssl._create_unverified_context

def get_image_list(root_folder):
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    image_list = []

    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            extension = os.path.splitext(filename)[-1].lower()
            if extension in image_extensions:
                image_path = os.path.join(foldername, filename)
                image_list.append(image_path)
    root_folder_len = len(root_folder)
    image_list = [image_path[root_folder_len:] for image_path in image_list]

    return image_list

def start_pool(image_l, model_path, args):
    with Pool(processes=cpu_count()) as pool:
        results = [pool.apply_async(transform_image, (image_name, model_path, args)) for image_name in image_l]

        progress = Progress(total=len(results), auto_refresh=False)
        task_id = progress.add_task("Processing images...", total=len(results))

        image_tl = []
        for r in results:
            if r is not None:
                image_tl.append(r.get())
            progress.update(task_id, advance=1)

    print("\n All Done, Check the output folder for see the results")
    return image_tl


def transform_image(image_name, model_path, args):
    try:
        with contextlib.redirect_stdout(StringIO()):
            out = run(image_name, model_path, args)
        return out
    except:
        return None



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
    parser.add_argument('--test', action='store_true', help='Set CNN In Testing mode')
    parser.add_argument('--train', action='store_true', help='Set CNN in Training mode')
    parser.add_argument('--notransform', action='store_true', help='disable the Transform function')
    args = parser.parse_args()

    #Delete all temp file
    delete()
    print("Delete temp file Completed \n")
    
    #Check folder files
    if not os.listdir('input'):
        print("The Input Folder is empty, make sure that there are images present")
        exit()
    

    image_l = get_image_list('input')
    
    #Check if empty
    if len(image_l) == 0:
        print("Inside the Input folder there are no imges")
        exit()

    if not args.notransform:
        #Start Loop 
        download_model_if_doesnt_exist(args.model_name)
        model_path = os.path.join("module/Transform/models", args.model_name)
        image_tl = start_pool(image_l=image_l,model_path=model_path,args=args)
        delete()

    out_path = 'output'
    in_path = 'input'

    if args.train:
        #Check if in input is added a Dataset
        if os.path.exists('output/train'):
            out_path = 'output/train'
            in_path = 'input/train'
        print("Start Training (Ci puo mettere vari minuti per il load dei Dataset): ")
        Start_Train(in_path,out_path)

    if args.test:
        #Check if in input is added a Dataset
        if os.path.exists('output/test'):
            out_path = 'output/test'
            in_path = 'input/test'
        print("Start Testing: ")   
        Start_test(in_path,out_path)
