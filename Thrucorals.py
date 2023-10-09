from __future__ import absolute_import, division, print_function
import os
from delete import delete
from SeaThru.transform import run
from multiprocessing import Pool, cpu_count
from CNN.valuate import Smart_sorting
import argparse
from rich.progress import Progress
import ssl
import contextlib
from io import StringIO
import torch

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
    with Pool(processes=cpu_count()) as pool :
        with Progress() as progress:
         results = [pool.apply_async(transform_image, (image_name, model_path, args,args.folder)) for image_name in image_l]
         task_id = progress.add_task("Processing images...", total=len(results))
         image_tl = []
         for r in results:
                if r is not None:
                 image_tl.append(r.get())
                progress.update(task_id, advance=1)

    print("\n SeaThru Enhancment Done, Now Classifing...")

def transform_image(image_name, model_path, args, data_in):
    try:
        with contextlib.redirect_stdout(StringIO()):
            out = run(image_name, model_path, args, data_in)
        return out
    except:
        return None

def start_iterative(image_l,model_path,args, data_in):
    print("<WARNING> : If the model is in CUDA mode, the progress bar sometimes doesn't update, don't worry, the software is going!")
    with Progress() as progress:
        task_id = progress.add_task("Processing images...", total=len(image_l))
        for element in image_l:
            progress.update(task_id, advance=1)
            x = transform_image(element,model_path,args,data_in)
        print("\n SeaThru Enhancment Done, Now Classifing...")

if __name__  == '__main__':
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
    parser.add_argument('--transform_only', action='store_true', help='Set Transform only')
    parser.add_argument('--folder', type=str, required=True,help='Input Folder for Classification')
    args = parser.parse_args()
    delete()

    if not os.listdir(args.folder):
        print(" <Error> : Input Folder not found")
        exit()
    
    image_l = get_image_list(args.folder)
    
    if len(image_l) == 0:
        print("<ERROR> : No Images Found in the folder, ")
        exit()

    model_path = os.path.join("SeaThru/models", args.model_name)

    if torch.cuda.is_available() :
        start_iterative(image_l=image_l,model_path=model_path,args=args, data_in= args.folder)
    else:
        start_pool(image_l=image_l,model_path=model_path,args=args)
    delete()

    if not os.listdir('output'):
        print("<ERROR> : Output Folder not")
        exit()
        
    if not (args.transform_only):
        Smart_sorting('output')
        
    print("Operation DONE, Terminating script...")
