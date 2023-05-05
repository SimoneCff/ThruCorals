from __future__ import absolute_import, division, print_function

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import rawpy
import PIL.Image as pil
import torch
from torchvision import transforms
from module.Transform.monodepth2.utils import download_model_if_doesnt_exist
import module.Transform.monodepth2.networks as networks
from module.Transform.seathru import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

def run(image_name, args):
    """Function to predict for a single image or folder of images
    """
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device, weights_only=False)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.to(device)
    encoder.load_state_dict(filtered_dict_enc)
    encoder.eval()

    print("   Loading pretrained decoder")

    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()


    # Load image and preprocess
    img = Image.fromarray(rawpy.imread('input/'+image_name).postprocess()) if image_name.endswith(".raw") else pil.open('input/'+image_name).convert('RGB')
    img.thumbnail((args.size, args.size), Image.ANTIALIAS)
    original_width, original_height = img.size
    # img = exposure.equalize_adapthist(np.array(img), clip_limit=0.03)
    # img = Image.fromarray((np.round(img * 255.0)).astype(np.uint8))
    input_image = img.resize((feed_width, feed_height), pil.LANCZOS)
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
    print('Preprocessed image', flush=True)

    # PREDICTION
    input_image = input_image.to(device)
    features = encoder(input_image)
    outputs = depth_decoder(features)

    disp = outputs[("disp", 0)]
    disp_resized = torch.nn.functional.interpolate(
        disp, (original_height, original_width), mode="bilinear", align_corners=False)

    # Saving colormapped depth image
    disp_resized_np = disp_resized.squeeze().cpu().detach().numpy()
    mapped_im_depths = ((disp_resized_np - np.min(disp_resized_np)) / (
            np.max(disp_resized_np) - np.min(disp_resized_np))).astype(np.float32)
    print("Processed image", flush=True)
    print('Loading image...', flush=True)
    depths = preprocess_monodepth_depth_map(mapped_im_depths, args.monodepth_add_depth,
                                        args.monodepth_multiply_depth)
    recovered = run_pipeline(np.array(img) / 255.0, depths, args)
    # recovered = exposure.equalize_adapthist(scale(np.array(recovered)), clip_limit=0.03)
    sigma_est = estimate_sigma(recovered, average_sigmas=True) / 10.0
    recovered = denoise_tv_chambolle(recovered, sigma_est)
    im = Image.fromarray((np.round(recovered * 255.0)).astype(np.uint8))
    output = "output/o_"+image_name+".png"
    im.save(output, format='png')
    print('Done.')
    print('\n')
    
    return output
    
    