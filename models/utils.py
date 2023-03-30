import os
import math
import numpy as np

from PIL import Image
from torchvision.utils import make_grid


def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    """
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    """
    tensor = tensor.clamp_(*min_max)  # clamp
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            "Only support 4D, 3D and 2D tensor."
            " But received with dimension: {:d}".format(n_dim)
        )
    if out_type == np.uint8:
        img_np = ((img_np + 1) * 127.5).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type).squeeze()


def postprocess(images):
    return [tensor2img(image) for image in images]


def save_images(results, result_dir, epoch):
    result_path = os.path.join(result_dir, str(epoch))
    os.makedirs(result_path, exist_ok=True)

    names = list(results.keys())
    outputs = postprocess(list(results.values()))
    for i in range(len(names)):
        Image.fromarray(outputs[i]).save(os.path.join(result_path, names[i]))
