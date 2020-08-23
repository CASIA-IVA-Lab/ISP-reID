#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import argparse
import numpy as np
from PIL import Image

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Visualization")

    parser.add_argument("--pseudo_labels_dir", type=str, default='./')
    parser.add_argument("--visualization_dir", type=str, default='./')
    parser.add_argument("--original_image_dir", type=str, default='./')
    parser.add_argument("--num_part", type=int, default=7)

    return parser.parse_args()


def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette


def main():
    args = get_arguments()

    num_part = args.num_part
    pseudo_labels_dir = args.pseudo_labels_dir
    visualization_dir = args.visualization_dir
    original_image_dir = args.original_image_dir


    palette = get_palette(num_part)
    
    for pseudo_labels in os.listdir(pseudo_labels_dir):
        parsing_result = Image.open(os.path.join(pseudo_labels_dir, pseudo_labels))
        parsing_result.putpalette(palette)
        if os.path.exists(os.path.join(original_image_dir, os.path.splitext(pseudo_labels)[0]+'.jpg')):
            orig_img = Image.open(os.path.join(original_image_dir, os.path.splitext(pseudo_labels)[0]+'.jpg'))
        else:
            orig_img = Image.open(os.path.join(original_image_dir, os.path.splitext(pseudo_labels)[0]+'.png'))
        orig_img = orig_img.resize((64,128), Image.ANTIALIAS).convert('RGBA')
        parsing_result = parsing_result.resize((64, 128), Image.NEAREST).convert('RGBA')
        fusion_img = Image.blend(orig_img, parsing_result, 0.5)
        fusion_img.save(os.path.join(visualization_dir, pseudo_labels))

    return


if __name__ == '__main__':
    main()
