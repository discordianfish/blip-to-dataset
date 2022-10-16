#!/bin/env python
"""Tool for dataset generation with BLIP."""

import argparse
import logging

from blip2ds.caption import captions
from blip2ds.dataset import dataset, Operation

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

def main():
    """Main function"""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True, dest="command")
    caption_command = subparsers.add_parser("captions",
        help="Add .txt file with caption for each image in <directory>")
    caption_command.add_argument("image_dir", help="Directory with images")
    caption_command.add_argument('--image-size', type=int, default=256,
        help='Image size to convert into before captioning')

    dataset_command = subparsers.add_parser("dataset",
        help="Create <directory>/metadata.json from captioned images in <directory>")
    dataset_command.add_argument("image_dir", help="Directory with images")
    dataset_command.add_argument("dataset_dir", help="Directory to create dataset in")
    dataset_command.add_argument('--operation', type=Operation, choices=list(Operation))

    kwargs = vars(parser.parse_args())
    globals()[kwargs.pop('command')](**kwargs)


if __name__ == "__main__":
    main()
