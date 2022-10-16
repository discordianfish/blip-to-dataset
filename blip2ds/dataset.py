"""This module contains the code for creating a dataset from captioned images."""

import logging
import os
import json
import shutil
from enum import Enum

logger = logging.getLogger(__name__)

class Operation(Enum):
    """Enum for operations"""
    LINK = "link"
    COPY = "copy"
    MOVE = "move"

def dataset(image_dir: str, dataset_dir: str, operation: Operation = Operation.LINK):
    """Create a dataset by moving, copying or symlinking data from image_dir to dataset_dir and 
    creating metadata.jsonl with images captions from .txt files"""

    # FIXME: Why is this needed? Why doesn't the function default work?
    if operation is None:
        operation = Operation.LINK

    logger.info("operation: %s", operation)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    metadata_path = os.path.join(dataset_dir, "metadata.jsonl")
    with open(metadata_path, "w", encoding="utf-8") as ds:
        i = 0
        for root, _, files in os.walk(image_dir):
            for file in files:
                i += 1
                path = os.path.join(root, file)
                base, ext = os.path.splitext(path)
                if ext == ".txt":
                    continue
                
                caption_path = base + ".txt"
                if not os.path.exists(caption_path):
                    logger.warning("Skipping image without caption %s", file)
                    continue

                ds_image_name = "%06d.jpg" % i
                ds_image_path = os.path.join(dataset_dir, ds_image_name)

                if not os.path.exists(ds_image_path):
                    if operation == Operation.LINK:
                        os.symlink(path, ds_image_path)
                    elif operation == Operation.COPY:
                        shutil.copyfile(path, ds_image_path)
                    elif operation == Operation.MOVE:
                        shutil.move(path, ds_image_path)
                    else:
                        raise ValueError("Unknown operation")
                else:
                    logger.warning("File in dataset path %s already exists, skipping", file)

                with open(caption_path, "r", encoding="utf-8") as caption_file:
                    caption = caption_file.read()
                json.dump({"file_name": ds_image_name, "text": caption}, ds)
