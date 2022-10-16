"""This module contains the code for creating a dataset from captioned images."""

import logging
import os
import json
import shutil
import datasets
from PIL import Image

from enum import Enum

logger = logging.getLogger(__name__)

class Operation(Enum):
    """Enum for operations"""
    LINK = "link"
    COPY = "copy"
    MOVE = "move"

class Format(Enum):
    """Enum for output formats"""
    JSON = "json"
    HF_LOCAL = "hf-local"
    HF = "hf"

class Writer(object):
    """Base class for writers
    """

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

    def write(self, path, caption, i):
        """Write image and caption to dataset_dir
        """
        raise NotImplementedError

    def close(self):
        """Close writer
        """
        raise NotImplementedError

class LocalHFWriter(Writer):
    """Writer for HuggingFace datasets
    """

    def __init__(self, dataset_dir):
        super().__init__(dataset_dir)
        self.ddict = {
            "image": [],
            "caption": [],
        }
        self.features = datasets.Features({
            "image": datasets.Value("binary"),
            "caption": datasets.Value("string"),
        })

    def write(self, path, caption, i):
        with Image.open(path).resize([512,512]).convert('RGB') as image:
            self.ddict["image"].append(image.tobytes())
        self.ddict["caption"].append(caption)

    def close(self):
        ds = datasets.Dataset.from_dict(self.ddict, self.features)
        ds.to_parquet(self.dataset_dir)

class HFWriter(Writer):
    """Writer to upload HuggingFace dataset directly
    """

    def __init__(self, dataset_dir):
        super().__init__(dataset_dir)
        self.ddict = {
            "image": [],
            "caption": [],
        }
        self.features = datasets.Features({
            "image": datasets.Image(decode=True),
            "caption": datasets.Value("string"),
        })

    def write(self, path, caption, i):
        self.ddict["image"].append(path)
        self.ddict["caption"].append(caption)

    def close(self):
        ds = datasets.Dataset.from_dict(self.ddict, self.features)
        ds.push_to_hub(self.dataset_dir)


class JSONWriter(Writer):
    """Writer for JSON format
    """

    def __init__(self, dataset_dir, operation):
        super().__init__(dataset_dir)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        self.operation = operation
        self.metadata_path = os.path.join(dataset_dir, "metadata.jsonl")
        self.metadata = open(self.metadata_path, "w", encoding="utf-8")

    def write(self, path, caption, i):
        ds_image_name = "%06d.jpg" % i
        ds_image_path = os.path.join(self.dataset_dir, ds_image_name)

        if not os.path.exists(ds_image_path):
            if self.operation == Operation.LINK:
                os.symlink(path, ds_image_path)
            elif self.operation == Operation.COPY:
                shutil.copyfile(path, ds_image_path)
            elif self.operation == Operation.MOVE:
                shutil.move(path, ds_image_path)
            else:
                raise ValueError("Unknown operation")
        else:
            logger.warning("File in dataset path %s already exists, skipping", path)

        json.dump({"file_name": ds_image_name, "text": caption}, self.metadata)

    def close(self):
        self.metadata.close()

def dataset(image_dir: str, dataset_dir: str, operation: Operation = "link", output_format: str = "hf"):
    """Create a dataset by moving, copying or symlinking data from image_dir to dataset_dir and 
    creating metadata.jsonl with images captions from .txt files"""

    logger.info("operation: %s", operation)

    if output_format == Format.JSON:
        writer = JSONWriter(dataset_dir, operation)
    elif output_format == Format.HF_LOCAL:
        writer = LocalHFWriter(dataset_dir)
    elif output_format == Format.HF:
        writer = HFWriter(dataset_dir)
    else:
        raise ValueError("Unknown output format")
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

            with open(caption_path, "r", encoding="utf-8") as caption_file:
                caption = caption_file.read()

            writer.write(path, caption, i)
    writer.close()
