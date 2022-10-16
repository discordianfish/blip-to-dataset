"""This module contains the code for the captioning model"""
import logging
import os
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image, UnidentifiedImageError

class Labeler(object):
    """Labeler class for labeling images using LAVIS."""
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name='blip_caption',
            model_type="base_coco",
            is_eval=True,
            device=self.device
        )

    def Caption(self, raw_image: Image):
        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)
        return self.model.generate({"image": image})


logger = logging.getLogger(__name__)

def captions(image_dir: str, image_size: int):
    """Add .txt file with caption for each image in <directory>"""
    logger.info("Loading model")
    labeler = Labeler()

    logger.info("Processing images")
    for root, _, files in os.walk(image_dir):
        for file in files:
            logger.info("Processing %s", file)
            path = os.path.join(root, file)
            caption_path = os.path.splitext(path)[0] + ".txt"
            if os.path.exists(caption_path):
                continue

            try:
                image = Image.open(path).resize((image_size, image_size)).convert('RGB')
            except UnidentifiedImageError:
                logger.warning("Skipping unreadable file %s", file)
                continue

            caption = labeler.Caption(image)[0]
            with open(caption_path, "w", encoding="utf-8") as file:
                file.write(caption)
            image.close()
            logger.info("Caption for %s: %s", path, caption)
