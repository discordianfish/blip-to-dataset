#!/bin/env python

import argparse
import torch, os
import pandas
from PIL import Image

from lavis.models import load_model_and_preprocess


class Labeler(object):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, required=True)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--output-file', type=str, required=True)

    args = parser.parse_args()

    labeler = Labeler()
    dataset = [[]]

    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".jpg"):
                image = Image.open(os.path.join(root, file)).resize((args.image_size, args.image_size))
                caption = labeler.Caption(image)
                dataset.append([file, caption])
                print(caption)

    pandas.DataFrame(dataset).to_parquet(args.output_file, index=False)


if __name__ == "__main__":
    main()
