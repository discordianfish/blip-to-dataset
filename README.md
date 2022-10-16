# blip-to-dataset

Script(s) to take some images and turning them into a (parquett) dataset with
captions using LAVIS/BLIP.

## Usage
Create a `.txt` file for each image in `path/to/images` with caption
```
./make.py captions path/to/images
```

Create a [ImageFolder](https://huggingface.co/docs/datasets/image_dataset#imagefolder)
with `metadata.jsonl` dataset from the images and captions:
```
./make.py dataset path/to/images path/to/dataset
```

By default this symlinks the source images to the dataset. You can also move or
copy them using the `--operation {link,move,copy}` flag.
