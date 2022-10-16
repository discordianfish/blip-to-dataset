# blip-to-dataset

Script(s) to take some images and turning them into a (parquett) dataset with
captions using LAVIS/BLIP.

## Usage
Create a `.txt` file for each image in `path/to/images` with caption
```
./make.py captions path/to/images
```

By default it create a
[ImageFolder](https://huggingface.co/docs/datasets/image_dataset#imagefolder)
dataset from the images and captions:
```
./make.py dataset path/to/images path/to/dataset
```

By default (`--format json`) this symlinks the source images to the dataset. You can also move or
copy them using the `--operation {link,move,copy}` flag.

The `--format {hf,hf-local,json}` can be used to generate different datasets:

- hf: Creates and uploads the dataset to huggingface, using `dataset_dir` as repo name
- hf-local: Creates a local huggingface dataset, saved to `dataset_dir`
- json: Creates a ImageFolder dataset, saved to `dataset_dir`
