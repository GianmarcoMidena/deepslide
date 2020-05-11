# DeepSlide: A Sliding Window Framework for Classification of High Resolution Microscopy Images (Whole-Slide Images)

This repository is a sliding window framework for classification of high resolution whole-slide images, often called microscopy or histopathology images. This is also the code for the paper [Pathologist-level Classification of Histologic Patterns on Resected Lung Adenocarcinoma Slides with Deep Neural Networks](https://www.nature.com/articles/s41598-019-40041-7). For a practical guide and implementation tips, see the Medium post [Classification of Histopathology Images with Deep Learning: A Practical Guide](https://medium.com/health-data-science/classification-of-histopathology-images-with-deep-learning-a-practical-guide-2e3ffd6d59c5). 

*If you email me with questions about this repository, I will not respond. Open an issue instead.*

![alt text](figures/figure-2-color.jpeg)

## Requirements
- [imageio](https://pypi.org/project/imageio/)
- [NumPy 1.16](https://numpy.org/)
- [OpenCV](https://opencv.org/)
- [OpenSlide](https://openslide.org/)
- [OpenSlide Python](https://openslide.org/api/python/)
- [pandas](https://pandas.pydata.org/)
- [PIL](https://pillow.readthedocs.io/en/5.3.x/)
- [Python 3.6](https://www.python.org/downloads/release/python-360/)
- [PyTorch](https://pytorch.org/)
- [scikit-image](https://scikit-image.org/)
- [scikit-learn](https://scikit-learn.org/stable/install.html)
- [SciPy](https://www.scipy.org/)
- [images-to-dataset](https://github.com/GianmarcoMidena/images-to-dataset)
- [NVIDIA GPU](https://www.nvidia.com/en-us/)
- [Ubuntu](https://ubuntu.com/)

# Usage

Take a look at `code/config.py` before you begin to get a feel for what parameters can be changed.

1. Downscaling
2. Splitting
3. Tiling
4. Training
5. Testing on patches
6. Slide inference
7. Visualization

## 1. Downscaling:

Scale down the original slides.

```
python deepslide downscale
```

**Inputs**: `original_slides`, `downscale_factor` 

**Outputs**: `slides`

Note that `original_slides` must contain subfolders of images labeled by class. For instance, if your two classes are `a` and `n`, you must have `a/*.jpg` with the images in class `a` and `n/*.svs` with images in class `n`.

### Example
```
python deepslide downscale --downscale_factor 16
```

## 2. Splitting:

Splits the data into `N` subsets. 

```
python deepslide split
```

**Inputs**: `slides`, `slides_metadata.csv` 

**Outputs**: `slides_part_1.csv`, ..., `slides_part_N.csv`

Note that `slides` must contain subfolders of images labeled by class. 
For instance, if your two classes are `a` and `n`, 
you must have `a/*.jpg` with the images in class `a` and `n/*.jpg` with images in class `n`.

### Example
```
python deepslide split --n_splits 10
```

## 3. Tiling

- Generate patches for the training set.
- Balance the class distribution for the training set.
- Generate patches for the validation set.
- Generate patches by folder for WSI in the validation set.
- Generate patches by folder for WSI in the testing set.

```
python deepslide tile
```

Note that this will take up a significant amount of space. Change `--num_train_per_class` to be smaller if you wish not to generate as many windows. If your histopathology images are H&E-stained, whitespace will automatically be filtered. Turn this off using the option `--type_histopath False`. Default overlapping area is 1/3 for test slides. Use 1 or 2 if your images are very large; you can also change this using the `--slide_overlap` option.

**Inputs**: `slides/metadata.csv`, `slides_part_1.csv`, ..., `slides_part_N.csv`

**Outputs**: `train_patches_folder` (fed into model for training), `eval_patches_folder` (for validation and testing, sorted by WSI),
`train_patches_part_1.csv`, ..., `train_patches_part_N.csv`, 
`val_patches_part_1.csv`, ..., `val_patches_part_N.csv`,
`eval_patches_part_1.csv`, ..., `eval_patches_part_N.csv`

### Example
```
python deepslide tile --num_train_per_class 20000 --slide_overlap 2
```


## 4. Model Training

```
CUDA_VISIBLE_DEVICES=0 python deepslide train
```

We recommend using ResNet-18 if you are training on a relatively small histopathology dataset. You can change hyperparameters using the `argparse` flags. There is an option to retrain from a previous checkpoint. Model checkpoints are saved by default every epoch in `checkpoints`.

**Inputs**: `slides_part_1.csv`, ..., `slides_part_N.csv`, `train_patches_part_1.csv`, ..., `train_patches_part_N.csv`

**Outputs**: `checkpoints`, `logs`

### Example
```
CUDA_VISIBLE_DEVICES=0 python deepslide train --batch_size 32 --num_epochs 100 --save_interval 5
```


## 5. Testing on patches

Run the model on all the patches for each WSI in the validation and test set.

```
CUDA_VISIBLE_DEVICES=0 python deepslide test_on_patches
```

We automatically choose the model with the best validation accuracy. You can also specify your own. You can change the thresholds used in the grid search by specifying the `threshold_search` variable in `code/config.py`.

**Inputs**: `patches_eval_val`, `patches_eval_test`

**Outputs**: `preds_val`, `preds_test`

### Example
```
CUDA_VISIBLE_DEVICES=0 python deepslide test --auto_select False
```


## 6. Slide inference

The simplest way to make a slide inference 
is to choose the class with the most patch predictions. 
We can also implement thresholding on the patch level to throw out noise. 
To find the best thresholds, we perform a grid search. 
This function will generate csv files for each WSI with the predictions for each patch.
Do the final testing to compute the confusion matrix on the test set.

```
python deepslide test_on_slides
```

**Inputs**: `preds_val`, `preds_test`, `slides_test.csv`, `inference_val` and `slides_val.csv` (for the best thresholds)

**Outputs**: `inference_val`, `inference_test` and confusion matrix to stdout


## 7. Visualization

A good way to see what the network is looking at is to visualize the predictions for each class.

```
python deepslide visualize
```

**Inputs**: `slides_val.csv`, `slides_test.csv`, `preds_val`

**Outputs**: `vis_val`

You can change the colors in `colors` in `code/config.py`

![alt text](figures/sample.jpeg)

### Example
```
python deepslide visualize --vis_test different_vis_test_directory
```


# Quick Run

If you want to run all code and change the default parameters in `code/config.py`, run
```
sh code/run_all.sh
```
and change the desired flags on each line of the `code/run_all.sh` script.


# Pre-Processing Scripts

See `code/z_preprocessing` for some code to convert images from svs into jpg. This uses OpenSlide and takes a while. How much you want to compress images will depend on the resolution that they were originally scanned, but a guideline that has worked for us is 3-5 MB per WSI.

# Known Issues and Limitations
- Only 1 GPU supported.
- Should work, but not tested on Windows.
- In cases where no crops are found for an image, empty directories are created. Current workaround uses `try` and `except` statements to catch errors.
- Image reading code expects colors to be in the RGB space. Current workaround is to keep first 3 channels.
- This code will likely work better when the labels are at the tissue level. It will still work for the entire WSI, but results may vary.

# Still not working? Consider the following...

- Ask a pathologist to look at your visualizations.
- Make your own heuristic for aggregating patch predictions to determine the WSI-level classification. Often, a slide thats 20% abnormal and 80% normal should be classified as abnormal.
- If each WSI can have multiple types of lesions/labels, you may need to annotate bounding boxes around these.
- Did you pre-process your images? If you used raw .svs files that are more than 1GB in size, its likely that the patches are way too zoomed in to see any cell structures.
- If you have less than 10 WSI per class in the training set, obtain more.
- Feel free to view our end-to-end attention-based model in JAMA Network Open: [https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2753982](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2753982).

# Future Work

- Contributions to this repository are welcome. 
- Code for generating patches on the fly instead of storing them in memory for training and testing would save a lot of disk space.
- If you have issues, please post in the issues section and we will do our best to help.

# Citations

DeepSlide is an open-source library and is licensed under the [GNU General Public License (v3)](https://www.gnu.org/licenses/gpl-3.0.en.html). If you are using this library please cite:

```Jason Wei, Laura Tafe, Yevgeniy Linnik, Louis Vaickus, Naofumi Tomita, Saeed Hassanpour, "Pathologist-level Classification of Histologic Patterns on Resected Lung Adenocarcinoma Slides with Deep Neural Networks", Scientific Reports;9:3358 (2019).```

