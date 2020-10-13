# NeRF++
Codebase for arXiv preprint: 
* Work with 360 capture of large-scale unbounded scenes.
* Support multi-gpu training and inference with PyTorch DistributedDataParallel (DDP). 
* Optimize per-image autoexposure (**experimental feature**)

## Data
* Download our preprocessed data from [tanks_and_temples](https://drive.google.com/file/d/11KRfN91W1AxAW6lOFs4EeYDbeoQZCi87/view?usp=sharing), [lf_data](https://drive.google.com/file/d/1gsjDjkbTh4GAR9fFqlIDZ__qR9NYTURQ/view?usp=sharing).
* Put the data in the sub-folder data/ of this code directory.
* Data format. 
    * Each scene consists of 3 splits: train/test/validation. 
    * Intrinsics and poses are stored as flattened 4x4 matrices.
    * Opencv camera coordinate system is adopted, i.e., x--->right, y--->down, z--->scene.
    * Scene normalization: move the average camera center to origin, and put all the camera centers inside the unit sphere.

## Create environment
```bash
conda env create --file environment.yml
conda activate nerf-ddp
```

## Training (Use all available GPUs by default)
```python
python ddp_train_nerf.py --config configs/tanks_and_temples/tat_training_truck.txt
```

## Testing (Use all available GPUs by default)
```python
python ddp_test_nerf.py --config configs/tanks_and_temples/tat_training_truck.txt \
                        --render_splits test,camera_path
```
