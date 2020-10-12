# NeRF++
Codebase for paper: 
* Work with 360 capture of large-scale unbounded scenes.
* Support multi-gpu training and inference.

## Data
* Download our preprocessed data from [tanks_and_temples](), [lf_data]().
* Put the data in the code directory.
* Data format. 
** Each scene consists of 3 splits: train/test/validation. 
** Intrinsics and poses are stored as flattened 4x4 matrices.
** Opencv camera coordinate system is adopted, i.e., x--->right, y--->down, z--->scene.
* Scene normalization: move the average camera center to origin, and put all the camera centers inside the unit sphere.

## Training
```python
python ddp_train_nerf.py --config configs/tanks_and_temples/tat_training_truck.txt
```

## Testing
```python
python ddp_test_nerf.py --config configs/tanks_and_temples/tat_training_truck.txt --render_splits test,camera_path
```