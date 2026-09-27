## 1. Download Data

Our training data are based on the segmented CCTA data (label) from [ImageCAS](https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT).

## 2. Environment configuration

We need [TIGRE Toolbox](https://github.com/CERN/TIGRE). It is recommended to install it using a conda environment with Python 3.8, and use the command `conda install ccpi::tigre` to install it.

## 3. Split Left and Right Coronary Artery Branches

Left and right coronary artery branches are separated using `data_split.py`. Only subjects that are successfully separated are retained. The results must be manually inspected to verify that the left and right branches are correctly matched: if mismatched, they should be swapped; if separation fails, they should be removed.

## 4. Generate Back-projection Volumes

Back-projection volumes are generated using `lca_simulation_BV.py` and `rca_simulation_BV.py`, which produce the back-projection volumes for the left and right coronary arteries, respectively, each under two views.
