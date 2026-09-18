## 1. Download Data

Our training data are based on the segmented CCTA data (label) from [ImageCAS](https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT).

## 2. Split Left and Right Coronary Artery Branches

Left and right coronary artery branches are separated using `data_split.py`. Only subjects that are successfully separated are retained. The results must be manually inspected to verify that the left and right branches are correctly matched: if mismatched, they should be swapped; if separation fails, they should be removed.
