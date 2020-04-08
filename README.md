# BBP's Effort for the COVID-19 Challenge

## Data and Assets
The notebooks in this repository assume the existence of the following
folders in the root folder:
- `data`
- `assets`

This folders are not part of the repository and have to be created locally.

The folder `assests` contains assets that do not depend on the CORD-19 dataset
and do not have to be versioned. These are for example
- synonym lists
- pre-trained models (e.g. BioSentVec)

The data folder contains the CORD-19 dataset and all files generated/derived from it.
Since this dataset gets updated on a regular basis, different versions of it need to be
kept separated. Therefore the `data` folder contains subfolders corresponding to
different versions and named by the date on which the CORD-19 dataset was downloaded.
The same subfolder should also contain all files derived from that dataset.

