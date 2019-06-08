# Weddell_polynya_paper

# *UNDER CONSTRUCTION...*

This repository contains the Python analysis scripts used for Campbell et al. (2019), *Nature*, "[Antarctic offshore polynyas linked to Southern Hemisphere climate anomalies](https://www.nature.com/articles/s41586-019-1294-0)", doi:10.1038/s41586-019-1294-0.

Please contact me at [ethancc@uw.edu](mailto:ethancc@uw.edu) if you have any questions or difficulties with using this code. **Note:** My goal is to achieve full reproducibility, but this repository is still a work in progress. In other words, the scripts here are complex and not everything runs perfectly 'out of the box' yet! Within a few weeks of publication I intend to release a final archived version with a Zenodo DOI.

### General info:
These scripts contain (almost) all steps from start to finish: downloading the raw data, processing them, and creating the published figures. I am happy to send intermediate data products, such as a processed time series within one of the figures, upon request if that would be most expedient.

### Data:
The data analyzed in our paper are publicly available (see the "Data availability" statement in the Methods section), with the exception of updates to the UW Calibrated O2 package for float 5903616. These scripts will enable you to download necessary large data files that are unlikely to change substantially in the future. For data files that are smaller (including the UW O2 updates) or liable to change in the future, I have archived the versions used in this paper as a ZIP file on Google Drive. The instructions below describe how to access these files.

### Prerequisites:
1. `Python 3.6.7` or higher and `conda` installed ([Anaconda](https://www.anaconda.com/distribution/) distribution recommended; **note**: there is no guarantee that functionality will remain the same with future Python releases)

### Instructions for running this code and reproducing the paper figures:
1. Clone or download this GitHub repository onto your local machine. Note that it contains an existing directory structure that is required by the analysis scripts.
2. Recreate the required environment with all dependencies using the `environment.yaml` file, e.g. using:
```conda env create -f environment.yaml```
