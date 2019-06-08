# Weddell_polynya_paper

# *WORK IN PROGRESS...*

This repository contains the Python analysis scripts used for Campbell et al. (2019), *Nature*, "[Antarctic offshore polynyas linked to Southern Hemisphere climate anomalies](https://www.nature.com/articles/s41586-019-1294-0)", doi:10.1038/s41586-019-1294-0.

Please contact me at [ethancc@uw.edu](mailto:ethancc@uw.edu) if you have any questions or difficulties with using this code. **Note:** My goal is to achieve full reproducibility, but this repository is still a work in progress. In other words, the scripts here are complex and not everything runs perfectly 'out of the box' yet! Within a few weeks of publication I intend to release a final archived version with a Zenodo DOI.

### Prerequisites:
1. `Python 3.6.7` or higher and `conda` installed ([Anaconda](https://www.anaconda.com/distribution/) distribution recommended; **note**: there is no guarantee that functionality will remain the same with future Python releases)

### Instructions for running this code and reproducing the paper figures:
1. Clone or download this GitHub repository onto your local machine. Note that it contains an existing directory structure that is required by the analysis scripts.
2. Recreate the required environment with all dependencies using the `environment.yaml` file, e.g. using:
```conda env create -f environment.yaml```
