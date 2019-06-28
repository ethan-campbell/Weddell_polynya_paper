[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3259013.svg)](https://doi.org/10.5281/zenodo.3259013)

**This repository contains the Python analysis scripts used for Campbell et al. (2019), "[Antarctic offshore polynyas linked to Southern Hemisphere climate anomalies](https://www.nature.com/articles/s41586-019-1294-0)", *Nature*, doi:10.1038/s41586-019-1294-0.**

Please contact me at [ethancc@uw.edu](mailto:ethancc@uw.edu) if you have any questions or difficulties with using this code.

### Attribution:
This code is freely available for reuse as described in the MIT License included in this repository. However, if you use this code or any derived data products in an academic publication, you are obliged to provide the following citations:

* Campbell, E.C. (2019, June 28). Analysis scripts for "Antarctic offshore polynyas linked to Southern Hemisphere climate anomalies" (Version v1.0.0). Zenodo. doi:[10.5281/zenodo.3259013](http://doi.org/10.5281/zenodo.3259013).

* Campbell, E.C., Wilson, E.A., Moore, G.W.K., Riser, S.C., Brayton, C.E., Mazloff, M.R., Talley, L.D. (2019). Antarctic offshore polynyas linked to Southern Hemisphere climate anomalies. *Nature*, **570**(7761), 319-325. doi:[10.1038/s41586-019-1294-0](https://www.nature.com/articles/s41586-019-1294-0). 

### Description:
These scripts carry out all steps from start to finish: downloading the raw data, processing and analyzing them, and generating the published figures. I am happy to send intermediate data products (such as a processed time series within one of the figures) upon request if that would be most expedient, since running the full analysis script requires downloading numerous large data files.

### Data:
The data analyzed in our paper are publicly available (see the "Data availability" statement in the Methods section), with the exception of updates to the UW Calibrated O2 package for float 5903616. As a summary:
- The ZIP file described below contains smaller data files (including the UW-O2 updates) and archived versions of data products that are liable to change in the future.
- The scripts in this repository include routines for downloading larger data files that are less likely to change substantially in the future.

### Organization and control flow:
The main analysis script is `weddell_polynya_paper.py`. Control flow within this script can be modified by changing boolean variables (`True`/`False`) near the top of the file to turn on/off sections of code that download data or generate the paper figures. Most of these sections have their own control flow, again modified using boolean variables.

This main script calls functions located in the other files: `plot_tools.py`, `load_product.py`, `download_product.py`, `download_file.py`, `geo_tools.py`, and `time_tools.py`. The functions in `plot_tools.py` are not well-documented, but most of the other functions are described thoroughly via docstrings.

Data are sourced from the `Data/` directory and subdirectories within. The required directory structure is included in the `Data.zip` file described below. Figures and analysis results are saved to the `Results/` directory.

Note that this repository also contains the required [`h4toh5`](https://support.hdfgroup.org/products/hdf5_tools/h4toh5/) command-line utility.

### Prerequisites:
1. `Python 3` and `conda` installed. The [Anaconda](https://www.anaconda.com/distribution/) distribution is recommended, and includes `conda`. While there is no guarantee that these scripts will remain compatible with future Python releases, the provided `conda` environment file (see instructions below) ensures compatibility by providing a list of the exact versions of Python packages &ndash; and Python itself &ndash; that were used when writing this code.

### Step-by-step instructions for preparing to run the main analysis script:
1. Clone or download this GitHub repository onto your local machine.
2. Download the `Data.zip` file (1.5 GB) from [this Google Drive link](https://drive.google.com/file/d/1KlaWsMHdDTIOkSCTpHAyBUFtGmdj3VKT/view?usp=sharing) or from the most recent [release](https://github.com/ethan-campbell/Weddell_polynya_paper/releases) associated with this GitHub repository. Unzip into the `Weddell_polynya_paper` directory.
    - This contains the following archived data: Argo profiles from the GDAC\*, SOCCOM\*, and UW-O2 (original and updated); ETOPO1 bathymetry; the Marshall SAM index; GSHHG coast shapefiles; ISD\* and READER meteorological station records; COARE 2.0 turbulent heat fluxes; ERA-Interim monthly mean fields\* and land-sea mask; NSIDC Nimbus-5 ESMR sea ice concentration data\*; polar stereographic gridfiles and areafiles for AMSR 12.5 km and NSIDC 25 km sea ice grids; and WOD shipboard and instrumented seal profiles. The data are organized in the directory structure expected by the Python scripts.
    - The starred (\*) items may be re-downloaded, if you wish, using functions within the provided Python scripts. The rest of the data may be re-downloaded from the hosting websites (see note regarding WOD), which are linked via DOI in the paper's References section as well as within function docstrings in `load_product.py`.
    - The WOD hydrographic profiles may be re-downloaded using the [NCEI WODselect](https://www.nodc.noaa.gov/OC5/SELECT/dbsearch/dbsearch.html) utility with the following search parameters: coordinates 65°W-60°E, 90-50°S (or similar); datasets OSD, CTD, and APB; and variables T and S in "Column 1."
    - The COARE 2.0 turbulent heat flux time series for 2016 (derived from ERA-Interim) is the only processed data for which the processing script is not included here, but available upon request.
    - Also included in `Data.zip` are serialized data files known as "pickles", which I've used to store intermediate results, often the product of computationally expensive analyses. These are generated and loaded within the scripts using the Python `pickle` module. They are provided here for expediency, but may be regenerated within the scripts if desired by modifying the boolean control flow (e.g. changing variables such as `use_fig_1a_pickle` from `True` to `False` before execution).
3. ***Optional**, unless you want to re-compute the Extended Data Fig. 9 sub-pycnocline temperature anomaly records, which are stored in serialized "pickle" files (see above):* Download the WOCE/Argo Global Hydrographic Climatology (WAGHC) fields (30 GB in total) from the [University of Hamburg](http://icdc.cen.uni-hamburg.de/1/daten/ocean/waghc/) website. The required files are labeled `WAGHC_BAR_**_UHAM-ICDC_v1_0.nc`, where the asterisks denote months from `01` to `12`. Store these in the directory `Data/Shipboard/WAGHC2017/`.
4. Recreate the required Python environment with all dependencies using the `environment.yml` file. From within the repository, execute `conda env create -f environment.yml`. This will create a new conda environment called `weddell_polynya`. Next, activate the environment using `conda activate weddell_polynya`.
5. Try running `weddell_polynya_paper.py` with all control variables set to `False` to confirm that the dependencies are available and imported properly.
6. ***Optional**:* Install Helvetica font for use by Matplotlib. This is not necessary but will make text in the figures appear closer to the published versions. First, install Homebrew (if not already installed) using instructions [here](https://brew.sh). Next, follow the instructions in [this](https://stackoverflow.com/a/35687613) StackOverflow post. Lastly, uncomment the `mpl.rc('font' ... )` line at the top of `weddell_polynya_paper.py`.
7. Download the following sea ice concentration data by setting the relevant boolean variables in `weddell_polynya_paper.py` to `True` and running the script (don't forget to change the variables back to `False` afterwards):
    - ASI AMSR2 and AMSR-E (6 GB): set `download_amsr2` and `download_amsre` to `True`
    - NSIDC GSFC Merged/CDR and NRT CDR (34 GB): set `download_dmsp_v3` and `download_dmsp_nrt` to `True`
8. ***Optional**, unless you want to re-compute any of the serialized "pickle" files (see note above) that include ERA-Interim data:* Download ERA-Interim six-hourly reanalysis fields for the Weddell Sea region (about 50 GB). Note that this requires a special procedure. First, register for an [ECMWF account](https://apps.ecmwf.int/registration/) if you don't have one already. Then, follow the instructions [from ECMWF](https://confluence.ecmwf.int/display/WEBAPI/Access+ECMWF+Public+Datasets) to install your private API login key onto your local machine. Note that the `ecmwf-api-client` library is already included in the `environment.yml` file. Next, run the `weddell_polynya_paper.py` script to submit MARS API requests for the relevant ERA-Interim parameters within the Weddell Sea region. To do this, set `which_to_download` in the `download_ecmwf` section to `1`, run the script, and halt execution using Ctrl-C immediately after "Request is queued" appears. Repeat with `which_to_download` set to `2`. After a few hours, the files can be downloaded from http://apps.ecmwf.int/webmars/joblist/ (note this works best in Chrome). Save the two files in the `Data/Reanalysis/ECMWF_Weddell_unprocessed` directory as `erai_daily_weddell.nc` and `erai_daily_weddell_forecast.nc`, respectively. Lastly, run the main script with `process_ecmwf` set to `True`, which will save processed versions of the files into `/ECMWF_Weddell_processed/`. Feel free to delete the original copies.
9. You're done with setup! The analysis portions of the `weddell_polynya_paper.py` script should run without issues. Set the boolean control flow variables at the top to run sections of code corresponding to each figure. Note that the section for Extended Data Table 1 produces a LaTeX file, which can subsequently be compiled and saved to a PDF file.
