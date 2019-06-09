# Weddell_polynya_paper

# *UNDER CONSTRUCTION...*

This repository contains the Python analysis scripts used for Campbell et al. (2019), *Nature*, "[Antarctic offshore polynyas linked to Southern Hemisphere climate anomalies](https://www.nature.com/articles/s41586-019-1294-0)", doi:10.1038/s41586-019-1294-0.

Please contact me at [ethancc@uw.edu](mailto:ethancc@uw.edu) if you have any questions or difficulties with using this code. **Note:** My goal is to achieve full reproducibility, but this repository is still a work in progress. In other words, the scripts here are complex and not everything runs perfectly 'out of the box' yet! Within a few weeks of publication I intend to release a final, bug-free archived version with a Zenodo DOI.

### General info:
These scripts carry out all steps from start to finish: downloading the raw data, processing and analyzing them, and generating the published figures. I am happy to send intermediate data products (such as a processed time series within one of the figures) upon request if that would be most expedient, since running the full analysis script requires downloading numerous large data files.

### Data:
The data analyzed in our paper are publicly available (see the "Data availability" statement in the Methods section), with the exception of updates to the UW Calibrated O2 package for float 5903616. These scripts will enable you to download necessary large data files that are unlikely to change substantially in the future. For data files that are smaller (including the UW-O2 updates) or liable to change in the future, I have archived the versions used in this paper as a ZIP file on Google Drive.

### Organization and control flow:
The main analysis script is `weddell_polynya_paper.py`. Control flow within this script can be modified by changing boolean variables (`True`/`False`) near the top of the file to turn on/off sections of code that download data or generate the paper figures. Most of these sections have their own control flow, again modified using boolean variables. This main script calls functions located in the other files: `plot_tools.py`, `load_product.py`, `download_product.py`, `download_file.py`, `geo_tools.py`, and `time_tools.py`. Many of these functions are well-documented via docstrings (feel free to reuse for your own purposes!), but some are not well-documented.

Data is sourced from the `Data/` directory and subdirectories within, which should be downloaded from the Google Drive URL below. Figures and analysis results are saved to the `Results/` directory. This repository also contains the required [`h4toh5`](https://support.hdfgroup.org/products/hdf5_tools/h4toh5/) command-line utility.

### Prerequisites:
1. `Python 3.6.7` or higher and `conda` installed ([Anaconda](https://www.anaconda.com/distribution/) distribution recommended; **note**: there is no guarantee that these scripts will remain compatible with future Python releases)

### Instructions for running this code and reproducing the paper figures:
1. Clone or download this GitHub repository onto your local machine.
2. Download the `Data.zip` file (**TBD** GB) from this Google Drive link: **TBD**. Unzip into the `Weddell_polynya_paper` directory.
    - This contains the following archived data: Argo profiles from the GDAC\*, SOCCOM\*, and UW-O2 (original and updated); ETOPO1 bathymetry; the Marshall SAM index; GSHHG coast shapefiles; ISD\* and READER meteorological station records; COARE 2.0 turbulent heat fluxes; ERA-Interim monthly mean fields\* and land-sea mask; polar stereographic gridfiles and areafiles for AMSR 12.5 km and NSIDC 25 km sea ice concentration data; and WOD shipboard and instrumented seal profiles. The data are organized in the directory structure expected by the Python scripts.
    - The starred (\*) items may be re-downloaded, if you wish, using functions within the provided Python scripts. The rest of the data may be re-downloaded from the hosting websites (see note regarding WOD), which are linked via DOI in the paper's References section as well as within function docstrings in `load_product.py`.
    - The WOD hydrographic profiles may be re-downloaded using the [NCEI WODselect](https://www.nodc.noaa.gov/OC5/SELECT/dbsearch/dbsearch.html) utility with the following search parameters: coordinates 65°W-60°E, 90-50°S (or similar); datasets OSD, CTD, and APB; and variables T and S in "Column 1."
    - The COARE 2.0 turbulent heat flux time series for 2016 (derived from ERA-Interim) is the only processed data for which the processing script is not included here, but available upon request.
    - Also included in `Data.zip` are serialized data files known as "pickles", which I've used to store intermediate results, often the product of computationally expensive analyses. These are generated and loaded within the scripts using the Python `pickle` module. They are provided here for expediency, but may be regenerated within the scripts if desired by modifying the boolean control flow (e.g. changing variables such as `use_fig_1a_pickle` from `True` to `False` before execution).
3. Download the WOCE/Argo Global Hydrographic Climatology (WAGHC) fields (30 GB in total) from [University of Hamburg](http://icdc.cen.uni-hamburg.de/1/daten/ocean/waghc/) website. The required files are labeled `WAGHC_BAR_**_UHAM-ICDC_v1_0.nc`, where the asterisks denote months from `01` to `12`.
4. Recreate the required environment with all dependencies using the `environment.yaml` file, e.g. using: `conda env create -f environment.yaml`.
5. Try running `weddell_polynya_paper.py` with all control variables set to `False` to confirm that the dependencies are available and imported properly.
6. Optional: install Helvetica font for use by Matplotlib by following the instructions in [this](https://stackoverflow.com/questions/3176350/cannot-change-font-to-helvetica-in-matplotlib-in-python-on-mac-os-x-10-6) StackOverflow post, then uncomment the `mpl.rc('font' ... )` line at the top of `weddell_polynya_paper.py`. This is not necessary but will make text in the figures appear closer to the published versions.
7. Download the following sea ice concentration data by setting the relevant boolean variables in `weddell_polynya_paper.py` to `True` and running the script (don't forget to change the variables back to `False` afterwards):
    - ASI AMSR2 and AMSR-E (6 GB)
    - NSIDC GSFC Merged/CDR and NRT CDR (34 GB)
    - NSIDC Nimbus-5 ESMR (about 100 MB)
8. Download ERA-Interim six-hourly reanalysis fields for the Weddell Sea region (about 50 GB). Note that this requires a special procedure. First, register for an [ECMWF account](https://apps.ecmwf.int/registration/) if you don't have one already. Then, follow the instructions [from ECMWF](https://confluence.ecmwf.int/display/WEBAPI/Access+ECMWF+Public+Datasets) to install your private API login key to your local machine. Note that the `ecmwf-api-client` library is already included in the `environment.yaml` file. Next, run the `weddell_polynya_paper.py` script to submit MARS requests for the relevant ERA-Interim parameters within the Weddell Sea region. To do this, set `which_to_download` in the `download_ecmwf` section to `1`, run the script, and halt execution using Ctrl-C immediately after "Request is queued" appears. Repeat with `which_to_download` set to `2`. After a few hours, the files can be downloaded from http://apps.ecmwf.int/webmars/joblist/ (note this works best in Chrome). Save the two files in the `ECMWF_Weddell_unprocessed` folder as `erai_daily_weddell.nc` and `erai_daily_weddell_forecast.nc`, respectively. Lastly, run the main script with `process_ecmwf` set to `True`, which will save processed versions of the files into `ECMWF_Weddell_processed`. Feel free to delete the original copies.
9. You're done with setup! The analysis portions of the `weddell_polynya_paper.py` script should run without issues. Set the boolean control flow variables at the top to run sections of code corresponding to each figure.

### Known bugs:
* Importing the Matplotlib `Basemap` toolkit currently might require one to manually set the location of `PROJ_LIB`. To do this, edit the relevant line within the `import` statements in `weddell_polynya_paper.py` and `plot_tools.py`.
