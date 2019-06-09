# -*- coding: utf-8 -*-

import os
import re
import warnings
import h5py
from scipy import constants
import scipy.interpolate as spin
import scipy.io.netcdf as spnc
import scipy.io as sio
from scipy import stats
from datetime import datetime, timedelta
from numpy import *
import pandas as pd
import xarray as xr
import pickle
import gsw
import seawater
import osr, gdal
import codecs

import geo_tools as gt
import time_tools as tt
import download_file as df


############# OCEAN DATA - OUTWARD-FACING FUNCTIONS ################


def argo_soccom(soccom_dir):
    """ Processes existing SOCCOM float profiles in text format, e.g. from quarterly snapshot with DOI.

    Example citation: see https://library.ucsd.edu/dc/object/bb0687110q

    """
    save_to_floats = soccom_dir

    # do a find-and-replace on data files to remove whitespace between some column names
    for data_filename in os.listdir(save_to_floats):
        if data_filename == 'README_snapshot.txt' or data_filename == 'get_FloatViz_data.m': continue

        orig_file_as_list = codecs.open(save_to_floats + data_filename,'rb',encoding='latin-1').readlines()
        new_file_as_list = []
        for line in orig_file_as_list:
            first_edit = line.replace('Lon [°E]', 'Lon[°E]')
            second_edit = first_edit.replace('Lat [°N]', 'Lat[°N]')
            new_file_as_list.append(second_edit)
        out_file = codecs.open(save_to_floats + data_filename,'wb',encoding='latin-1')
        out_file.writelines(new_file_as_list)
        out_file.close()


def argo_gdac_load_index(save_to_root):
    """ Accessor for index of locally stored Argo profiles from GDAC.

    Returns dict 'argo_gdac_index' with keys:
        'local_prof_index': full index of locally downloaded profiles in array format (see original file for details)
        'wmoids': list of float WMOids
        'num_profs': number of profiles in index
        'num_floats': number of floats in index

    """
    save_to_meta = save_to_root + 'Meta/'
    local_index_filename = 'ar_index_local_prof.txt'

    argo_gdac_index = {}
    data_frame = pd.read_csv(save_to_meta + local_index_filename, header=-1, low_memory=False)
    argo_gdac_index['local_prof_index'] = data_frame.values
    argo_gdac_index['num_profs'] = argo_gdac_index['local_prof_index']

    argo_gdac_index['wmoids'] = unique(argo_gdac_index['local_prof_index'][:,0])
    argo_gdac_index['num_floats'] = len(argo_gdac_index['wmoids'])

    return argo_gdac_index


def argo_soccom_load_index(save_to_soccom,save_to_UW_O2,include_UW_O2=True,verbose=False):
    """ Accessor for index of locally stored Argo profiles from SOCCOM FloatViz (plus a basic index of UW-O2 data)).
    
    Returns dict 'argo_soccom_index' with keys:
        'num_floats': number of SOCCOM floats
        'wmoids': list of SOCCOM float WMOids
        'uwids': list of SOCCOM float University of Washington IDs (NOTE: these are strings, not ints)
        'filenames': list of SOCCOM float data filenames
        'profile_nums': list of arrays of profile numbers for each float
        'UW_O2_wmoids': list of UW-O2 float WMOids (if include_UW_O2 is True)
        'UW_O2_filenames': list of UW-O2 float data filenames
        
    """
    save_to_floats = save_to_soccom

    argo_soccom_index = {}
    all_filenames = os.listdir(save_to_floats)
    argo_soccom_index['num_floats'] = len(all_filenames)
    argo_soccom_index['filenames'] = all_filenames
    argo_soccom_index['wmoids'] = []
    argo_soccom_index['uwids'] = []
    argo_soccom_index['profile_nums'] = []
    uwid_regexp = re.compile('//Univ. of Washington ID: ([0-9]*)')
    if '.DS_Store' in all_filenames: all_filenames.remove('.DS_Store')
    if 'README_snapshot.txt' in all_filenames: all_filenames.remove('README_snapshot.txt')
    if 'get_FloatViz_data.m' in all_filenames: all_filenames.remove('get_FloatViz_data.m')
    for filename in all_filenames:
        if verbose:
            print(filename)
        header_line_counter = 0
        with open(save_to_floats + filename, 'rb') as f:
            for line in f:
                if header_line_counter == 4:
                    argo_soccom_index['uwids'].append(uwid_regexp.findall(line.decode('latin-1'))[0])
                if 'Cruise	' in line.decode('latin-1'):
                    break
                header_line_counter += 1
        data_frame = pd.read_csv(save_to_floats + filename, header=header_line_counter, delim_whitespace=True,
                                 na_values=-1e10, encoding='latin-1')
        argo_soccom_index['wmoids'].append(unique(data_frame['Cruise'].values)[0])
        argo_soccom_index['profile_nums'].append(unique(data_frame['Station'].values))

    if include_UW_O2:
        save_to_floats = save_to_UW_O2
        all_filenames = os.listdir(save_to_floats)
        if '.DS_Store' in all_filenames: all_filenames.remove('.DS_Store')
        filename_regexp_1_1 = re.compile('UW_O2_V1_1.*.WMO([0-9]*).*.nc')
        filename_regexp_1_2b = re.compile('UW_O2_V1.2b.*.WMO([0-9]*).*.nc')
        argo_soccom_index['UW_O2_wmoids'] = []
        argo_soccom_index['UW_O2_filenames'] = []
        for filename in all_filenames:
            if filename[-3:] == '.nc':
                try:
                    this_wmoid = int(filename_regexp_1_2b.findall(filename)[0])
                    new_version = True
                except IndexError:
                    this_wmoid = int(filename_regexp_1_1.findall(filename)[0])
                    new_version = False
                if this_wmoid not in argo_soccom_index['UW_O2_wmoids']:
                    argo_soccom_index['UW_O2_wmoids'].append(this_wmoid)
                    argo_soccom_index['UW_O2_filenames'].append(filename)
                elif new_version:
                    argo_soccom_index['UW_O2_wmoids'][argo_soccom_index['UW_O2_wmoids'].index(this_wmoid)] = this_wmoid
                    argo_soccom_index['UW_O2_filenames'][argo_soccom_index['UW_O2_wmoids'].index(this_wmoid)] = filename

    return argo_soccom_index


def argo_gdac_float_meta(profile_index,float_wmoid):
    """ Accessor for metadata on a specific Argo float from GDAC.

    Returns dict 'this_float_meta' relevant to the given float's profiles, with following keys:
        'num_profs': number of profiles (integer)
        'prof_nums': array of profile numbers (integer form, so '_000D' and '_000' would both be integer 0)
        'prof_nums_full': array of profile numbers (full string [alphanumeric] form, preserving, e.g. '_000D')
        'prof_datetimes': array of profile datetimes (18-digit integer format)
        'prof_statuses': array of profile statuses (e.g. 'D' for delayed mode)
        'prof_lats': array of profile latitudes
        'prof_lons': array of profile longitudes
        'prof_position_flags': array of profile position QC flags (1 = likely good, 2 = interpolated, assumed under ice, 9 = bad)
        'prof_filenames': array of profile filenames

    """
    this_float_mask = profile_index[:,0] == float_wmoid

    prof_filenames = profile_index[this_float_mask, 1]
    this_float_num_profs = len(prof_filenames)
    filename_regexp_full = re.compile('[A-Z][0-9]*_([0-9]*[A-Z]*).nc')
    filename_regexp_int = re.compile('[A-Z][0-9]*_([0-9]*)[A-Z]*.nc')
    this_float_meta = {}
    this_float_meta['num_profs'] = this_float_num_profs
    this_float_meta['prof_nums'] = [int(filename_regexp_int.findall(prof_filenames[n])[0]) for n in range(this_float_num_profs)]
    this_float_meta['prof_nums_full'] = [filename_regexp_full.findall(prof_filenames[n])[0] for n in range(this_float_num_profs)]
    this_float_meta['prof_datetimes'] = profile_index[this_float_mask, 5]
    this_float_meta['prof_statuses'] = profile_index[this_float_mask, 2]
    this_float_meta['prof_lats'] = profile_index[this_float_mask, 6]
    this_float_meta['prof_lons'] = profile_index[this_float_mask, 7]
    this_float_meta['prof_position_flags'] = profile_index[this_float_mask, 3]
    this_float_meta['prof_filenames'] = profile_index[this_float_mask, 1]

    return this_float_meta


def argo_float_data(wmoid,argo_gdac_dir,argo_gdac_index,argo_soccom_index,prof_nums='all',smooth_sal=True,
                    compute_extras=False,smooth_N2_PV=False,smooth_N2_PV_window=25.0,use_unadjusted=False,
                    use_UW_O2_not_SOCCOM=False,verbose=False,correct_5904468_interim=True,allow_bad_soccom_qc=False):
    """ Accessor for Argo profile data. Parses/aggregates netCDF-3 files from GDAC, FloatViz text files from SOCCOM,
        and UW O2 v1.1 and assorted v1.2b netCDF-4 files from Robert Drucker.

    Args:
        wmoid: int (disallows the handful [one? two?] of early SOCCOM floats without WMOids)
        argo_gdac_dir: add 'Profiles/' + this_float_meta['prof_filenames'][prof] to get a GDAC profile
                       add 'Argo_index_pickles/' + argo_soccom_index['filenames'][float] to get a SOCCOM float
        argo_gdac_index: index of locally stored profiles from GDAC
        argo_soccom_index: index of locally stored float data files from SOCCOM FloatViz
        prof_nums: 'all', or specified as array of ints
        smooth_sal ([True] or False): apply mild 1-D quadratic smoothing spline fit to salinity measurements to
                                      eliminate artificial "staircases" at depth due to 0.001 psu resolution of data
        compute_extras (True or [False]): compute N^2 (buoyancy frequency / Brunt-Väisäla frequency squared),
                                             approximate isopycnic potential vorticity (IPV), and buoyancy loss required
                                             for convection to reach each observed depth
                                          note: attached 'pres' vectors for 'Nsquared' and 'PV' represent midpoints of
                                             original CTD pressures
        smooth_N2_PV (True or [False]): apply running average smoothing to noisy Nsquared and PV data
        smooth_N2_PV_window: running average window (in meters)
        use_unadjusted (True or [False]): import unadjusted profiles and ignore bad QC flags
        use_UW_O2_not_SOCCOM (True or [False]): if available, use UW-calibrated O2 profiles instead of SOCCOM FloatViz
        correct_5904468_interim ([True] or False): apply interim salinity drift correction to 5904468 (see below)
        allow_bad_soccom_qc (True or [False]): allow SOCCOM data with QC flag of '4' (questionable) as well as '0' (good)
    Returns: dict 'this_float_data' with keys:
        wmoid: int (WMOid)
        uwid: 'unknown_or_not_applicable' or str (University of Washington ID; will only fill in for SOCCOM floats)
        is_soccom: True or False (is this a SOCCOM float?)
        is_uw_o2: True or False (does this float have O2 data corrected by UW?)
        institution: string, e.g. 'UW'
        num_profs: int (number of profiles in GDAC; not necessarily same as in SOCCOM FloatViz file;
                        also does not reflect removal of profiles with bad QC in this function)
        profiles: list of dicts for each profile in GDAC, with keys:
            prof_num: int (profile number; integer form, so '_000D' and '_000' would both be integer 0)
            prof_num_full: string (GDAC full profile number; alphanumeric form, preserving, e.g. '_000D')
            datetime: int (GDAC datetime; 14-digit integer format)
            status: str (GDAC profile status, e.g. 'D' for delayed mode)
            lat: float (GDAC latitude)
            lon: float (GDAC longitude; -180 to 180)
            position_flag: int (GDAC profile position QC flag, as below):
                1 = likely good, 2 = interpolated, assumed under ice, 9 = bad
            DATA PARAMETERS: from GDAC: temp, ptmp, ctmp, psal, asal, sigma_theta, Nsquared, PV, destab (if included)
                                        note: if NUM_PROFS > 1, this will be ignoring all except the first profile
                                        note: these GDAC parameters are guaranteed to have identical pres/depth arrays
                             from SOCCOM or UW-O2: SEE ARRAY 'soccom_param_names' FOR PARAMETER NAMES
                each is a dict with keys:
                    data: array of measured values, in direction of increasing depth (surface downwards)
                    name: string of name of parameter, formatted for a plot axis
                    pres: array of corresponding pressure (dbar)
                    depth: array of corresponding depths (m, positive)
                    units: string of units of data
    Example:
        this_float_data['wmoid']
        this_float_data['num_profs']
        len(this_float_data['profiles'])    or consider    for profile in this_float_data['profiles']
        this_float_data['profiles'][55]['prof_num']
        this_float_data['profiles'][55]['ptmp']['data']
        this_float_data['profiles'][55]['ptmp']['pres']
        this_float_data['profiles'][55]['ptmp']['units']

    """
    save_to_gdac_dir = argo_gdac_dir + 'Profiles/'
    save_to_soccom_dir = argo_gdac_dir + 'SOCCOM/'
    save_to_UW_O2_dir = argo_gdac_dir + 'UW-O2/'
    soccom_params_to_save = array(['Oxygen[µmol/kg]','OxygenSat[%]','Nitrate[µmol/kg]','Chl_a[mg/m^3]',
                                   'Chl_a_corr[mg/m^3]','b_bp700[1/m]','b_bp_corr[1/m]','POC[mmol/m^3]',
                                   'pHinsitu[Total]','pH25C[Total]','TALK_LIAR[µmol/kg]','DIC_LIAR[µmol/kg]',
                                   'pCO2_LIAR[µatm]'])
    soccom_param_abbrevs = array(['Oxygen','OxygenSat','Nitrate','Chl_a','Chl_a_corr','b_bp700','b_bp_corr','POC',
                                  'pHinsitu','pH25C','TALK_LIAR','DIC_LIAR','pCO2_LIAR'])
    soccom_param_names = array(['Oxygen', 'Oxygen saturation', 'Nitrate', 'Chl-a', 'Chl-a', 'Backscatter (700 nm)',
                                'Backscatter', 'POC', 'In-situ pH', 'pH25C', 'Total alkalinity', 'DIC', 'pCO2'])
    soccom_units_names = array(['µmol/kg','%','µmol/kg',r'mg/m$^3$',r'mg/m$^3$','1/m','1/m',r'mmol/m$^3$','Total','Total',
                                'µmol/kg','µmol/kg','µatm'])
    this_float_meta = argo_gdac_float_meta(argo_gdac_index['local_prof_index'], wmoid)

    this_float_data = {}
    this_float_data['wmoid'] = wmoid
    this_float_data['num_profs'] = this_float_meta['num_profs']
    this_float_data['profiles'] = []

    # load GDAC profiles
    for prof_index, prof in enumerate(this_float_meta['prof_nums']):
        if verbose:
            print('examining profile #: ' + str(prof))
        if prof_nums is not 'all':
            if prof not in prof_nums:
                continue

        # load profile netCDF file; save institution
        gdac_prof_file = spnc.netcdf_file(save_to_gdac_dir + this_float_meta['prof_filenames'][prof_index],'r',mmap=False)
        if prof_index is 1:
            try:
                this_float_data['institution'] = str(gdac_prof_file.institution,'utf-8')
            except:
                this_float_data['institution'] = 'Unknown institution'

        # ignore this profile if it has major QC issues
        gdac_prof_position_flag = this_float_meta['prof_position_flags'][prof_index]
        if gdac_prof_position_flag is 9:  # bad QC!
            continue
        gdac_data_mode = str(gdac_prof_file.variables['DATA_MODE'][0], 'utf-8')
        if (gdac_data_mode is 'D' or gdac_data_mode is 'A') and use_unadjusted is False:
            var_suffix = '_ADJUSTED'
            qc_var_suffix = '_ADJUSTED_QC'
        else:
            var_suffix = ''
            qc_var_suffix = '_QC'
        try:
            gdac_pres_qc = gdac_prof_file.variables['PRES' + qc_var_suffix][0].astype(int)
            gdac_temp_qc = gdac_prof_file.variables['TEMP' + qc_var_suffix][0].astype(int)
            gdac_psal_qc = gdac_prof_file.variables['PSAL' + qc_var_suffix][0].astype(int)
        except:
            continue

        # examine QC flags of CTD data; create mask that is True where all data is good
        if use_unadjusted is False:
            gdac_qc_mask = ~logical_or(logical_and(gdac_pres_qc != 1, gdac_pres_qc != 2),
                                       logical_or(logical_and(gdac_temp_qc != 1, gdac_temp_qc != 2),
                                                  logical_and(gdac_psal_qc != 1, gdac_psal_qc != 2)))
        else:
            gdac_qc_mask = tile(True,len(gdac_psal_qc))

        # FIXME: temporary patch to allow bad-flagged data from 5904468 from May 23, 2017 (#84) to May 8, 2018 (#118),
        # FIXME:    a period of approximately linear drift following the GDAC-corrected data for #84 and prior profiles
        if wmoid == 5904468:
            if this_float_meta['prof_nums'][prof_index] <= 83:
                pass # these profiles should be delayed mode and corrected by GDAC already
            if this_float_meta['prof_nums'][prof_index] >= 84:
                if gdac_data_mode == 'D':
                    # when profiles #84 and beyond become 'D', this code and correction must be reevaluated
                    raise RuntimeError('IMPORTANT: reevaluate ldp.argo_float_data() correction for 5904468')
                else: # still 'R'
                    if use_unadjusted:
                        gdac_qc_mask = tile(True,len(gdac_psal_qc))  # accept without correction
                        pass
                    elif use_unadjusted is False:
                        if not correct_5904468_interim:
                            # accept; this situation should only occur during calibration routine in main script
                            gdac_qc_mask = tile(True,len(gdac_psal_qc))
                        if correct_5904468_interim and this_float_meta['prof_nums'][prof_index] >= 119:
                            continue # reject; a correction scheme hasn't been developed for profiles 119 and onwards yet
                        if correct_5904468_interim and this_float_meta['prof_nums'][prof_index] <= 118:
                            gdac_qc_mask = tile(True,len(gdac_psal_qc)) # accept; use manual correction below...

        # reject profile if T, S, and/or P had completely bad QC
        if sum(gdac_qc_mask) <= 2:
            if verbose:
                print('this profile has completely or almost completely bad QC')
            continue

        # manual blacklist
        if wmoid == 7900123 and str(this_float_meta['prof_datetimes'][prof_index])[:8] == '20071229':
            continue   # why? big cold, fresh bias - could be real but more likely bad data, since it's December
        if wmoid == 7900407 and this_float_meta['prof_nums'][prof_index] in [3,4,5,*range(59,78+1),114]:
            continue   # why? position jumps, date jumps
        if wmoid == 7900343:
            continue   # why? hugely anomalous surface salinity (+0.5 psu), other QC issues

        # only save first profile taken on a given day; reject additional profiles
        # note: this is a common issue with the early German floats (7900***), which probably recorded date
        #       (usually January) of transmission of all of past winter's under-ice profiles
        #       so, for these 7900*** floats, I am discarding the first profile in these series, too
        if int(wmoid/1000) == 7900:
            if len(this_float_meta['prof_datetimes']) > prof_index+1:
                if int(this_float_meta['prof_datetimes'][prof_index]/1000000) \
                        == int(this_float_meta['prof_datetimes'][prof_index+1]/1000000):
                    continue
        if prof_index > 0:
            if int(this_float_meta['prof_datetimes'][prof_index]/1000000) \
                    == int(this_float_meta['prof_datetimes'][prof_index-1]/1000000):
                continue

        # save profile metadata
        this_float_data['profiles'].append({})
        good_prof_index = len(this_float_data['profiles']) - 1
        this_float_data['profiles'][good_prof_index]['position_flag'] = gdac_prof_position_flag
        this_float_data['profiles'][good_prof_index]['prof_num'] = this_float_meta['prof_nums'][prof_index]
        this_float_data['profiles'][good_prof_index]['prof_num_full'] = this_float_meta['prof_nums_full'][prof_index]
        this_float_data['profiles'][good_prof_index]['datetime'] = this_float_meta['prof_datetimes'][prof_index]
        this_float_data['profiles'][good_prof_index]['status'] = gdac_data_mode
        this_float_data['profiles'][good_prof_index]['lat'] = this_float_meta['prof_lats'][prof_index]
        this_float_data['profiles'][good_prof_index]['lon'] = this_float_meta['prof_lons'][prof_index]

        # # interim lat/lon patch to allow missing-location profiles from 5904468
        # # (keep as example for other floats; edit 'if gdac_prof_position_flag is 9' statement above if using this)
        # if wmoid == 5904468 and this_float_data['profiles'][good_prof_index]['position_flag'] == 9:
        #     this_float_data['profiles'][good_prof_index]['position_flag'] = 2
        #     this_float_data['profiles'][good_prof_index]['lat'] = this_float_meta['prof_lats'][95]
        #     this_float_data['profiles'][good_prof_index]['lon'] = this_float_meta['prof_lons'][95]

        # use T, S, P to derive related data fields and save everything
        gdac_pres = gdac_prof_file.variables['PRES' + var_suffix][0][gdac_qc_mask]
        gdac_temp = gdac_prof_file.variables['TEMP' + var_suffix][0][gdac_qc_mask]
        gdac_psal = gdac_prof_file.variables['PSAL' + var_suffix][0][gdac_qc_mask]

        # FIXME: interim custom linear salinity drift correction for 5904468 (see above)
        if correct_5904468_interim is True and use_unadjusted is False \
                    and wmoid == 5904468 and 84 <= this_float_meta['prof_nums'][prof_index] <= 118:
            # update these filepaths as necessary
            data_dir = os.getcwd() + '/Data/'
            argo_gdac_dir = data_dir + 'Argo/'
            argo_index_pickle_dir = argo_gdac_dir + 'Argo_index_pickles/'

            [cal_prof_nums,cal_deltas] = pickle.load(open(argo_index_pickle_dir + 'argo_5904468_cal.pickle','rb'))
            cal_delta_this_prof = cal_deltas[where(cal_prof_nums == this_float_meta['prof_nums'][prof_index])[0][0]]
            gdac_psal = gdac_psal + cal_delta_this_prof
            # print('profile num and cal delta: ',this_float_meta['prof_nums'][prof_index],cal_delta_this_prof)

        if smooth_sal:
            # note: 's' is a smoothing factor (s=0 is no smoothing)
            # s=0.00010 and below is insufficient for smoothing over the spike around 985 m, caused by switch
            #               from spot (single) to continuous sampling in some floats (per Annie Wong)
            # s=0.00015 was determined visually to give best results over most of the water column,
            #               and successfully mitigates this spike
            psal_spline_interpolant = spin.UnivariateSpline(gdac_pres,gdac_psal,k=2,s=0.00015)
            gdac_psal = psal_spline_interpolant(gdac_pres)
        gdac_depth = -1 * gsw.z_from_p(gdac_pres, this_float_data['profiles'][good_prof_index]['lat'])
        gdac_asal = gsw.SA_from_SP(gdac_psal,gdac_pres,this_float_data['profiles'][good_prof_index]['lon'],
                                   this_float_data['profiles'][good_prof_index]['lat'])
        gdac_ptmp = gsw.pt0_from_t(gdac_asal,gdac_temp,gdac_pres)
        gdac_ctmp = gsw.CT_from_pt(gdac_asal,gdac_ptmp)
        gdac_sigma_theta = gsw.sigma0(gdac_asal,gdac_ctmp)
        this_float_data['profiles'][good_prof_index]['sigma_theta'] = {}
        this_float_data['profiles'][good_prof_index]['sigma_theta']['data'] = gdac_sigma_theta
        this_float_data['profiles'][good_prof_index]['sigma_theta']['name'] = r'$\sigma_\theta$'
        this_float_data['profiles'][good_prof_index]['sigma_theta']['units'] = r'kg/m$^3$'
        this_float_data['profiles'][good_prof_index]['sigma_theta']['pres'] = gdac_pres
        this_float_data['profiles'][good_prof_index]['sigma_theta']['depth'] = gdac_depth
        this_float_data['profiles'][good_prof_index]['temp'] = {}
        this_float_data['profiles'][good_prof_index]['temp']['data'] = gdac_temp
        this_float_data['profiles'][good_prof_index]['temp']['name'] = 'Temperature'
        this_float_data['profiles'][good_prof_index]['temp']['units'] = '°C'
        this_float_data['profiles'][good_prof_index]['temp']['pres'] = gdac_pres
        this_float_data['profiles'][good_prof_index]['temp']['depth'] = gdac_depth
        this_float_data['profiles'][good_prof_index]['ptmp'] = {}
        this_float_data['profiles'][good_prof_index]['ptmp']['data'] = gdac_ptmp
        this_float_data['profiles'][good_prof_index]['ptmp']['name'] = 'Potential temperature'  # r'$\Theta$'
        this_float_data['profiles'][good_prof_index]['ptmp']['units'] = '°C'
        this_float_data['profiles'][good_prof_index]['ptmp']['pres'] = gdac_pres
        this_float_data['profiles'][good_prof_index]['ptmp']['depth'] = gdac_depth
        this_float_data['profiles'][good_prof_index]['ctmp'] = {}
        this_float_data['profiles'][good_prof_index]['ctmp']['data'] = gdac_ctmp
        this_float_data['profiles'][good_prof_index]['ctmp']['name'] = 'Conservative temperature'
        this_float_data['profiles'][good_prof_index]['ctmp']['units'] = '°C'
        this_float_data['profiles'][good_prof_index]['ctmp']['pres'] = gdac_pres
        this_float_data['profiles'][good_prof_index]['ctmp']['depth'] = gdac_depth
        this_float_data['profiles'][good_prof_index]['psal'] = {}
        this_float_data['profiles'][good_prof_index]['psal']['data'] = gdac_psal
        this_float_data['profiles'][good_prof_index]['psal']['name'] = 'Salinity'
        this_float_data['profiles'][good_prof_index]['psal']['units'] = 'PSS-78'
        this_float_data['profiles'][good_prof_index]['psal']['pres'] = gdac_pres
        this_float_data['profiles'][good_prof_index]['psal']['depth'] = gdac_depth
        this_float_data['profiles'][good_prof_index]['asal'] = {}
        this_float_data['profiles'][good_prof_index]['asal']['data'] = gdac_asal
        this_float_data['profiles'][good_prof_index]['asal']['name'] = 'Absolute salinity'
        this_float_data['profiles'][good_prof_index]['asal']['units'] = 'g/kg'
        this_float_data['profiles'][good_prof_index]['asal']['pres'] = gdac_pres
        this_float_data['profiles'][good_prof_index]['asal']['depth'] = gdac_depth

        if compute_extras:
            # compute N^2
            gdac_Nsquared, gdac_midpoint_pres = gsw.Nsquared(gdac_asal,gdac_ctmp,gdac_pres,
                                                             this_float_data['profiles'][good_prof_index]['lat'])
            gdac_midpoint_depth = -1 * gsw.z_from_p(gdac_midpoint_pres,
                                                    this_float_data['profiles'][good_prof_index]['lat'])
            this_float_data['profiles'][good_prof_index]['Nsquared'] = {}
            this_float_data['profiles'][good_prof_index]['Nsquared']['data'] = gdac_Nsquared * 10e7
            this_float_data['profiles'][good_prof_index]['Nsquared']['name'] = 'Buoyancy frequency squared'
            this_float_data['profiles'][good_prof_index]['Nsquared']['units'] = r'10$^{-7}$ s$^{-2}$'
            this_float_data['profiles'][good_prof_index]['Nsquared']['pres'] = gdac_midpoint_pres
            this_float_data['profiles'][good_prof_index]['Nsquared']['depth'] = gdac_midpoint_depth
            if smooth_N2_PV:
                smooth_N2_depths, smooth_N2 = gt.vert_prof_running_mean(this_float_data['profiles'][good_prof_index],
                                                                        'Nsquared',window=smooth_N2_PV_window,
                                                                        extrap='NaN',top='top',bottom='bottom')
                smooth_N2_pres = gsw.p_from_z(-1 * smooth_N2_depths,this_float_data['profiles'][good_prof_index]['lat'])
                this_float_data['profiles'][good_prof_index]['Nsquared']['data'] = smooth_N2
                this_float_data['profiles'][good_prof_index]['Nsquared']['pres'] = smooth_N2_pres
                this_float_data['profiles'][good_prof_index]['Nsquared']['depth'] = smooth_N2_depths

            # compute PV (isopycnic potential vorticity, which neglects contribution of relative vorticity)
            # per Talley et al. (2011), Ch. 3.5.6 (Eq. 3.12b): Q ~ -(f/g)*(N^2)
            omega = 7.2921e-5   # rad/s
            coriolis_freq = 2 * omega * sin(this_float_data['profiles'][good_prof_index]['lat'] * pi/180)
            gdac_PV = -1 * (coriolis_freq / constants.g) * gdac_Nsquared
            this_float_data['profiles'][good_prof_index]['PV'] = {}
            this_float_data['profiles'][good_prof_index]['PV']['data'] = gdac_PV * 10e12
            this_float_data['profiles'][good_prof_index]['PV']['name'] = 'Isopycnic potential vorticity'
            this_float_data['profiles'][good_prof_index]['PV']['units'] = r'10$^{-12}$ m$^{-1}$ s$^{-1}$'
            this_float_data['profiles'][good_prof_index]['PV']['pres'] = gdac_midpoint_pres
            this_float_data['profiles'][good_prof_index]['PV']['depth'] = gdac_midpoint_depth
            if smooth_N2_PV:
                smooth_PV_depths, smooth_PV = gt.vert_prof_running_mean(this_float_data['profiles'][good_prof_index],
                                                                        'PV',window=smooth_N2_PV_window,
                                                                        extrap='NaN',top='top',bottom='bottom')
                smooth_PV_pres = gsw.p_from_z(-1 * smooth_PV_depths,this_float_data['profiles'][good_prof_index]['lat'])
                this_float_data['profiles'][good_prof_index]['PV']['data'] = smooth_PV
                this_float_data['profiles'][good_prof_index]['PV']['pres'] = smooth_PV_pres
                this_float_data['profiles'][good_prof_index]['PV']['depth'] = smooth_PV_depths

            # compute buoyancy loss required for convection to reach each observed depth, as in de Lavergne et al. (2014)
            # see gt.destab() for details
            gdac_destab = gt.destab(this_float_data['profiles'][good_prof_index],gdac_depth[1:],verbose_warn=True)
            this_float_data['profiles'][good_prof_index]['destab'] = {}
            this_float_data['profiles'][good_prof_index]['destab']['data'] = gdac_destab
            this_float_data['profiles'][good_prof_index]['destab']['name'] = 'Buoyancy loss required for destabilization to given depth'
            this_float_data['profiles'][good_prof_index]['destab']['units'] = r'm$^2$ s$^{-2}$'
            this_float_data['profiles'][good_prof_index]['destab']['pres'] = gdac_pres[1:]
            this_float_data['profiles'][good_prof_index]['destab']['depth'] = gdac_depth[1:]

    # load SOCCOM data and merge with GDAC profiles
    # TODO: modify to include SOCCOM profiles even when no matching GDAC profile (or when GDAC temp/salinity is bad QC)
    if wmoid in argo_soccom_index['wmoids']:
        soccom_index_number = argo_soccom_index['wmoids'].index(wmoid)
        this_float_data['is_soccom'] = True
        this_float_data['uwid'] = argo_soccom_index['uwids'][soccom_index_number]
        soccom_filename = argo_soccom_index['filenames'][soccom_index_number]
        header_line_counter = 0
        with open(save_to_soccom_dir + soccom_filename,'rb') as f:
            for line in f:
                if 'Cruise	' in line.decode('latin-1'):
                    break
                header_line_counter += 1
        data_frame = pd.read_csv(save_to_soccom_dir + soccom_filename, header=header_line_counter,
                                 delim_whitespace=True, na_values=-1e10, encoding='latin-1')
        soccom_var_names = data_frame.axes[1].values
        soccom_prof_nums = argo_soccom_index['profile_nums'][soccom_index_number]
        soccom_all_entries_prof_nums = data_frame['Station'].values
        for gdac_prof_index, gdac_profile_list_of_dicts in enumerate(this_float_data['profiles']):
            this_prof_num = this_float_data['profiles'][gdac_prof_index]['prof_num']
            if this_prof_num in soccom_prof_nums:
                soccom_all_entries_match_prof = (soccom_all_entries_prof_nums == this_prof_num)
                # ignore blank last line-entry of profile
                soccom_all_entries_match_prof[where(soccom_all_entries_match_prof)[0][-1]] = False
                soccom_pres = data_frame['Pressure[dbar]'].values[soccom_all_entries_match_prof]
                soccom_depth = data_frame['Depth[m]'].values[soccom_all_entries_match_prof]
                for param_string in soccom_params_to_save:
                    if param_string in soccom_var_names:
                        soccom_param_data = data_frame[param_string].values[soccom_all_entries_match_prof]
                        soccom_param_QC_str = soccom_var_names[1 + where(soccom_var_names == param_string)[0][0]]
                        soccom_param_QC = data_frame[soccom_param_QC_str].values[soccom_all_entries_match_prof].astype(int)
                        if allow_bad_soccom_qc:
                            soccom_QC_mask = logical_or(soccom_param_QC == 0,soccom_param_QC == 4)
                        else:
                            soccom_QC_mask = (soccom_param_QC == 0)

                        soccom_params_index = where(soccom_params_to_save == param_string)[0][0]
                        soccom_param_name = soccom_param_names[soccom_params_index]
                        soccom_param_abbrev = soccom_param_abbrevs[soccom_params_index]
                        soccom_param_units = soccom_units_names[soccom_params_index]
                        this_float_data['profiles'][gdac_prof_index][soccom_param_abbrev] = {}
                        this_float_data['profiles'][gdac_prof_index][soccom_param_abbrev]['data'] \
                            = soccom_param_data[soccom_QC_mask][::-1]
                        this_float_data['profiles'][gdac_prof_index][soccom_param_abbrev]['name'] = soccom_param_name
                        this_float_data['profiles'][gdac_prof_index][soccom_param_abbrev]['units'] = soccom_param_units
                        this_float_data['profiles'][gdac_prof_index][soccom_param_abbrev]['pres'] \
                            = soccom_pres[soccom_QC_mask][::-1]
                        this_float_data['profiles'][gdac_prof_index][soccom_param_abbrev]['depth'] \
                            = soccom_depth[soccom_QC_mask][::-1]
                        # re-compute oxygen saturation using quality controlled T and S from GDAC
                        # note: O2 solubility converted from ml/L to µmol/kg
                        if soccom_param_abbrev == 'OxygenSat':
                            assert 'Oxygen' in this_float_data['profiles'][gdac_prof_index].keys(), 'No O2 data found.'
                            soccom_pres_for_interp = this_float_data['profiles'][gdac_prof_index]['OxygenSat']['pres']
                            gdac_psal_interp = gt.vert_prof_eval(this_float_data['profiles'][gdac_prof_index],'psal',
                                                                 soccom_pres_for_interp,z_coor='pres')
                            gdac_temp_interp = gt.vert_prof_eval(this_float_data['profiles'][gdac_prof_index],'temp',
                                                                 soccom_pres_for_interp,z_coor='pres')
                            soccom_O2_sol = 44.6596 * seawater.satO2(gdac_psal_interp,gdac_temp_interp)
                            soccom_O2_sat = 100.0 * this_float_data['profiles'][gdac_prof_index]['Oxygen']['data'] \
                                            / soccom_O2_sol
                            this_float_data['profiles'][gdac_prof_index]['OxygenSat']['data'] = soccom_O2_sat
    else:
        this_float_data['is_soccom'] = False
        this_float_data['uwids'] = 'unknown_or_not_applicable'

    # load UW O2 data and merge with GDAC profiles
    if wmoid in argo_soccom_index['UW_O2_wmoids']:
        UW_O2_index_number = argo_soccom_index['UW_O2_wmoids'].index(wmoid)
        UW_O2_filename = argo_soccom_index['UW_O2_filenames'][UW_O2_index_number]
        with h5py.File(save_to_UW_O2_dir + UW_O2_filename,'r') as UW_O2_file:
            UW_O2_prof_nums = UW_O2_file['PROFILE'].value[0][0].astype(int)
            UW_O2_pres = UW_O2_file['PRES'].value[0]
            UW_O2_data = UW_O2_file['OXYGEN_UW_CORRECTED'].value[0]

            # manual quality control for 5903616 based on offset of O2 estimated in density space
            if wmoid == 5903616: UW_O2_data = UW_O2_data - 6.0

        for gdac_prof_index in range(len(this_float_data['profiles'])):
            this_prof_num = this_float_data['profiles'][gdac_prof_index]['prof_num']
            if use_UW_O2_not_SOCCOM: load_UW_O2_if_available = True
            else:                    load_UW_O2_if_available = ('Oxygen' not in
                                                                this_float_data['profiles'][gdac_prof_index].keys())
            if load_UW_O2_if_available and (this_prof_num in UW_O2_prof_nums):
                UW_prof_index = list(UW_O2_prof_nums).index(this_prof_num)
                if not all(isnan(UW_O2_data[:,UW_prof_index])):
                    good_z = ~isnan(UW_O2_pres[:,UW_prof_index])
                    unique_indices = unique(UW_O2_pres[:,UW_prof_index],return_index=True)[1]
                    for good_or_not_idx, good_or_not in enumerate(good_z):
                        if good_or_not_idx not in unique_indices: good_z[good_or_not_idx] = False
                    UW_O2_depth = -1 * gsw.z_from_p(UW_O2_pres[:,UW_prof_index][good_z],
                                                    this_float_data['profiles'][gdac_prof_index]['lat'])
                    # unfortunately no O2 solubility routine in GSW-Python yet...
                    # note: O2 solubility converted from ml/L to µmol/kg
                    UW_O2_psal = gt.vert_prof_eval(this_float_data['profiles'][gdac_prof_index],'psal',
                                                   UW_O2_pres[:,UW_prof_index][good_z],z_coor='pres')
                    UW_O2_temp = gt.vert_prof_eval(this_float_data['profiles'][gdac_prof_index],'temp',
                                                   UW_O2_pres[:,UW_prof_index][good_z],z_coor='pres')
                    UW_O2_sol = 44.6596 * seawater.satO2(UW_O2_psal,UW_O2_temp)
                    assert len(UW_O2_sol) == len(UW_O2_data[:,UW_prof_index][good_z]), 'Check UW-O2 data/sol vector lengths.'
                    UW_O2_sat = 100.0 * UW_O2_data[:,UW_prof_index][good_z] / UW_O2_sol
                    this_float_data['profiles'][gdac_prof_index]['Oxygen'] = {}
                    this_float_data['profiles'][gdac_prof_index]['Oxygen']['data'] = UW_O2_data[:,UW_prof_index][good_z]
                    this_float_data['profiles'][gdac_prof_index]['Oxygen']['name'] = 'Oxygen'
                    this_float_data['profiles'][gdac_prof_index]['Oxygen']['units'] = 'µmol/kg'
                    this_float_data['profiles'][gdac_prof_index]['Oxygen']['pres'] = UW_O2_pres[:,UW_prof_index][good_z]
                    this_float_data['profiles'][gdac_prof_index]['Oxygen']['depth'] = UW_O2_depth
                    this_float_data['profiles'][gdac_prof_index]['OxygenSat'] = {}
                    this_float_data['profiles'][gdac_prof_index]['OxygenSat']['data'] = UW_O2_sat
                    this_float_data['profiles'][gdac_prof_index]['OxygenSat']['name'] = 'Oxygen saturation'
                    this_float_data['profiles'][gdac_prof_index]['OxygenSat']['units'] = '%'
                    this_float_data['profiles'][gdac_prof_index]['OxygenSat']['pres'] = UW_O2_pres[:,UW_prof_index][good_z]
                    this_float_data['profiles'][gdac_prof_index]['OxygenSat']['depth'] = UW_O2_depth
                    this_float_data['is_uw_o2'] = True

    return this_float_data


def wod_load_index(wod_dir,data_dirs):
    """ Parse index of WOD cast data in netCDF format.

    Args:
        data_dirs: list of strings specifying data directory locations to examine

    Returns:
        wod_index: dict with following keys to NumPy arrays of equal length
                        'filepaths' (complete paths to data files)
                        'datetimes' (Datetime objects)
                        'lats' and 'lons'

    Data provenance:
        NCEI/NODC WODselect utility: https://www.nodc.noaa.gov/OC5/SELECT/dbsearch/dbsearch.html
        Information on updates every 3 months: https://www.nodc.noaa.gov/OC5/WOD/wod_updates.html
        Note on prelease of WOD 2018 data: https://www.nodc.noaa.gov/OC5/WOD/wod18-notes.html

    Acknowledgements:
        see https://www.nodc.noaa.gov/OC5/wod-woa-faqs.html

    """
    wod_index = {'filepaths':array([]), 'datetimes':array([]), 'lats':array([]), 'lons':array([])}
    index_file_prefix = 'ocldb'
    for data_subdir in data_dirs:
        all_filenames = os.listdir(wod_dir + data_subdir)
        for filename in all_filenames:
            if filename[0:5] == index_file_prefix:
                index_file = spnc.netcdf_file(wod_dir + data_subdir + '/' + filename,'r',mmap=False)
                valid_time_mask = index_file.variables['time'][:] >= 0
                wod_index['lats'] = append(wod_index['lats'],index_file.variables['lat'][valid_time_mask])
                wod_index['lons'] = append(wod_index['lons'],index_file.variables['lon'][valid_time_mask])
                wod_index['datetimes'] = append(wod_index['datetimes'],
                                                array([tt.convert_days_since_ref_to_datetime(days_since,1770,1,1)
                                                       for days_since in index_file.variables['time'][valid_time_mask]]))
                wod_index['filepaths'] = append(wod_index['filepaths'],
                                                array([wod_dir + data_subdir + '/' + 'wod_{0:09}O.nc'.format(cast_num)
                                                       for cast_num in index_file.variables['cast'][valid_time_mask]]))
                break
            else:
                continue
    return wod_index


def wod_load_cast(filepath):
    """ Accessor method for WOD cast observed data in netCDF format. See ldp.wod_load_index() for details.
        QC flag handling: returns strictly good data from good profiles.

    Args:
        filepath: string representing full filepath of data file

    Returns:
        None if cast contained bad depth, temperature, and/or salinity profiles
        (or)
        cast_data: dict with keys:
            country: country in all caps (string)
            platform: long description of ship, for instance (string)
            cast_num: unique WOD cast number (integer)
            lat: float
            lon: float
            datetime: Datetime object
            cast: dict with keys: temp, ptmp, ctmp, psal, asal, sigma_theta
                each is a dict with keys:
                    data: array of measured values, in direction of increasing depth (surface downwards)
                    name: string of name of parameter, formatted for a plot axis
                    pres: array of corresponding pressure (dbar)
                    depth: array of corresponding depths (m, positive)
                    units: string of units of data

    Example:
        cast_data['datetime'] = datetime(2017,1,1)
        cast_data['cast']['sigma_theta']['data'] = ...
        cast_data['cast']['sigma_theta']['pres'] = ...

    """
    data_file = spnc.netcdf_file(filepath,'r',mmap=False)
    cast_data = dict()
    cast_data['country'] = b''.join(data_file.variables['country'][:]).decode('utf-8')
    if 'Platform' in data_file.variables.keys():
        cast_data['platform'] = b''.join(data_file.variables['Platform'][:]).decode('utf-8')
    else:
        cast_data['platform'] = 'unknown'
    cast_data['cast_num'] = int(data_file.variables['wod_unique_cast'].data)
    cast_data['lat'] = float(data_file.variables['lat'].data)
    cast_data['lon'] = float(data_file.variables['lon'].data)
    cast_data['datetime'] = tt.convert_days_since_ref_to_datetime(float(data_file.variables['time'].data),1770,1,1)

    cast_data['cast'] = dict()
    depth = data_file.variables['z'][:]
    depth_good_qc_mask = data_file.variables['z_WODflag'][:] == 0
    if all(~depth_good_qc_mask): return None  # if all depths are bad
    if 'Temperature' in data_file.variables.keys():
        temp_qc_flag = int(data_file.variables['Temperature_WODprofileflag'].data)
        if temp_qc_flag == 0:
            temp = data_file.variables['Temperature'][:]
            temp_good_qc_mask = data_file.variables['Temperature_WODflag'][:] == 0
            temp_good_qc_mask = logical_and(temp_good_qc_mask, temp > -2.5)
            cast_data['cast']['temp'] = dict()
        else: return None
    else: return None
    if 'Salinity' in data_file.variables.keys():
        psal_qc_flag = int(data_file.variables['Salinity_WODprofileflag'].data)
        if psal_qc_flag == 0:
            psal = data_file.variables['Salinity'][:]
            psal_good_qc_mask = data_file.variables['Salinity_WODflag'][:] == 0
            psal_good_qc_mask = logical_and(psal_good_qc_mask, psal > 0.0)
            cast_data['cast']['psal'] = dict()
        else: return None
    else: return None

    cast_qc_mask = logical_and(depth_good_qc_mask,logical_and(temp_good_qc_mask,psal_good_qc_mask))
    if all(~cast_qc_mask): return None

    depth = depth[cast_qc_mask]
    temp = temp[cast_qc_mask]
    psal = psal[cast_qc_mask]

    pres = gsw.p_from_z(-1 * depth,cast_data['lat'])
    asal = gsw.SA_from_SP(psal,pres,cast_data['lon'],cast_data['lat'])
    ptmp = gsw.pt0_from_t(asal,temp,pres)
    ctmp = gsw.CT_from_pt(asal,ptmp)
    sigma_theta = gsw.sigma0(asal,ctmp)

    cast_data['cast']['sigma_theta'] = {}
    cast_data['cast']['sigma_theta']['data'] = sigma_theta
    cast_data['cast']['sigma_theta']['name'] = r'$\sigma_\theta$'
    cast_data['cast']['sigma_theta']['units'] = r'kg/m$^3$'
    cast_data['cast']['sigma_theta']['pres'] = pres
    cast_data['cast']['sigma_theta']['depth'] = depth
    cast_data['cast']['temp'] = {}
    cast_data['cast']['temp']['data'] = temp
    cast_data['cast']['temp']['name'] = 'Temperature'
    cast_data['cast']['temp']['units'] = '°C'
    cast_data['cast']['temp']['pres'] = pres
    cast_data['cast']['temp']['depth'] = depth
    cast_data['cast']['ptmp'] = {}
    cast_data['cast']['ptmp']['data'] = ptmp
    cast_data['cast']['ptmp']['name'] = 'Potential temperature'  # r'$\Theta$'
    cast_data['cast']['ptmp']['units'] = '°C'
    cast_data['cast']['ptmp']['pres'] = pres
    cast_data['cast']['ptmp']['depth'] = depth
    cast_data['cast']['ctmp'] = {}
    cast_data['cast']['ctmp']['data'] = ctmp
    cast_data['cast']['ctmp']['name'] = 'Conservative temperature'
    cast_data['cast']['ctmp']['units'] = '°C'
    cast_data['cast']['ctmp']['pres'] = pres
    cast_data['cast']['ctmp']['depth'] = depth
    cast_data['cast']['psal'] = {}
    cast_data['cast']['psal']['data'] = psal
    cast_data['cast']['psal']['name'] = 'Salinity'
    cast_data['cast']['psal']['units'] = 'PSS-78'
    cast_data['cast']['psal']['pres'] = pres
    cast_data['cast']['psal']['depth'] = depth
    cast_data['cast']['asal'] = {}
    cast_data['cast']['asal']['data'] = asal
    cast_data['cast']['asal']['name'] = 'Absolute salinity'
    cast_data['cast']['asal']['units'] = 'g/kg'
    cast_data['cast']['asal']['pres'] = pres
    cast_data['cast']['asal']['depth'] = depth

    return cast_data


def waghc_load_field(subdir):
    """ Use xarray to load WOCE/Argo Global Hydrographic Climatology (WAGHC) 2017 fields in netCDF format.

    Information: http://icdc.cen.uni-hamburg.de/1/daten/ocean/waghc/

    Citation: Gouretski and Koltermann (2004) as well as the following:

    Gouretski, Viktor (2018). WOCE-Argo Global Hydrographic Climatology (WAGHC Version 1.0). World Data Center for
        Climate (WDCC) at DKRZ. https://doi.org/10.1594/WDCC/WAGHC_V1.0

    Example plot:
        plt.pcolormesh(fields['salinity'].sel(time=3,depth=15.0,latitude=slice(-70,-55),longitude=slice(-20,20)))

    """
    fields = xr.open_mfdataset(subdir + '*.nc',decode_times=False)
    return fields


def waghc_interp_to_location(waghc_fields,param,depth,lon,lat,month_exact=None,datetime_for_interp=None):
    """ Use nearest neighbor interpolation to estimate WAGHC value at a given (lat,lon).

    Args:
        waghc_fields: see ldp.waghc_load_field()
        param: 'salinity' or 'psal' or 'temperature' or 'temp'
        depth: integer or float depth
        lat: latitude to interpolate to
        lon: longitude to interpolate to
        month_exact: integer month (1-12) to use that month's field, or None to interpolate between months
        datetime_for_interp: Datetime to use to interpolate between months, or None to use specific month's field

    Returns:
        val: interpolated value

    """
    if   param == 'psal': param = 'salinity'
    elif param == 'temp': param = 'temperature'
    if month_exact is not None:
        return float(waghc_fields[param].sel(time=month_exact,depth=depth,latitude=lat,longitude=lon,method='nearest'))
    else:
        climo_vector = waghc_fields[param].sel(depth=depth,latitude=lat,longitude=lon,method='nearest').values
        climo_vector_padded = array([climo_vector[-1],*climo_vector,climo_vector[0]])
        climo_vector_doys = arange(-365.24/12/2,365+30,365.24/12)  # somewhat imprecise, but good enough
        interpolator = spin.interp1d(climo_vector_doys,climo_vector_padded)
        return float(interpolator(datetime_for_interp.timetuple().tm_yday))


def compile_hydrographic_obs(argo_index_pickle_dir,argo_gdac_dir,wod_dir,
                             wod_ship_dirs=['WOD_CTD_Weddell','WOD_OSD_Weddell'],
                             wod_seal_dirs=['WOD_APB_Weddell'],
                             lon_bounds=[-60,15],lat_bounds=[-90,-55],toi_bounds=[datetime(1970,1,1),datetime.today()],
                             distance_check=None,distance_center=[None,None],
                             include_argo=True,include_wod=True,params=['ptmp','psal','sigma_theta'],
                             compute_extras=False,max_cast_min_depth=20,min_cast_max_depth=1000,
                             reject_mld_below=500,reject_mld_ref=False,strict_mld_reject=True,
                             interp_spacing=0.1,interp_depths=(0,1000),calc_mld=True,calc_ml_avg=True,
                             calc_at_depths=None,calc_depth_avgs=None,calc_sd=None,calc_tb=None,
                             pickle_dir=None,pickle_filename=None,
                             prof_count_dir=None,prof_count_filename=None,verbose=True):
    """ Compile regularly-interpolated Argo and/or WOD hydrographic profiles and calculate quantities of interest.
        Save in pickle if requested.

    Args:
        lon_bounds: longitude range [W,E] to search for profiles/casts
        lat_bounds: latitude range [S,N] to search for profiles/casts
        toi_bounds: Datetime range [start,end] to search for profiles/casts
        distance_check: None or maximum distance (in km) from <<distance_center>> as requirement for profiles/casts
                             or [smaller_radius,larger_radius] to specify toroid (donut) to search within
            note: must still specify lon_bounds, lat_bounds to whittle down search
        distance_center: [lat,lon] of center location from which to check distance
        include_argo: True or False
        include_wod: True or False
        params: list of parameter abbreviations to include
            note: data for all params listed here must be available for a profile to be examined
        compute_extras: False or True (compute N2, convection resistance, etc.) for float profiles
        max_cast_min_depth: None or deepest minimum depth of cast to allow (if deeper, ignore)
        min_cast_max_depth: None or shallowest maximum depth of cast to allow (if shallower, ignore)
        reject_mld_below: None or depth (set MLD to NaN, or ignore cast, if calculated MLD is below given depth; this
                          behavior depends on strict_mld_reject)
            note: only tested if not None AND calc_mld is True
        reject_mld_ref: False (default)
                     or True (recommended; set MLD to NaN if shallowest measurement is below reference depth)
            note: only tested if calc_mld is True
        strict_mld_reject: True (default; ignore cast if MLD is NaN, None, or greater than reject_mld_below)
                        or False (recommended; keep cast if these conditions met, but don't calculate ML averages)
        interp_spacing: vertical depth spacing (meters) of interpolated profiles returned ('nearest' extrap used)
        interp_depths: None or depth range of interpolated profiles to return: (shallow,deep)
        calc_mld: True or False
        calc_ml_avg: False or True (to calculate parameter averages within ML) (calc_mld must be True)
        calc_at_depths: None or list of:
                 single depths to calculate parameter values at using interpolation
                 tuples of depths, all of which must have valid calculated parameter values using interpolation
                    NOTE: these tuples must contain at least 3 depths
        calc_depth_avgs: None or LIST (!) of depth ranges (shallow,deep) to calculate parameter averages by interp
        calc_sd: None or list of depths for calculating Martinson salt deficit (see geo_tools.martinson())
                 if single depths provided, calculate original 0-[depth] metrics
                 if tuple of depths provided, e.g. (upper,lower), calculate interior upper-lower metrics
                 if three-tuple provided, e.g. (upper,lower,sd_ref_psal), calculate interior metrics with ref_psal
        calc_tb: None or list of depths for calculating Martinson thermal barrier (")
                 if single depths provided, calculate original 0-[depth] metrics
                 if tuple of depths provided, e.g. (upper,lower), calculate interior upper-lower metrics
        pickle_dir: None (simply return data found) or filepath to store pickle
        pickle_filename: None (") or name of pickle in which to save data
        prof_count_dir: None or filepath to save text file with profile and float counts
        prof_count_filename: None or filename for text file with profile and float counts
        verbose: True or False

    Returns:
        compiled_obs: dict with following keys:
            'platforms': NumPy vector of all obs' WMOids (for floats) or ship name / other platform ID (for WOD)
            'types': NumPy vector of all obs' type: 'ship', 'float', or 'seal'
            'datetimes': NumPy vector of all obs' datetimes
            'doys': NumPy vector of all obs' dates-of-year
            'lats': NumPy vector of all obs' latitudes
            'lons': NumPy vector of all obs' longitudes
            'mlds': NumPy vector of all obs' MLDs (if calc_mld is True)
            'depths': NumPy vector of evenly spaced depths used in profile interpolation
            each parameter abbreviation in <<params>>: dicts with following keys for each parameter's data:
                'profiles': NumPy 2D array of profiles corresponding to <<depths>> (see above)
                    note: shape is (# profiles, # depths)
                'ml_avg': NumPy vector of profiles' ML averages (if calc_mld and calc_ml_avg are True)
                depths given in calc_at_depths (will automatically truncate trailing zeros):
                    NumPy vector of interpolated values at each depth
                two-tuples corresponding to depth ranges given in calc_depth_avgs:
                    NumPy vector of average values for each depth range
                tuples of two or more depths corresponding to tuples given in calc_at_depths
            'detailed_info': dict with same structure as above (see example) containing
                             lat/lon/type/platform/datetime/doy for each non-NaN derived data
                e.g. compiled_obs['detailed_info']['psal'][15]['lats'] is a list of lats (not a NumPy array)

    """

    compiled_data = {'platforms':[],'types':[],'datetimes':[],'doys':[],'lats':[],'lons':[],'depths':[],
                     'detailed_info':dict()}
    if calc_mld: compiled_data['mlds'] = []

    float_wmoids_used = []
    float_prof_count = 0
    wod_ship_prof_count = 0
    wod_seal_prof_count = 0

    counts = dict()
    def counter(count_dict,param_name,key_name,prof_data_dict):
        if param_name not in count_dict.keys(): count_dict[param_name] = dict()
        if key_name not in count_dict[param_name].keys():
            count_dict[param_name][key_name] \
                = {'float_wmoids_used':[], 'float_profs':0, 'wod_ship_profs':0, 'wod_seal_profs':0}
        if 'seal_platform' in prof_data_dict.keys():
            count_dict[param_name][key_name]['wod_seal_profs'] += 1
        elif 'ship_platform' in prof_data_dict.keys():
            count_dict[param_name][key_name]['wod_ship_profs'] += 1
        elif 'wmoid' in prof_data_dict.keys():
            count_dict[param_name][key_name]['float_profs'] += 1
            count_dict[param_name][key_name]['float_wmoids_used'].append(prof_data_dict['wmoid'])

        # store lat/lon/type/platform corresponding to specific non-rejected, non-NaN derived data
        if param_name not in compiled_data['detailed_info'].keys():
            compiled_data['detailed_info'][param_name] = dict()
        if key_name not in compiled_data['detailed_info'][param_name].keys():
            compiled_data['detailed_info'][param_name][key_name] = dict()
        if 'types' not in compiled_data['detailed_info'][param_name][key_name].keys():
            compiled_data['detailed_info'][param_name][key_name]['platforms'] = []
            compiled_data['detailed_info'][param_name][key_name]['types'] = []
            compiled_data['detailed_info'][param_name][key_name]['datetimes'] = []
            compiled_data['detailed_info'][param_name][key_name]['doys'] = []
            compiled_data['detailed_info'][param_name][key_name]['lats'] = []
            compiled_data['detailed_info'][param_name][key_name]['lons'] = []
        compiled_data['detailed_info'][param_name][key_name]['lats'].append(prof_data['lat'])
        compiled_data['detailed_info'][param_name][key_name]['lons'].append(prof_data['lon'])
        compiled_data['detailed_info'][param_name][key_name]['datetimes'].append(prof_data['datetime'])
        compiled_data['detailed_info'][param_name][key_name]['doys'].append(prof_data['datetime'].timetuple().tm_yday)
        if 'seal_platform' in prof_data.keys():
            compiled_data['detailed_info'][param_name][key_name]['platforms'].append(prof_data['seal_platform'])
            compiled_data['detailed_info'][param_name][key_name]['types'].append('seal')
        elif 'ship_platform' in prof_data.keys():
            compiled_data['detailed_info'][param_name][key_name]['platforms'].append(prof_data['ship_platform'])
            compiled_data['detailed_info'][param_name][key_name]['types'].append('ship')
        elif 'wmoid' in prof_data.keys():
            compiled_data['detailed_info'][param_name][key_name]['platforms'].append(prof_data['wmoid'])
            compiled_data['detailed_info'][param_name][key_name]['types'].append('float')
        else:
            compiled_data['detailed_info'][param_name][key_name]['platforms'].append('Unknown platform')
            compiled_data['detailed_info'][param_name][key_name]['types'].append('unknown')

    def dist_check(obs_lat,obs_lon):
        if distance_check is None:
            return True
        else:
            if isnan(obs_lat) or isnan(obs_lon): return False
            radius = gt.distance_between_two_coors(obs_lat,obs_lon,distance_center[0],distance_center[1])
            if type(distance_check) == int or type(distance_check) == float:
                if (distance_check * 1000.0) <= radius: return False
                else:                                   return True
            elif len(distance_check) == 2:
                if (distance_check[0] * 1000.0) < radius < (distance_check[1] * 1000.0): return True
                else:                                                                    return False

    all_profs = []
    if include_argo:
        argo_gdac_index = pickle.load(open(argo_index_pickle_dir + 'argo_gdac_index.pickle','rb'))
        argo_soccom_index = pickle.load(open(argo_index_pickle_dir + 'argo_soccom_index.pickle','rb'))
        toi_int = [tt.convert_datetime_to_14(toi_bounds[0]),tt.convert_datetime_to_14(toi_bounds[1])]
        all_float_data = []
        for wmoid in argo_gdac_index['wmoids']:
            this_float_meta = argo_gdac_float_meta(argo_gdac_index['local_prof_index'],wmoid)
            toi_match = logical_and(this_float_meta['prof_datetimes'] >= toi_int[0],
                                    this_float_meta['prof_datetimes'] <= toi_int[1])
            lon_match = logical_and(this_float_meta['prof_lons'] >= lon_bounds[0],
                                    this_float_meta['prof_lons'] <= lon_bounds[1])
            lat_match = logical_and(this_float_meta['prof_lats'] >= lat_bounds[0],
                                    this_float_meta['prof_lats'] <= lat_bounds[1])
            prof_match = logical_and(logical_and(toi_match,lon_match),lat_match)
            if any(prof_match):
                if distance_check is not None:
                    for p_idx in range(len(prof_match)):
                        if prof_match[p_idx]:
                            prof_match[p_idx] = dist_check(this_float_meta['prof_lats'][p_idx],
                                                           this_float_meta['prof_lons'][p_idx])
                if any(prof_match):
                    if verbose: print(wmoid)
                    prof_nums_match = array(this_float_meta['prof_nums'])[prof_match]
                    this_float_data = argo_float_data(wmoid,argo_gdac_dir,argo_gdac_index,argo_soccom_index,
                                                      prof_nums=prof_nums_match,compute_extras=compute_extras)
                    all_float_data.append(this_float_data)
        for float_idx,float_data in enumerate(all_float_data):
            for prof_idx in range(len(float_data['profiles'])):
                prof_data = float_data['profiles'][prof_idx]
                prof_data['wmoid'] = float_data['wmoid'] # annoying but necessary
                prof_data['datetime'] = tt.convert_tuple_to_datetime(tt.convert_14_to_tuple(prof_data['datetime']))
                all_profs.append(prof_data)

    if include_wod:
        wod_index_ships = wod_load_index(wod_dir,wod_ship_dirs)
        for cast_idx, filepath in enumerate(wod_index_ships['filepaths']):
            if lat_bounds[0] <= wod_index_ships['lats'][cast_idx] <= lat_bounds[1] \
                    and lon_bounds[0] <= wod_index_ships['lons'][cast_idx] <= lon_bounds[1] \
                    and toi_bounds[0] <= wod_index_ships['datetimes'][cast_idx] <= toi_bounds[1]:
                if dist_check(wod_index_ships['lats'][cast_idx],wod_index_ships['lons'][cast_idx]):
                    cast_data = wod_load_cast(filepath)
                    if cast_data is not None:
                        cast_data['cast']['ship_platform'] = cast_data['platform']
                        cast_data['cast']['lat'] = cast_data['lat']
                        cast_data['cast']['lon'] = cast_data['lon']
                        cast_data['cast']['datetime'] = cast_data['datetime']
                        if verbose: print(cast_data['cast']['ship_platform'])
                        all_profs.append(cast_data['cast'])
        wod_index_seals = wod_load_index(wod_dir,wod_seal_dirs)
        for cast_idx,filepath in enumerate(wod_index_seals['filepaths']):
            if lat_bounds[0] <= wod_index_seals['lats'][cast_idx] <= lat_bounds[1] \
                    and lon_bounds[0] <= wod_index_seals['lons'][cast_idx] <= lon_bounds[1] \
                    and toi_bounds[0] <= wod_index_seals['datetimes'][cast_idx] <= toi_bounds[1]:
                if dist_check(wod_index_seals['lats'][cast_idx],wod_index_seals['lons'][cast_idx]):
                    cast_data = wod_load_cast(filepath)
                    if cast_data is not None:
                        cast_data['cast']['seal_platform'] = cast_data['platform']
                        cast_data['cast']['lat'] = cast_data['lat']
                        cast_data['cast']['lon'] = cast_data['lon']
                        cast_data['cast']['datetime'] = cast_data['datetime']
                        if verbose: print(cast_data['cast']['seal_platform'])
                        all_profs.append(cast_data['cast'])

    for idx, prof_data in enumerate(all_profs):
        if verbose: df.how_far(idx, all_profs, 0.01)  # print progress
        if not all([param_abbrev in prof_data.keys() for param_abbrev in params]):
            continue
        if max_cast_min_depth is not None:
            if prof_data[params[0]]['depth'][0] > max_cast_min_depth:
                if verbose:
                    print('>>> rejected cast/profile with depth range ({0},{1})'
                          ''.format(prof_data[params[0]]['depth'][0],prof_data[params[0]]['depth'][-1]))
                continue
        if min_cast_max_depth is not None:
            if prof_data[params[0]]['depth'][-1] < min_cast_max_depth:
                if verbose:
                    print('>>> rejected cast/profile with depth range ({0},{1})'
                          ''.format(prof_data[params[0]]['depth'][0],prof_data[params[0]]['depth'][-1]))
                continue
        if calc_mld:
            mld = gt.mld(prof_data,ref_reject=reject_mld_ref,bottom_return='NaN',verbose_warn=verbose)
            mld_reject = False
            if mld is None: mld_reject = True
            if isnan(mld):  mld_reject = True
            if reject_mld_below is not None:
                if mld > reject_mld_below:
                    mld_reject = True
                    if verbose:
                        print('>>> ATTENTION!!!!!! cast/profile with deep MLD: {0} m'.format(mld))
                        print('>>> ... {0} depth levels present'.format(len(prof_data[params[0]]['depth'])))
                        print('>>> ... {0}, {1}, {2}'.format(prof_data['datetime'],prof_data['lat'],prof_data['lon']))
                        if 'seal_platform' in prof_data.keys(): print('>>> ... seal {0} '.format(prof_data['seal_platform']))
                        if 'ship_platform' in prof_data.keys(): print('>>> ... ship {0} '.format(prof_data['ship_platform']))
                        if 'wmoid' in prof_data.keys(): print('>>> ... float {0} '.format(prof_data['wmoid']))
            if mld_reject and strict_mld_reject:
                if verbose: print('>>> rejected cast/profile with MLD: {0} m'.format(mld))
                continue
            elif mld_reject and not strict_mld_reject:
                mld = NaN
        if calc_sd is not None:
            if 'sd' not in compiled_data.keys(): compiled_data['sd'] = dict()
            for depth in calc_sd:
                if type(depth) == int or type(depth) == float:
                    val_sd = gt.martinson(prof_data,metric='SD',to_depth=depth,max_depth=depth+50)
                elif len(depth) == 2:
                    val_sd = gt.martinson(prof_data,metric='SD',to_depth=depth[1],sd_from_depth=depth[0],
                                          max_depth=depth[1]+50)
                elif len(depth) == 3:
                    val_sd = gt.martinson(prof_data,metric='SD',to_depth=depth[1],sd_from_depth=depth[0],
                                          sd_ref_psal=depth[2],max_depth=depth[1]+50)
                if val_sd is None: val_sd = NaN
                if depth not in compiled_data['sd'].keys(): compiled_data['sd'][depth] = []
                compiled_data['sd'][depth].append(val_sd)
                if ~isnan(val_sd): counter(counts,'sd',depth,prof_data)
        if calc_tb is not None:
            if 'tb' not in compiled_data.keys(): compiled_data['tb'] = dict()
            for depth in calc_tb:
                if type(depth) == int or type(depth) == float:
                    val_tb = gt.martinson(prof_data,metric='TB',to_depth=depth,max_depth=depth+50)
                elif len(depth) == 2:
                    val_tb = gt.martinson(prof_data,metric='TB',to_depth=depth[1],tb_from_depth=depth[0],
                                          max_depth=depth[1]+50)
                if val_tb is None: val_tb = NaN
                if depth not in compiled_data['tb'].keys(): compiled_data['tb'][depth] = []
                compiled_data['tb'][depth].append(val_tb)
                if ~isnan(val_tb): counter(counts,'tb',depth,prof_data)
        for param in params:
            if param not in compiled_data.keys(): compiled_data[param] = dict()
            if calc_mld and calc_ml_avg and ~isnan(mld):
                ml_avg = gt.vert_prof_eval(prof_data,param,(0.0,mld),z_coor='depth',extrap='nearest')
                if ml_avg is None: ml_avg = NaN
                if 'ml_avg' not in compiled_data[param].keys(): compiled_data[param]['ml_avg'] = []
                compiled_data[param]['ml_avg'].append(ml_avg)
                if ~isnan(ml_avg): counter(counts,param,'ml_avg',prof_data)
            if calc_at_depths is not None:
                for depth in calc_at_depths:
                    if type(depth) == int or type(depth) == float:
                        val_at_depth = gt.vert_prof_eval(prof_data,param,depth,z_coor='depth',extrap='nearest')
                        if val_at_depth is None: val_at_depth = NaN
                        if depth > prof_data[params[0]]['depth'][-1]: val_at_depth = NaN
                        if depth not in compiled_data[param].keys(): compiled_data[param][depth] = []
                        compiled_data[param][depth].append(val_at_depth)
                        if ~isnan(val_at_depth): counter(counts,param,depth,prof_data)
                    elif type(depth) == tuple:
                        vals_at_depths = gt.vert_prof_eval(prof_data,param,list(depth),z_coor='depth',extrap='nearest')
                        if None in vals_at_depths: vals_at_depths = NaN
                        if NaN in vals_at_depths: vals_at_depths = NaN
                        if max(depth) > prof_data[params[0]]['depth'][-1]: vals_at_depths = NaN
                        if depth not in compiled_data[param].keys(): compiled_data[param][depth] = []
                        compiled_data[param][depth].append(vals_at_depths)
                        if ~(isnan(vals_at_depths).any()): counter(counts,param,depth,prof_data)
            if calc_depth_avgs is not None:
                for depth_range in calc_depth_avgs:
                    val_depth_avg = gt.vert_prof_eval(prof_data,param,(depth_range[0],depth_range[1]),
                                                      z_coor='depth',extrap='nearest')
                    if val_depth_avg is None: val_depth_avg = NaN
                    if depth_range[1] > prof_data[params[0]]['depth'][-1]: val_depth_avg = NaN
                    if depth_range not in compiled_data[param].keys(): compiled_data[param][depth_range] = []
                    compiled_data[param][depth_range].append(val_depth_avg)
                    if ~isnan(val_depth_avg): counter(counts,param,depth_range,prof_data)
            if interp_depths is not None:
                interp_depth_vec,interp_prof \
                    = gt.vert_prof_even_spacing(prof_data,param,z_coor='depth',spacing=interp_spacing,
                                                interp_method='linear',extrap='NaN',top=interp_depths[0],
                                                bottom=interp_depths[1],verbose_error=verbose)
                compiled_data['depths'] = interp_depth_vec
                if 'profiles' not in compiled_data[param].keys(): compiled_data[param]['profiles'] = interp_prof
                else:
                    if isinstance(interp_prof,float):
                        # NOTE: this is for case of interp_prof = NaN, and will cause problems if the NaN interp_prof
                        #       occurs at very start or end of this loop
                        interp_prof = tile(NaN,shape(compiled_data[param]['profiles'])[1])
                    compiled_data[param]['profiles'] = vstack((compiled_data[param]['profiles'],interp_prof))
        compiled_data['datetimes'].append(prof_data['datetime'])
        compiled_data['doys'].append(prof_data['datetime'].timetuple().tm_yday)
        compiled_data['lats'].append(prof_data['lat'])
        compiled_data['lons'].append(prof_data['lon'])
        if calc_mld: compiled_data['mlds'].append(mld)
        if 'seal_platform' in prof_data.keys():
            compiled_data['platforms'].append(prof_data['seal_platform'])
            compiled_data['types'].append('seal')
        elif 'ship_platform' in prof_data.keys():
            compiled_data['platforms'].append(prof_data['ship_platform'])
            compiled_data['types'].append('ship')
        elif 'wmoid' in prof_data.keys():
            compiled_data['platforms'].append(prof_data['wmoid'])
            compiled_data['types'].append('float')
        else:
            compiled_data['platforms'].append('Unknown platform')
            compiled_data['types'].append('unknown')
            if verbose: print('>>> ERROR: no platform name or WMOid for this profile.')

    # convert lists to NumPy vectors
    compiled_data['platforms'] = array(compiled_data['platforms'])
    compiled_data['types'] = array(compiled_data['types'])
    compiled_data['datetimes'] = array(compiled_data['datetimes'])
    compiled_data['doys'] = array(compiled_data['doys'])
    compiled_data['lats'] = array(compiled_data['lats'])
    compiled_data['lons'] = array(compiled_data['lons'])
    compiled_data['depths'] = array(compiled_data['depths'])
    if calc_mld: compiled_data['mlds'] = array(compiled_data['mlds'])
    if calc_sd is not None:
        for depth in calc_sd: compiled_data['sd'][depth] = array(compiled_data['sd'][depth])
    if calc_tb is not None:
        for depth in calc_tb: compiled_data['tb'][depth] = array(compiled_data['tb'][depth])
    for param in params:
        if calc_mld and calc_ml_avg and ~isnan(mld):
            compiled_data[param]['ml_avg'] = array(compiled_data[param]['ml_avg'])
        if calc_at_depths is not None:
            for depth in calc_at_depths:
                compiled_data[param][depth] = array(compiled_data[param][depth])
        if calc_depth_avgs is not None:
            for depth_range in calc_depth_avgs:
                compiled_data[param][depth_range] = array(compiled_data[param][depth_range])

    if pickle_filename is not None and pickle_dir is not None:
        pickle.dump(compiled_data,open(pickle_dir + pickle_filename,'wb'))

    count_str = 'Profile counts from ldp.compile_hydrographic_data():\n\n' \
                'Basic search parameters:\n' \
                '- lons: {0}, lats: {1}\n' \
                '- toi: {2}\n\n'.format(lon_bounds,lat_bounds,toi_bounds)

    for param in counts.keys():
        for key_name in counts[param].keys():
            count_str_add = 'Results for parameter <<{0}>> and key, depth, or depth average <<{1}>>:\n' \
                            '- GDAC float profiles: {2} from {3} floats\n' \
                            '- WOD shipboard casts: {4}\n' \
                            '- WOD pinniped profiles: {5}\n\n' \
                            ''.format(param,key_name,counts[param][key_name]['float_profs'],
                                      len(unique(counts[param][key_name]['float_wmoids_used'])),
                                      counts[param][key_name]['wod_ship_profs'],
                                      counts[param][key_name]['wod_seal_profs'])
            count_str = count_str + count_str_add

    if verbose: print(count_str)
    if prof_count_dir is not None and prof_count_filename is not None:
        text_file = open(prof_count_dir + prof_count_filename + '.txt','w')
        text_file.write(count_str)
        text_file.close()

    return compiled_data


def hydro_obs_to_doy_series(obs_dict,param,key,doy_mean=False,doy_std=False,
                            doy_median=False,doy_iqr_25=False,doy_iqr_75=False,
                            specific_year=None,specific_platform=None,
                            days_per_bin=7,drop_nan=False,rm_days=None):
    """ To be used with ldp.compile_hydrographic_obs() to create Pandas series of quantity of interest by date of year.

    Args:
        obs_dict: as returned by ldp.compile_hydrographic_data()
        param: parameter abbreviation as string, or None to index directly into <<obs_dict>> using <<key>>
        key: quantity abbreviation as string (e.g. 'ml_avg')
        doy_mean: True to calculate mean over each DOY, False if looking for specific year/platform's observations
        doy_std: as above; standard deviation
        doy_median: as above; median
        doy_iqr_25: as above; 25-50%; returns positive value
        doy_iqr_75: as above; 50-75%; returns positive value
        specific_year: None, single year as integer, or list of years to limit search
        specific_platform: None, single float WMOid as integer / WOD platform string, or list of platforms
        days_per_bin: take mean/std/etc. of observations within overlapping bins of N days
                      note: this should be an odd number
        drop_nan: True to keep NaNs (useful if some bins have no obs)
               or False to drop NaNs (useful when, e.g., searching for specific float's obs time series)
        rm_days: None or integer number of days to apply centered rolling mean filter to DOY series before returning
    Returns:
        doy_series: Pandas series with DOYs as index
                    note: if pad_specific_year_days is not None, may include DOYs < 1 and/or > 366

    """
    doy_index = []
    doy_data = []

    if param is None:
        obs = obs_dict[key]
    else:
        obs = obs_dict[param][key]

    all_doys = obs_dict['doys'].copy()
    all_years = array([dt.year for dt in obs_dict['datetimes']])

    doy_bins = arange(1,366+1)  # bin centers (with width = <<days_per_bin>>)
    for bin_center in doy_bins:
        bin_range = [bin_center - int((days_per_bin-1)/2), bin_center + int((days_per_bin-1)/2)]
        if bin_range[0] <= 0: # prior year
            bin_range[0] += 366
            left_mask = logical_or(all_doys >= bin_range[0],all_doys <= bin_center)
        else:
            left_mask = logical_and(all_doys >= bin_range[0],all_doys <= bin_center)
        if bin_range[1] >= 367: # next year
            bin_range[1] -= 366
            right_mask = logical_or(all_doys <= bin_range[1],all_doys >= bin_center)
        else:
            right_mask = logical_and(all_doys <= bin_range[1],all_doys >= bin_center)
        obs_mask = logical_or(left_mask,right_mask)
        if specific_year is not None:
            if type(specific_year) == int:
                obs_mask = logical_and(obs_mask,all_years == specific_year)
            else:  # assume multiple years were given
                year_mask = tile(False,len(obs_mask))
                for year in specific_year:
                    year_mask = logical_or(year_mask,all_years == year)
                obs_mask = logical_and(obs_mask,year_mask)
        if specific_platform is not None:
            if type(specific_platform) is not list:
                obs_mask = logical_and(obs_mask,obs_dict['platforms'] == str(specific_platform))
            else:  # assume multiple platforms were given
                platform_mask = tile(False,len(obs_mask))
                for platform in specific_platform:
                    platform_mask = logical_or(platform_mask,obs_dict['platforms'] == str(platform))
                obs_mask = logical_and(obs_mask,platform_mask)
        if sum(obs_mask) == 0:
            bin_obs = NaN
        else:
            if doy_mean:
                bin_obs = nanmean(obs[obs_mask])
            elif doy_std:
                bin_obs = nanstd(obs[obs_mask])
            elif doy_median:
                bin_obs = nanmedian(obs[obs_mask])
            elif doy_iqr_25:
                bin_obs = stats.iqr(obs[obs_mask],rng=(25,50),nan_policy='omit')
            elif doy_iqr_75:
                bin_obs = stats.iqr(obs[obs_mask],rng=(50,75),nan_policy='omit')
        doy_index.append(bin_center)
        doy_data.append(bin_obs)
    doy_index = array(doy_index)
    doy_data = array(doy_data)

    doy_series = pd.Series(index=doy_index,data=doy_data)
    if rm_days is not None: doy_series = doy_series.rolling(window=rm_days,center=True).mean()
    if drop_nan: doy_series = doy_series.dropna()

    return doy_series


def hydro_obs_to_monthly_series(pickle_dir,input_pickle_name,output_pickle_name,params,param_subset_for_climo_anomaly,
                                depths_for_anomalies,climo_dir,climo='waghc17',
                                reject_anomaly=None,N_min=5,verbose=True):
    """ To be used with ldp.compile_hydrographic_obs() to create monthly-, quarterly-, and half-year-binned records of
        hydrographic obs, in which individual depths are calculated as anomalies from WAGHC17, and all metrics
        are also simply grouped and saved.

    Args:
        pickle_dir: directory containing pickled output from ldp.compile_hydrographic_obs()
                    pickled output from this function will be saved here
        input_pickle_name: filename of pickled output from ldp.compile_hydrographic_obs()
        output_pickle_name: filename for output from this function
        params: list of all parameter abbrevations in input file, e.g. ['psal','temp','ptmp']
        param_subset_for_climo_anomaly: subset of <<params>> corresponding to climatology field files in <<woa_fields>>,
                                      e.g. ['psal','temp']
        depths_for_anomalies: list of depths included in input pickle for climatology anomaly calculation
                              if a tuple is included in this list, treat as following:
                                    (LOWER,UPPER,tuple_of_depths) where tuple_of_depths encompasses LOWER-UPPER
                                    this will calculate anomalies at each depth in <<tuple_of_depths>>, then interpolate
                                    the anomalies to 1.0 m spacing, then save the average anomaly between LOWER-UPPER
                                    note: <<tuple_of_depths>> must correspond to a key in <<input_pickle_name>> that
                                          accesses a list of values at each depth in <<tuple_of_depths>>
                              note: if using WAGHC17, all depths for anomalies should *ideally* correspond
                              note: all depths, even within a tuple depth range, must be integers
        climo_dir: FOR WAGHC:
                        directory path to all netCDF files
        climo: 'waghc17' (other options removed)
        reject_anomaly: None or dict with params as keys and rejection criteria for obs anomaly from WOA as value
                        (i.e. reject_anomaly={'psal':1.0} implies rejection of psal anomalies >1.0 or <-1.0)
        N_min: save binned value as NaN if derived from fewer than <<N_min>> sources of data
               (e.g. # unique floats + # shipboard casts + # pinniped casts;
                    note that this treats shipboard casts and pinniped casts as statistically "independent"
                    this calculation of # sources is used for # independent observations in standard error formula)
        verbose: print progress

    Returns:
        hydro_record, a dict with following structure:
            hydro_record[param][depth or depth range][{'obs','monthly','monthly_std_error'}]
                where 'obs' is a Pandas Series with sorted datetime index and individual observations as data
                      'monthly' is a Pandas Series with datetime index of 16th of each month and mean obs as data
                      'monthly_std_error' is a Pandas Series with " index and standard error of obs as data
                      ... for other keys, dive into the code below ...
    """
    input_obs = pickle.load(open(pickle_dir + input_pickle_name,'rb'))

    def hydro_record_slice_and_dice(data_dict,data_param,data_key,new_data,new_data_datetimes,new_data_sources):
        data_dict[data_param][data_key]['obs'] \
            = pd.Series(data=new_data,index=new_data_datetimes)
        data_dict[data_param][data_key]['obs'] \
            = data_dict[data_param][data_key]['obs'].sort_index()
        data_dict[data_param][data_key]['sources'] \
            = pd.Series(data=new_data_sources,index=new_data_datetimes)
        data_dict[data_param][data_key]['sources'] \
            = data_dict[data_param][data_key]['sources'].sort_index()

        nsources \
            = data_dict[data_param][data_key]['sources'].resample('MS',loffset=timedelta(days=16)).nunique()
        data_dict[data_param][data_key]['monthly'] \
            = data_dict[data_param][data_key]['obs'].resample('MS',loffset=timedelta(days=16)).mean()
        data_dict[data_param][data_key]['monthly_std_error'] \
            = data_dict[data_param][data_key]['obs'].resample('MS',loffset=timedelta(days=16)).std() / \
              sqrt(nsources)
        data_dict[data_param][data_key]['monthly'][nsources < N_min] = NaN
        data_dict[data_param][data_key]['monthly_std_error'][nsources < N_min] = NaN
        data_dict[data_param][data_key]['monthly_grouped'] \
            = data_dict[data_param][data_key]['obs'].resample('MS',loffset=timedelta(days=16)).aggregate(lambda x: tuple(x))

        nsources \
            = data_dict[data_param][data_key]['sources'].resample('QS',loffset=timedelta(days=int(30.5 * 1.5))).nunique()
        data_dict[data_param][data_key]['quarterly_mean'] \
            = data_dict[data_param][data_key]['obs'].resample('QS',loffset=timedelta(days=int(30.5 * 1.5))).mean()
        data_dict[data_param][data_key]['quarterly_std_error'] \
            = data_dict[data_param][data_key]['obs'].resample('QS',loffset=timedelta(days=int(30.5 * 1.5))).std() / \
              sqrt(nsources)
        data_dict[data_param][data_key]['quarterly_mean'][nsources < N_min] = NaN
        data_dict[data_param][data_key]['quarterly_std_error'][nsources < N_min] = NaN

        quarterly_median \
            = data_dict[data_param][data_key]['obs'].resample('QS',loffset=timedelta(days=int(30.5 * 1.5))).median()
        data_dict[data_param][data_key]['quarterly_median'] = quarterly_median
        quarterly_median_to_qr25 \
            = data_dict[data_param][data_key]['obs'].resample('QS',loffset=timedelta(days=int(30.5 * 1.5)))\
            .apply(stats.iqr,rng=(25,50))
        quarterly_median_to_qr75 \
            = data_dict[data_param][data_key]['obs'].resample('QS',loffset=timedelta(days=int(30.5 * 1.5)))\
            .apply(stats.iqr,rng=(50,75))
        data_dict[data_param][data_key]['quarterly_quartile_25'] = quarterly_median - quarterly_median_to_qr25
        data_dict[data_param][data_key]['quarterly_quartile_75'] = quarterly_median + quarterly_median_to_qr75
        data_dict[data_param][data_key]['quarterly_grouped'] = data_dict[data_param][data_key]['obs']\
            .resample('QS',loffset=timedelta(days=int(30.5 * 1.5))).aggregate(lambda x: tuple(x))

        # necessary to add dummy date on Jan. 1 of earliest year for semi-annual ('2QS') sampling to align with Jan. 1
        # thus output Pandas Series will have some Nan values at start... just ignore these
        obs_with_dummy = data_dict[data_param][data_key]['obs'].copy()
        obs_with_dummy.sort_index(inplace=True) # can't chain with copy() command for some reason
        obs_with_dummy.loc[datetime(obs_with_dummy.index[0].year - 1,1,1)] = NaN
        halfyear_median = obs_with_dummy.resample('2QS',loffset=timedelta(days=int(30.5 * 3))).median()
        with warnings.catch_warnings():  # to ignore RuntimeWarning regarding computing stats.iqr([NaN])
            warnings.simplefilter('ignore')
            halfyear_median_to_qr25 \
                = obs_with_dummy.resample('2QS',loffset=timedelta(days=int(30.5 * 3)))\
                .apply(stats.iqr,rng=(25,50),nan_policy='omit')
            halfyear_median_to_qr75 \
                = obs_with_dummy.resample('2QS',loffset=timedelta(days=int(30.5 * 3)))\
                .apply(stats.iqr,rng=(50,75),nan_policy='omit')
        data_dict[data_param][data_key]['halfyear_median'] = halfyear_median
        data_dict[data_param][data_key]['halfyear_quartile_25'] = halfyear_median - halfyear_median_to_qr25
        data_dict[data_param][data_key]['halfyear_quartile_75'] = halfyear_median + halfyear_median_to_qr75
        n_obs = obs_with_dummy.resample('2QS',loffset=timedelta(days=int(30.5 * 3))).count()
        data_dict[data_param][data_key]['halfyear_median'][n_obs < N_min] = NaN
        data_dict[data_param][data_key]['halfyear_quartile_25'][n_obs < N_min] = NaN
        data_dict[data_param][data_key]['halfyear_quartile_75'][n_obs < N_min] = NaN
        data_dict[data_param][data_key]['halfyear_n_obs'] \
            = n_obs[data_dict[data_param][data_key]['halfyear_median'].index]
        data_dict[data_param][data_key]['halfyear_grouped'] \
            = obs_with_dummy.resample('2QS',loffset=timedelta(days=int(30.5 * 3))).aggregate(lambda x: tuple(x))

    # for individual depths, calculate anomaly from climatology and save
    hydro_record = dict()
    for param_idx,param in enumerate(param_subset_for_climo_anomaly):
        if param not in hydro_record.keys(): hydro_record[param] = dict()
        for depth in depths_for_anomalies:
            if type(depth) == tuple:
                method = 'range'
                if verbose: print('>>> working on climo interp for param {0} and depths {1[0]}-{1[1]}'.format(param,depth))
                depth_string = '{0}-{1}_anomaly'.format(int(depth[0]),int(depth[1]))
                if depth[0] < min(depth[2]) or depth[1] > max(depth[2]):
                    print('>>> ERROR: tuple of depths corresponding to data does not encompass range of depths for anomaly')
            else:
                method = 'single'
                if verbose: print('>>> working on climo interp for param {0} and depth {1}'.format(param,depth))
                depth_string = '{0}_anomaly'.format(int(depth))

            hydro_record[param][depth_string] = dict()
            fields = waghc_load_field(climo_dir)
            obs_datetimes = []
            obs_anomalies = []
            obs_sources = []
            ship_counter = 1
            seal_counter = 1
            for obs_idx,dt in enumerate(input_obs['datetimes']):
                df.how_far(obs_idx,input_obs['datetimes'],0.05)
                if method == 'single':  depths_to_interp = [int(depth)]
                elif method == 'range': depths_to_interp = [int(d) for d in depth[2]]
                climo_vals = []
                for this_depth in depths_to_interp:
                    climo_val = waghc_interp_to_location(fields,param,this_depth,
                                                         input_obs['lons'][obs_idx],input_obs['lats'][obs_idx],
                                                         datetime_for_interp=dt)
                    climo_vals.append(climo_val)
                if method == 'single':
                    anomaly = input_obs[param][depth][obs_idx] - climo_vals[0]
                elif method == 'range':
                    anomalies = array(input_obs[param][depth[2]][obs_idx]) - array(climo_vals)
                    regularly_spaced_depths = arange(depth[0],depth[1]+1.0,1.0)
                    regularly_spaced_anoms = interp(regularly_spaced_depths,depth[2],anomalies)
                    anomaly = mean(regularly_spaced_anoms)
                if reject_anomaly is not None:
                    if param in reject_anomaly.keys():
                        if abs(anomaly) >= reject_anomaly[param]: continue
                if isnan(anomaly): continue
                obs_datetimes.append(dt)
                obs_anomalies.append(anomaly)
                if input_obs['types'][obs_idx] == 'ship':
                    platform = '{0} (ship observation {1})'.format(input_obs['platforms'][obs_idx],ship_counter)
                    ship_counter += 1
                elif input_obs['types'][obs_idx] == 'seal':
                    platform = '{0} (seal observation {1})'.format(input_obs['platforms'][obs_idx],seal_counter)
                    seal_counter += 1
                else:
                    platform = input_obs['platforms'][obs_idx]
                obs_sources.append(platform)

            hydro_record_slice_and_dice(hydro_record,param,depth_string,obs_anomalies,obs_datetimes,obs_sources)

            if climo == 'woa13':
                del fields
            elif climo == 'waghc17':
                fields.close()

    # save unmodified depth values, depth averages, and other metrics (don't calculate anomalies)
    for param_idx,param in enumerate(params):
        if param not in hydro_record.keys(): hydro_record[param] = dict()
        for metric in input_obs[param].keys():
            if type(metric) == tuple:
                if len(metric) >= 3: continue   # ignore long tuples of depths, e.g. those used above for depth-range anomalies
            hydro_record[param][metric] = dict()
            obs_datetimes = []
            obs_depth_avgs = []
            obs_sources = []
            ship_counter = 1
            seal_counter = 1
            for obs_idx,dt in enumerate(input_obs['datetimes']):
                depth_avg = input_obs[param][metric][obs_idx]
                if isnan(depth_avg): continue
                obs_datetimes.append(dt)
                obs_depth_avgs.append(depth_avg)
                if input_obs['types'][obs_idx] == 'ship':
                    platform = '{0} (ship {1})'.format(input_obs['platforms'][obs_idx],ship_counter)
                    ship_counter += 1
                elif input_obs['types'][obs_idx] == 'seal':
                    platform = '{0} (seal {1})'.format(input_obs['platforms'][obs_idx],seal_counter)
                    seal_counter += 1
                else:
                    platform = input_obs['platforms'][obs_idx]
                obs_sources.append(platform)

            hydro_record_slice_and_dice(hydro_record,param,metric,obs_depth_avgs,obs_datetimes,obs_sources)

    pickle.dump(hydro_record,open(pickle_dir + output_pickle_name,'wb'))
    return hydro_record


############# SEA ICE - OUTWARD-FACING FUNCTIONS ################


def sea_ice_data_prep(nimbus5_dir,dmsp_dir,dmsp_nrt_dir,amsre_dir,amsr2_dir,amsr_gridfile,amsr_areafile,nsidc_ps25_grid_dir):
    """ Returns meta-information on sea ice data to be fed to other accessor functions.

    Meant to be called before performing any sea ice analysis, e.g.:
        [sea_ice_grids,sea_ice_data_avail,sea_ice_all_dates] = ldp.sea_ice_data_prep(<ARGS ABOVE>)
        sea_ice_data_avail['nimbus5'][(1973,2,3)] = [<FILEPATH>,True]
        sea_ice_grids['nimbus5']['areas'] = <2D array by lat/lon>

    Returns:
        - grids: dictionary {'nimbus5','dmsp','amsre','amsr2'}, where each entry is a dictionary of grid
          information {'lats','lons','areas'}, for which each entry is a 2D array
        - data_avail: dictionary {'nimbus5','dmsp','amsre','amsr2'}, where each entry is a dictionary of
          date tuple keys (YYYY,MM,DD) returning [filepath, exists]
        - all_dates: simple list of date tuples from first satellite data to today

    """
    grids = {}
    grids['amsre'] = load_amsr_grid(amsr_gridfile, amsr_areafile)
    grids['amsr2'] = grids['amsre']
    grids['amsre_25km'] = load_amsr_grid(amsr_gridfile, amsr_areafile, regrid_to_25km=True)
    grids['amsr2_25km'] = grids['amsre_25km']
    grids['dmsp'] = load_nsidc_ps_25km_grid(nsidc_ps25_grid_dir)
    grids['nimbus5'] = grids['dmsp']

    all_dates = tt.dates_in_range((1972,12,12),tt.now())
    data_avail = {'nimbus5':{}, 'dmsp':{}, 'amsre':{}, 'amsr2':{}}
    for index, d in enumerate(all_dates):
        data_avail['nimbus5'][d] = sea_ice_filename('nimbus5', d, nimbus5_dir, dmsp_dir, dmsp_nrt_dir, amsre_dir, amsr2_dir)
        data_avail['dmsp'][d] = sea_ice_filename('dmsp', d, nimbus5_dir, dmsp_dir, dmsp_nrt_dir, amsre_dir, amsr2_dir)
        data_avail['amsre'][d] = sea_ice_filename('amsre', d, nimbus5_dir, dmsp_dir, dmsp_nrt_dir, amsre_dir, amsr2_dir)
        data_avail['amsr2'][d] = sea_ice_filename('amsr2', d, nimbus5_dir, dmsp_dir, dmsp_nrt_dir, amsre_dir, amsr2_dir)

    return [grids,data_avail,all_dates]


def sea_ice_concentration(date,lats,lons,sea_ice_grids,sea_ice_data_avail,open_threshold=25,only_return_extent=False,
                          use_goddard_over_amsr=False,use_gsfc_merged_not_cdr=True,
                          expand_search_by_x_days=0,start_search_forward=False,
                          circumant_lons_for_extent=None,circumant_lats_for_extent=None,
                          use_only=['dmsp','amsr2','amsre','nimbus5']):
    """ 'Magic' accessor function. Returns sea ice concentration on a given date at a lat/lon, or average sea ice
        concentration, open water area, and/or extent within a box.

    Args:
        - date: tuple (YYYY,MM,DD) [if outside range of sea_ice_data_avail, then will return NaNs]
        - lats, lons: from -180 to 180 only
            -> single values will return nearest-neighbor sea ice concentration
            -> pair of [lower,upper] will return average sea ice concentration within the specified box,
               excluding NaNs, and weighted by area of the grid cells included
        - only_return_extent: False to return SIC conc, open_area, and day_offset
                              True to calculate sea ice extent (total pixels with SIC >= open_threshold) and return
                                      nothing but this result
        - use_goddard_over_amsr (True or [False]): use GSFC Merged or (!) NSIDC CDR or NRT CDR instead of AMSR-E/AMSR2
               (note the argument naming is a bit misleading... oops)
        - use_gsfc_merged_not_cdr ([True] or False): if GSFC Merged/NSIDC CDR is available, choose GSFC Merged over CDR
        - use_only: list of satellite abbreviations to limit search to

    Returns [conc, day_offset]:
        - conc: 0.0 to 1.0 (fraction) sea ice concentration (NaN if no result found within range of acceptable dates)
        - open_area: estimated open water area (km^2) within given box, computed as sum of grid cell areas where SIC <
                     open_threshold, scaled to total area of interest so NaN is not treated as full ice cover
                     (simply returns NaN if single lat/lon pair given)
        - day_offset: +/- offset of date used for result provided (0 if conc is NaN)
        ... or ...
        - extent: estimated extent (km^2) within given box, computed as sum of grid cell areas where SIC >= open_threshold

    Known issues:
        - if date has too low data coverage and returns NaN for conc and open_area, won't expand search to nearby
          dates even if expand_search_by_x_days > 0 (how to remedy this? rewrite to broaden search when
            it discovers the day being examined has low data coverage [high grid cell % NaN])
        - might not handle area-average ice concentrations correctly when given single longitude within 0.75° of 180°E

    """
    if date not in sea_ice_data_avail['amsr2'].keys() and date not in sea_ice_data_avail['amsre'].keys() \
            and date not in sea_ice_data_avail['dmsp'].keys() and date not in sea_ice_data_avail['nimbus5'].keys():
        if not only_return_extent: return NaN, NaN, 0
        else:                      return NaN
    elif use_goddard_over_amsr and sea_ice_data_avail['dmsp'][date][1] and 'dmsp' in use_only:
        if use_gsfc_merged_not_cdr: conc_field = load_dmsp(sea_ice_data_avail['dmsp'][date][0],date,use_goddard=True)
        else:                       conc_field = load_dmsp(sea_ice_data_avail['dmsp'][date][0],date,use_goddard=False)
        sat_used = 'dmsp'
    elif sea_ice_data_avail['amsr2'][date][1] and 'amsr2' in use_only:
        conc_field = load_amsr(sea_ice_data_avail['amsr2'][date][0])
        sat_used = 'amsr2'
    elif sea_ice_data_avail['amsre'][date][1] and 'amsre' in use_only:
        conc_field = load_amsr(sea_ice_data_avail['amsre'][date][0])
        sat_used = 'amsre'
    elif sea_ice_data_avail['dmsp'][date][1] and 'dmsp' in use_only:
        if use_gsfc_merged_not_cdr: conc_field = load_dmsp(sea_ice_data_avail['dmsp'][date][0],date,use_goddard=True)
        else:                       conc_field = load_dmsp(sea_ice_data_avail['dmsp'][date][0],date,use_goddard=False)
        sat_used = 'dmsp'
    elif sea_ice_data_avail['nimbus5'][date][1] and 'nimbus5' in use_only:
        conc_field = load_nimbus5(sea_ice_data_avail['nimbus5'][date][0])
        sat_used = 'nimbus5'
    elif expand_search_by_x_days == 0:
        if not only_return_extent: return NaN, NaN, 0
        else:                      return NaN
    else:
        assert expand_search_by_x_days >= 1, 'Check given parameter for expand_search_by_x_days.'
        if start_search_forward:
            search_starting_directionality = 1
        else:
            search_starting_directionality = -1
        days_away_from_orig = search_starting_directionality
        while abs(days_away_from_orig) <= expand_search_by_x_days:
            conc,open_area = sea_ice_concentration(tt.date_offset(date,int(days_away_from_orig)),lats,lons,sea_ice_grids,
                                                   sea_ice_data_avail,open_threshold=open_threshold,
                                                   use_goddard_over_amsr=use_goddard_over_amsr,
                                                   use_gsfc_merged_not_cdr=use_gsfc_merged_not_cdr,
                                                   expand_search_by_x_days=0)[0:2]
            if not isnan(conc): return conc, open_area, days_away_from_orig
            days_away_from_orig *= -1
            conc,open_area = sea_ice_concentration(tt.date_offset(date,int(days_away_from_orig)),lats,lons,sea_ice_grids,
                                                   sea_ice_data_avail,open_threshold=open_threshold,
                                                   use_goddard_over_amsr=use_goddard_over_amsr,
                                                   use_gsfc_merged_not_cdr=use_gsfc_merged_not_cdr,
                                                   expand_search_by_x_days=0)[0:2]
            if not isnan(conc): return conc, open_area, days_away_from_orig
            days_away_from_orig = (days_away_from_orig * -1) + sign(days_away_from_orig * -1)
        return NaN, NaN, 0

    assert (size(lats) is 1 and size(lons) is 1) or (size(lats) is 2 and size(lons) is 2), 'Check lat/lon arguments.'
    if size(lats) is 1 and size(lons) is 1:
        # reduce number of points being used to create the NearestNDInterpolator to a 1.5°-square lat/lon box
        poi_lons = logical_and(sea_ice_grids[sat_used]['lons'] <= gt.convert_360_lon_to_180(lons+0.75),
                               sea_ice_grids[sat_used]['lons'] >= gt.convert_360_lon_to_180(lons-0.75))
        poi_lats = logical_and(sea_ice_grids[sat_used]['lats'] >= lats-0.75, sea_ice_grids[sat_used]['lats'] <= lats+0.75)
        poi = logical_and(poi_lats, poi_lons)
        # interpolate to given lon/lat
        ice_interpolant = spin.NearestNDInterpolator(array([sea_ice_grids[sat_used]['lons'][poi].ravel(),
                                                     sea_ice_grids[sat_used]['lats'][poi].ravel()]).transpose(),
                                                     array(conc_field[poi]).ravel())
        conc = ice_interpolant(lons,lats)
        return conc, NaN, 0
    else:
        if lons[0] < lons[1]:  # i.e. longitude range doesn't span 180°E
            poi_lons = logical_and(sea_ice_grids[sat_used]['lons'] <= lons[1],
                                   sea_ice_grids[sat_used]['lons'] >= lons[0])
        else:
            poi_lons = logical_or(sea_ice_grids[sat_used]['lons'] <= lons[1],
                                  sea_ice_grids[sat_used]['lons'] >= lons[0])
        poi_lats = logical_and(sea_ice_grids[sat_used]['lats'] >= lats[0], sea_ice_grids[sat_used]['lats'] <= lats[1])
        poi = logical_and(poi_lats, poi_lons)
        conc_poi = array(conc_field[poi])
        conc_poi_nan = isnan(conc_poi)
        areas_poi = array(sea_ice_grids[sat_used]['areas'][poi])
        if not only_return_extent:
            conc = sum((conc_poi[logical_not(conc_poi_nan)] / 100) * areas_poi[logical_not(conc_poi_nan)]) \
                   / sum(areas_poi[logical_not(conc_poi_nan)])
            poi_open = logical_and(poi, conc_field <= open_threshold)
            open_area = (sum(sea_ice_grids[sat_used]['areas'][poi_open]) / sum(areas_poi[logical_not(conc_poi_nan)])) \
                        * sum(sea_ice_grids[sat_used]['areas'][poi])
            if sum(conc_poi_nan)/sum(poi) > 0.25: return NaN, NaN, 0 # i.e. if over 25% of grid cells are NaN
            else:                                 return conc, open_area, 0
        elif only_return_extent:
            if gt.nan_fraction_domain(sea_ice_grids[sat_used]['lons'],sea_ice_grids[sat_used]['lats'],conc_field,
                                      circumant_lons_for_extent,circumant_lats_for_extent) > 0.045:
                return NaN
            poi_containing_ice = logical_and(poi, conc_field >= open_threshold)
            extent = (sum(sea_ice_grids[sat_used]['areas'][poi_containing_ice]) / sum(areas_poi[logical_not(conc_poi_nan)])) \
                      * sum(sea_ice_grids[sat_used]['areas'][poi])
            return extent


def sea_ice_concentration_along_track(dates,lats,lons,sea_ice_grids,sea_ice_data_avail,expand_search_by_x_days=0,
                                      start_search_forward=False):
    """ Simple accessor to return vector of sea_ice_concentration() at a series of date/lat/lon coordinates.

    """
    assert(len(dates) == len(lats) == len(lons)), 'Check length of input date/lat/lon vectors.'
    ice_cons = []
    for index,date in enumerate(dates):
        conc = sea_ice_concentration(date,lats[index],lons[index],sea_ice_grids,sea_ice_data_avail,
                                     expand_search_by_x_days=expand_search_by_x_days,
                                     start_search_forward=start_search_forward)[0]
        ice_cons.append(conc)
    return ice_cons


def load_amsr(filepath, regrid_to_25km=False):
    """ Opens AMSR-E or AMSR2 data file and returns ice concentration (IC)

    If regrid_to_25km is True, regrids IC onto NSIDC 25-km-square grid by averaging IC of the 16 nearest
    6.25-km-square pixels to each location on the 25-km-square grid.

    Returns NumPy array of shape (1328, 1264), with concentrations from 0 to 100 (or NaN).

    """
    assert os.path.isfile(filepath), 'AMSR data file cannot be found at {0}.'.format(filepath)
    with h5py.File(filepath, 'r') as data:
        ic_orig = data['ASI Ice Concentration'].value
    if regrid_to_25km is not True:
        return ic_orig
    else:
        old_h = shape(ic_orig)[0]
        old_w = shape(ic_orig)[1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ic_regridded = nanmean(nanmean(ic_orig.reshape([old_h,old_w//4,4]),2).T.reshape(old_w//4,old_h//4,4),2).T
        return ic_regridded


def load_dmsp(filepath,date,use_goddard=True,switch_to_nrt=(2017,12,31)):
    """ Opens NSIDC SMMR/DMSP data file and returns ice concentration.

    Returns NumPy array of shape (332,316), with concentrations from 0 to 100 (or NaN, for land/coasts/missing data).

    Arguments:
        - date: date tuple (YYYY,MM,DD)
        - use_goddard: if True,  will return Goddard Merged (GSFC) NT/BT up to date <<switch_to_nrt>>
                                          or NSIDC Near Real-Time (NRT) Climate Data Record (CDR) thereafter
                       if False, will return Goddard Merged (GSFC) NT/BT for dates prior to 1987-07-09,
                                             NSIDC CDR for dates from 1987-07-09 to date <<switch_to_nrt>>,
                                          or NSIDC Near Real-Time (NRT) CDR thereafter

    """
    if tt.is_time_in_range((1978,11,1),switch_to_nrt,date):
        if use_goddard or (use_goddard is False and tt.is_time_in_range((1978,11,1),(1987,7,9),date)):
            data_field = 'goddard_merged_seaice_conc'
        else:
            data_field = 'seaice_conc_cdr'
    else:  # assume NRT
        data_field = 'seaice_conc_cdr'

    assert os.path.isfile(filepath), 'DMSP data file cannot be found at {0}.'.format(filepath)
    with h5py.File(filepath, 'r') as data:
        ice_conc = data[data_field].value.astype(float32)
    # note: missing data encoded as -1, land/coast/lake/pole hole masks as -2/-3/-4/-5 respectively. All flags
    #       changed here to NaN
    ice_conc[logical_or.reduce((ice_conc == -1,ice_conc == -2,ice_conc == -3,ice_conc == -4, ice_conc == -5))] = NaN
    return ice_conc[0]


def load_nimbus5(filepath):
    """ Opens Nimbus-5 data file and returns ice concentration.

    Returns NumPy array of shape (332,316), with concentrations from 0 to 100 (or NaN, for land/coasts/missing data).

    """
    assert os.path.isfile(filepath), 'Nimbus-5 data file cannot be found at {0}.'.format(filepath)
    with h5py.File(filepath, 'r') as data:
        ice_conc = data['Raster Image #0'].value.astype(float32)
    # note: missing data encoded as 157, land/coast/lake masks as 168/178/120 respectively, and
    #       ocean (ice-free) as 125; here, ocean is changed to 0% ice, other flags changed to NaN
    ice_conc[logical_or.reduce((ice_conc == 157,ice_conc == 168,ice_conc == 178,ice_conc == 120))] = NaN
    ice_conc[ice_conc == 125] = 0
    return ice_conc


############# REANALYSIS - OUTWARD-FACING FUNCTIONS ################


def load_ecmwf(data_dir,filename,datetime_range=None,lat_range=None,lon_range=None,
               export_to_dir=None,export_filename=None,export_chunks=True,verbose=False,super_verbose=False):
    """ Opens ERA-Interim or ERA-40 reanalysis data files downloaded in netCDF format with a custom grid.

    Secondary use:
        Run this routine on newly downloaded files to calculate derived quantities and de-accumulate forecasts. Use
        argument <<export_to_dir>> to export new version, then manually delete original.

    Args:
        data_dir: directory of data file
        filename: filename including extension
        datetime_range: None or [Datetime0,Datetime1] or [Datestring0,Datestring1] to subset fields
            note: to slice with open right end, e.g., use [Datetime0,None]
            note: selection is generous, so ['2016-1-1','2016-1-1'] will include all hours on January 1, 2016
            note: example of Datestring: '2016-1-1-h12'
        lat_range: None or [lat_N,lat_S] to subset fields (!!! - descending order - !!!)
        lon_range: None or [lon_W,lon_E] to subset fields
        export_to_dir: None or directory to export new netCDF containing derived quantities, modified variables
            note: in this case, will not return Dataset, and will ignore datetime_range when export_chunks is True
            note: this must be a *different* directory than data_dir!
        export_filename: None or new filename to use when exporting (including extension)
        export_chunks: True or False (True for first call on a large file; False when calling on chunks of that file)
        verbose: True or False
        super_verbose: True or False (print every processing time step)

    Returns:
        all_data: xarray Dataset with coordinates (time,lats,lons); examples of accessing/slicing follow:
            all_data.loc[dict(time='2016-1-1')]                            to extract without slicing
            all_data.sel(lats=slice(-60,-70))                              to slice all variables
            all_data['skt'].values                                         to convert to eager NumPy array
            all_data['skt'][0,:,:]                                         to slice data using indices (t=0)
            all_data['skt'].loc['2016-1-1':'2016-2-1',-60:-70,0:10]        to slice data using values
            all_data['skt']['latitude']                                    to get view of 1-D coordinate
            all_data['skt']['time']                                        NumPy Datetime coordinate
            all_data['skt']['doy']                                         fractional day-of-year coordinate
            pd.to_datetime(all_data['skt']['time'].values)                 useable Datetime version of the above
            all_data['skt'].attrs['units']
            all_data['skt'].attrs['long_name']

    Note: as shown above, 'doy' (fractional day-of-year) is included as a secondary coordinate with dimension 'time'.

    The following derived quantities are calculated here:
        'curlt': wind stress curl using 'iews' and 'inss'
        'div': divergence of 10-m wind using 'u10' and 'v10'
        'div_ice': estimated sea-ice divergence given 30% turning to left and 2% scaling of u10 and v10
        'q2m': 2-m specific humidity from 'msl' and 'd2m'
        'si10': 10-m wind speed from 'u10' and 'v10' (evaluated lazily using Dask only if export_to_dir is None)

    Saved data files:
        'erai_monthly_mean_weddell.nc':    ERA-Interim, Jan 1979 - Dec 2017, monthly mean of daily mean
                                           grid: 0.75° x 0.75°, area: 40.0°S 90.0°W 90.0°S 90.0°E
                                     vars: msl - Mean sea level pressure (Pa) –> (hPa)
                                           sp - Surface pressure (Pa) –> (hPa)
                                           sst - Sea surface temperature (K) –> (°C)
                                           skt - Skin temperature (K) –> (°C)
                                           t2m - Temperature at 2 meters (K) –> (°C)
                                           u10, v10 - U, V wind components at 10 m (m/s)
                                           si10 - wind speed at 10 m (m/s)
        'erai_monthly_mean_weddell_forecast.nc':
                                           ERA-Interim, Jan 1979 - Dec 2017, monthly mean of daily mean
                                           grid: 0.75° x 0.75°, area: 40.0°S 90.0°W 90.0°S 90.0°E
                                     vars: iews, inss - U, V instantaneous turbulent surface stress (N/m^2)
        'erai_daily_weddell.nc':           ERA-Interim, 1979-01-01 - 2017-12-31, daily, step=0 (analysis),
                                             times 0:00, 6:00, 12:00, 18:00
                                           grid: 0.75° x 0.75°, area: 40.0°S 90.0°W 90.0°S 90.0°E
                                     vars: msl - Mean sea level pressure (Pa) –> (hPa)
                                           sst - Sea surface temperature (K) –> (°C)
                                           skt - Skin temperature (K) –> (°C)
                                           t2m - Temperature at 2 meters (K) –> (°C)
                                           u10, v10 - U, V wind components at 10 m (m/s)
                                           d2m - Dewpoint temperature at 2 meters (K)
                                           q2m - Specific humidity at 2 meters, calculated here from msl and 2d (kg/kg)
        'erai_daily_weddell_forecast.nc':  ERA-Interim, 1979-01-01 - 2017-12-31, daily, steps = 6, 12 (forecast)
                                             times 0:00 and 12:00
                                           grid: 0.75° x 0.75°, area: 40.0°S 90.0°W 90.0°S 90.0°E
                                     vars: iews, inss - U, V instantaneous turbulent surface stress (N/m^2)
                                           tp - Total precipitation (m) –> Precipitation rate (m/s)
                                           sf - Snowfall (m water equivalent) –> Snowfall rate (m/s)
                                           e - Evaporation (m) –> Evaporation rate (m/s), positive for evap to atmos
                                           sshf - Surface sensible heat flux (J/m^2) -> (W/m^2)
                                           slhf - Surface latent heat flux (J/m^2) -> (W/m^2)
                                           ssr - Surface net solar radiation (shortwave) (J/m^2) -> (W/m^2)
                                           str - Surface net thermal radiation (longwave) (J/m^2) -> (W/m^2)
                                           strd - Surface thermal radiation (longwave) downwards (J/m^2) –> (W/m^2)
    """

    # export mode may require splitting numerical processing into chunks
    max_chunk = 0.5  # in GB, maximum file size to attempt to process in memory
    if export_to_dir is not None and export_chunks:
        file_size = os.path.getsize(data_dir + filename)/10e8  # file size in GB
        if file_size > max_chunk:
            num_chunks = int(ceil(file_size/max_chunk))
            all_data = xr.open_dataset(data_dir + filename)
            num_times = all_data.dims['time']
            times_per_chunk = int(ceil(num_times/num_chunks))
            all_times = all_data['time'].values
            all_data.close()

            # process and save data in chunks
            slice_start_indices = arange(0,num_times,times_per_chunk)
            just_filename,just_extension = os.path.splitext(filename)
            for chunk_counter, start_idx in enumerate(slice_start_indices):
                end_idx = start_idx + times_per_chunk - 1
                if end_idx >= len(all_times): end_idx = -1  # for final chunk, use last time index
                dt_range = [str(all_times[start_idx]),str(all_times[end_idx])]
                if verbose: print('>> Processing chunk {0} of {1} from {2} to {3}'
                                  ''.format(chunk_counter+1,len(slice_start_indices),*dt_range))
                load_ecmwf(data_dir,filename,datetime_range=dt_range,lat_range=lat_range,lon_range=lon_range,
                           export_filename='{0}_chunk_{1:03d}{2}'.format(just_filename,chunk_counter+1,just_extension),
                           export_to_dir=export_to_dir,export_chunks=False,verbose=verbose)

            # open all chunks and concatenate as Dataset
            if verbose: print('>> Opening all chunks of {0}'.format(filename))
            all_data = xr.open_mfdataset(export_to_dir + '{0}_chunk_*{1}'.format(just_filename,just_extension),
                                         concat_dim='time',chunks={'time':100})
            bypass_normal_open = True
        else:
            bypass_normal_open = False
    else:
        bypass_normal_open = False

    if not bypass_normal_open:
        if verbose: print('>> Opening {0}'.format(filename))
        all_data = xr.open_dataset(data_dir+filename,chunks={'time':100})   # O(100 MB) per chunk

    if 'longitude' in all_data and 'latitude' in all_data:
        all_data = all_data.rename({'latitude':'lats','longitude':'lons'})

    if datetime_range is not None:
        all_data = all_data.sel(time=slice(datetime_range[0],datetime_range[1]))
    if lat_range is not None:
        all_data = all_data.sel(lats=slice(lat_range[0],lat_range[1]))
    if lon_range is not None:
        all_data = all_data.sel(lons=slice(lon_range[0],lon_range[1]))

    for var in all_data.data_vars:
        if verbose: print('>>>> Examining variable {0}'.format(var))

        if all_data[var].attrs['units'] == 'Pa':
            orig_name = all_data[var].attrs['long_name']
            all_data[var] /= 100.0
            all_data[var].attrs = {'units':'hPa','long_name':orig_name}
        elif all_data[var].attrs['units'] == 'K' and var != 'd2m':
            orig_name = all_data[var].attrs['long_name']
            all_data[var] -= 273.15
            all_data[var].attrs = {'units':'°C','long_name':orig_name}

        # de-accumulate forecast fields (hours 0 and 12), if not already
        if var in ['tp','e','sf','sshf','slhf','ssr','str','strd'] and 'deaccumulated' not in all_data[var].attrs:
            orig_name = all_data[var].attrs['long_name']
            orig_units = all_data[var].attrs['units']
            time_index = pd.to_datetime(all_data[var].time.values)

            if time_index[0].hour == 0 or time_index[0].hour == 12:
                all_data[var][dict(time=0)] /= 2
                first_step = 1
            else:
                first_step = 0
            if time_index[-1].hour in [3,6,9,15,18,21]:  # handles 3-hourly and 6-hourly steps
                last_step = len(time_index) - 1
            else:
                last_step = len(time_index)

            all_data[var].load() # load Dask array into memory (which means reasonably small chunks are necessary!)
            all_data[var][first_step+1:last_step:2] -= all_data[var][first_step:last_step:2].values

            if   (all_data['time'].values[1] - all_data['time'].values[0]) == timedelta64(6,'h'):
                step_hours = 6
            elif (all_data['time'].values[1] - all_data['time'].values[0]) == timedelta64(3,'h'):
                step_hours = 3
            else:
                raise ValueError('Error from ldp.load_ecmwf(): Forecast time interval is not 3 or 6 hours.')

            seconds_in_step = step_hours * 60 * 60
            all_data[var] /= seconds_in_step

            if var == 'e': all_data[var] *= -1

            all_data[var].attrs['long_name'] = orig_name
            all_data[var].attrs['units'] = orig_units

            if   all_data[var].attrs['units'] == 'm':       all_data[var].attrs['units'] = 'm/s'
            elif all_data[var].attrs['units'] == 'J m**-2': all_data[var].attrs['units'] = 'W/m^2'

            all_data[var].attrs['deaccumulated'] = 'True'

    # calculate 2-m specific humidity from surface pressure and dewpoint temperature, if available
    # uses Equations 7.4 and 7.5 on p. 92 of ECMWF IFS Documentation, Ch. 7:
    #   https://www.ecmwf.int/sites/default/files/elibrary/2015/9211-part-iv-physical-processes.pdf
    if 'q2m' not in all_data and 'd2m' in all_data and 'msl' in all_data:
        if verbose: print('>>>> Calculating 2-m specific humidity')

        # constants for Teten's formula for saturation water vapor pressure over water [not ice] (Eq. 7.5)
        # origin: Buck (1981)
        a1 = 611.21 # Pa
        a3 = 17.502 # unitless
        a4 = 32.19  # K
        T_0 = 273.16 # K

        # saturation water vapor pressure; units: Pa
        e_sat_at_Td = a1 * exp(a3 * (all_data['d2m'] - T_0) / (all_data['d2m'] - a4))

        # saturation specific humidity at dewpoint temperature (Eq. 7.4)
        # note conversion of surface pressure from hPa back to Pa
        R_dry_over_R_vap = 0.621981  # gas constant for dry air over gas constant for water vapor, p. 110
        q_sat_at_Td = R_dry_over_R_vap * e_sat_at_Td / (100*all_data['msl'] - (e_sat_at_Td*(1.0 - R_dry_over_R_vap)))

        all_data['q2m'] = q_sat_at_Td
        all_data['q2m'].attrs['units'] = 'kg/kg'
        all_data['q2m'].attrs['long_name'] = 'Specific humidity at 2 m'

    # calculate 10-m wind speed from u, v
    # note: this evaluates lazily using Dask, so expect processing hangs upon computation (instead of load)
    # note: included only if not exporting to a new netCDF file (don't want to take up unnecessary space)
    if 'si10' not in all_data and 'u10' in all_data and 'v10' in all_data and export_to_dir is None:
        if verbose: print('>>>> Calculating 10-m wind speed')
        all_data['si10'] = (all_data['u10']**2 + all_data['v10']**2)**0.5
        all_data['si10'].attrs['units'] = 'm/s'
        all_data['si10'].attrs['long_name'] = '10 metre wind speed'

    # calculate estimated sea-ice drift velocity using 10-m wind u, v
    # uses formulation for thin Weddell Sea pack ice:
    #   3% drift velocity scaling and turning angle of 23° to left of winds
    #   (Martinson and Wamser 1990)
    # other option:
    #   2% drift velocity scaling and turning angle of 30° to left of winds
    #  (Wang et al. 2014, Scientific Reports, "Cyclone-induced rapid creation of extreme Antarctic sea ice conditions")
    # note: these vectors are used to compute estimated sea-ice divergence, and are deleted afterwards
    if 'ui10' not in all_data and 'vi10' not in all_data and 'u10' in all_data and 'v10' in all_data:
        if verbose: print('>>>> Calculating estimated sea-ice drift velocity')
        scaling = 0.03
        turning_angle = 23.0   # positive for counter-clockwise (to left of wind)
        turning_angle_radians = turning_angle / 180.0 * pi
        transform = cos(turning_angle_radians) + sin(turning_angle_radians) * 1j
        rotated_u_v = (all_data['u10'] + all_data['v10'] * 1j) * transform
        all_data['ui10'] = rotated_u_v.real * scaling
        all_data['ui10'].attrs['units'] = 'm/s'
        all_data['ui10'].attrs['long_name'] = 'Estimated sea-ice drift velocity, eastward component'
        all_data['vi10'] = rotated_u_v.imag * scaling
        all_data['vi10'].attrs['units'] = 'm/s'
        all_data['vi10'].attrs['long_name'] = 'Estimated sea-ice drift velocity, northward component'

    # second-order-accurate central differencing
    def field_derivs(for_dx,for_dy):
        data_shape = for_dy.shape
        ddx = zeros(data_shape)
        ddy = zeros(data_shape)
        lat_spacing = gt.distance_between_two_coors(for_dy['lats'][0],for_dy['lons'][0],
                                                    for_dy['lats'][1],for_dy['lons'][0]) * -1
        lon_spacing = array([gt.distance_between_two_coors(for_dy['lats'][lat_idx],for_dy['lons'][0],
                                                           for_dy['lats'][lat_idx],for_dy['lons'][1])
                             for lat_idx in range(len(for_dy['lats']))])
        nonzero_mask = lon_spacing > 0   # to deal with poles (90 and -90), where dx is zero
        for dt_idx in range(len(for_dy['time'])):
            if super_verbose: print('>>>>>> time {0} of {1}'.format(dt_idx+1,len(for_dy['time'])))
            ddy[dt_idx,:,:] = gradient(for_dy[dt_idx,:,:],lat_spacing,axis=0)
            ddx[dt_idx,nonzero_mask,:] \
                = gradient(for_dx[dt_idx,nonzero_mask,:],1.0,axis=1) / lon_spacing[nonzero_mask,None]
            ddx[dt_idx,~nonzero_mask,:] = NaN   # to deal with poles (90 and -90), where dx is zero
        return ddx, ddy

    # calculate wind stress curl (d(tau_y)/dx - d(tau_x)/dy)
    if 'curlt' not in all_data and 'iews' in all_data and 'inss' in all_data:
        if verbose: print('>>>> Calculating wind stress curl')

        ddx, ddy = field_derivs(for_dx=all_data['inss'],for_dy=all_data['iews'])
        all_data['curlt'] = all_data['iews'].copy()
        all_data['curlt'].values = (ddx - ddy) * 10**7
        all_data['curlt'].attrs['units'] = r'10$^{-7}$ N m$^{-3}$'
        all_data['curlt'].attrs['long_name'] = 'Wind stress curl'

    # calculate 10-m wind divergence (d(u10)/dx + d(v10)/dy)
    if 'div' not in all_data and 'u10' in all_data and 'v10' in all_data:
        if verbose: print('>>>> Calculating 10-m wind divergence')

        ddx,ddy = field_derivs(for_dx=all_data['u10'],for_dy=all_data['v10'])
        all_data['div'] = all_data['u10'].copy()
        all_data['div'].values = (ddx + ddy) * 10**5
        all_data['div'].attrs['units'] = r'10$^{-5}$ s$^{-1}$'
        all_data['div'].attrs['long_name'] = '10-m wind divergence'

    # calculate estimated sea-ice divergence (d(ui10)/dx + d(vi10)/dy)
    if 'div_ice' not in all_data and 'ui10' in all_data and 'vi10' in all_data:
        if verbose: print('>>>> Calculating estimated sea-ice divergence')

        ddx,ddy = field_derivs(for_dx=all_data['ui10'],for_dy=all_data['vi10'])
        all_data['div_ice'] = all_data['ui10'].copy()
        all_data['div_ice'].values = (ddx + ddy) * 10**5
        all_data['div_ice'].attrs['units'] = r'10$^{-5}$ s$^{-1}$'
        all_data['div_ice'].attrs['long_name'] = 'Estimated sea-ice divergence'

    # add day-of-year as a secondary coordinate with dimension 'time'
    if 'doy' not in all_data.coords:
        datetime_index = pd.to_datetime(all_data['time'].values)
        doy_index = datetime_index.dayofyear + datetime_index.hour / 24. + datetime_index.minute / 60.
        all_data.coords['doy'] = ('time',doy_index)

    if export_to_dir is not None:
        # set encoding only if exporting to a new netCDF file here (!)
        # remember to do this if exporting to a new netCDF file elsewhere...
        #
        # changing encoding (scale factor and offset) is necessary because the original netCDF file's encoding
        #   results in truncation/loss of precision when applied to the processed variables here (some of which
        #   where divided by large numbers, for instance)
        # these formulae for optimal scale factors and offsets are from:
        #   http://james.hiebert.name/blog/work/2015/04/18/NetCDF-Scale-Factors/
        for var in all_data.data_vars:
            n_bits = 16  # because int16
            var_max = asscalar(all_data[var].max().values)  # .values necessary because of lazy Dask evaluation
            var_min = asscalar(all_data[var].min().values)
            all_data[var].encoding['dtype'] = 'int16'
            all_data[var].encoding['scale_factor'] = (var_max - var_min) / ((2**n_bits) - 1)
            all_data[var].encoding['add_offset'] = var_min + (2**(n_bits - 1) * all_data[var].encoding['scale_factor'])
            all_data[var].encoding['_FillValue'] = -9999

        if export_filename is None: new_filename = filename
        else:                       new_filename = export_filename
        all_data.to_netcdf(export_to_dir + new_filename)
        all_data.close()
    else:
        return all_data


def load_ecmwf_mask(data_dir,filename,var_name='lsm'):
    """ Returns xarray DataArray of mask (e.g. land-sea mask) for ECMWF reanalysis grid (e.g. ERA-Interim).

    Downloaded manually:
        0.75x0.75° ERA-Interim land-sea mask, netCDF: http://apps.ecmwf.int/datasets/data/interim-full-invariant/
            variable name is 'lsm' (0 = sea, 1 = land)

    """
    mask = xr.open_dataset(data_dir + filename)
    mask = mask[var_name].isel(time=0)

    mask = mask.rename({'latitude':'lats','longitude':'lons'})
    lons = mask['lons'].values
    lons[lons > 180.0] -= 360
    mask['lons'] = lons

    return mask


def climate_index_DataFrame_to_Series(index_DataFrame,dropna=False):
    """ Convert N_years x N_months Pandas DataFrame to Pandas Series of length N_samples.

    """
    index_years_months = list(zip(index_DataFrame.stack().index.get_level_values(0),
                                  index_DataFrame.stack().index.get_level_values(1)))
    index_datetimes = [datetime(d[0],d[1],1) for d in index_years_months]
    if dropna:
        return pd.Series(data=index_DataFrame.stack().values,index=index_datetimes).dropna()
    else:
        return pd.Series(data=index_DataFrame.stack().values,index=index_datetimes)


def load_sam_index(data_dir):
    """ Loads monthly Southern Annular Mode index created by Gareth Marshall according to Marshall (2003).

    Download manually from website in ASCII format: https://legacy.bas.ac.uk/met/gjma/sam.html
    Relevant citation: Marshall, G. J., 2003: Trends in the Southern Annular Mode from observations and reanalyses.
                       J. Clim., 16, 4134-4143.
    Also see: https://climatedataguide.ucar.edu/climate-data/marshall-southern-annular-mode-sam-index-station-based

    Returns:
        'sam_index': Pandas DataFrame with shape (N_years, N_months), where N_months = 12
                     (missing data indicated by NaN)

    Example of slicing data:
        sam_index.loc[2016]
        sam_index.loc[:,1]                      # month columns labeled 1 to 12
        sam_index.loc[1973:1976,7:10]
        sam_index.loc[1973:1976,7:10].values

    """
    sam_index = pd.read_csv(data_dir + 'SAM_Marshall2003.txt',delim_whitespace=True)
    sam_index.columns = range(1,13)
    return sam_index


def create_reanalysis_index(dataset,param_name='msl',avg_box=None,nearest_to_lat_lon=None,annual_mean_years=[1979,2016],
                            rm_window=None,rm_min=None,rm_center=True,
                            mask_land=None,mask_sea=None,avg_box_north_of_antarctic_coast=False,circumant_lats=None,
                            min_not_mean=False,max_not_mean=False,abs_val=False,
                            calc_box_here=False,search_box=None,just_return_weddell_low_pos=False,
                            create_climo=False,create_climo_iqr=False,climo_years=None,make_year=None,
                            climo_abs_val=['u10','v10','iews','inss']):
    """ Create a record/index of any reanalysis parameter (<<param_name>>) averaged within a given box (<<avg_box>>).

    Args:
        dataset: xarray Dataset produced by load_ecmwf()
        param_name: abbreviation (key) for parameter of interest
        avg_box: list of length 4, [lon_E,lon_W,lat_S,lat_N], representing extent of averaging box
        nearest_to_lat_lon: None or (lat,lon) tuple to only analyze nearest grid cell, instead of using avg_box
        annual_mean_years: None or [start,end] (inclusive) indicating years with complete data to use in annual means
        rm_window: number of periods (observations, e.g. months) for rolling mean filter used to make <<index_filtered>>
                   if None, return None for index_filtered
        rm_min: minimum number of periods for rolling mean to be calculated
                note: defaults to int(rolling_mean_periods/2)
        rm_center: True to filter using centered rolling mean; False for right-edge (accumulated) rolling mean
        mask_land: None or xarray DataArray with corresponding land-sea mask (0 = sea, 1 = land) to ignore land
        mask_sea: None or [see above] to ignore sea
        avg_box_north_of_antarctic_coast: if True, mask south of the Antarctic coast
            note: this is an earlier, more crude version of mask_land above
        circumant_lats: vector of Antarctic coast latitudes, corresponding to longitudes of -180 to 179 (inclusive)
                        with a spacing of 1.0 (must be supplied if <<avg_box_north_of_antarctic_coast>> is True)
        min_not_mean: return minimum value found within <<avg_box>>, not the mean
        max_not_mean: return maximum value found within <<avg_box>>, not the mean
        abs_val: use absolute value of data when finding max, min, or mean in avg_box
        calc_box_here: True to calculate mean position of field minimum and stddev within <<search_box>, and use that to
                       construct an averaging box (e.g. for finding climatological location of Weddell Low)
                       False to use an already-found box, <<avg_box>>
        search_box: list of length 4, [lon_E,lon_W,lat_S,lat_N], representing extent of search box for calc_box_here
        just_return_weddell_low_pos: circumvent entire routine and return [wl_lats,wl_lons] Pandas Series
                                     of calc_box_here minimum position (e.g. Weddell Low position)
        create_climo: if True, return only climatology (mean, std) instead of return args below
                    note: climo series format is Pandas Series with fractional DOY index (e.g. 1.0 to 365.75)
        create_climo_iqr: if True, return climatology (median, iqr_25, iqr_75) instead of return args below
                    note: climo series format is Pandas Series with fractional DOY index (e.g. 1.0 to 365.75)
        climo_years: None (to use all available years)
                     or years as [start,end] (inclusive) to specify which to include in climatology
        make_year: None or specific year (integer YYYY) to export Series (either normal indices, or climo) with
                           Datetime index starting from Jan. 1 of that year
                           note: requires one-year-long series; otherwise non-unique datetimes will be generated
        climo_abs_val: list of parameters for which to calculate climatology on absolute value of series

    Returns:
        index: Pandas Series of calculated values with Pandas Timestamp index
               note: Pandas Timestamps are essentially interchangeable with Datetime, but if conversion needed, see:
                     https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Timestamp.html
        index_filtered: None or rolling mean of index (see above)
        index_annual: annual mean version, with indices (still Datetimes) set to July 1 or July 2 (i.e. halfway
                      through year); to obtain year only, use index_annual.index.year
               note: will include years with incomplete monthly data unless annual_mean_years is specified
        avg_box: returns <<avg_box>> if given, or averaging box found here if calc_box_here is True
        center: returns None, or [longitude, latitude] of found climatological minimum if calc_box_here is True

    """
    if calc_box_here:
        assert search_box is not None,'Error from ldp.create_reanalysis_index(): missing a search box.'

        search_data = dataset[param_name].sel(lons=slice(search_box[0],search_box[1]),
                                              lats=slice(search_box[3],search_box[2]))
        search_data_flat = search_data.stack(lonlat=('lons','lats'))
        found_lonlats = search_data_flat['lonlat'][search_data_flat.argmin(dim='lonlat').values]
        found_lons = pd.Series(data=found_lonlats['lons'],
                               index=array([pd.Timestamp(dt64) for dt64 in search_data['time'].values]))
        found_lats = pd.Series(data=found_lonlats['lats'],
                               index=array([pd.Timestamp(dt64) for dt64 in search_data['time'].values]))
        found_center = [found_lons.mean(),found_lats.mean()]
        found_std = [found_lons.std(),found_lats.std()]
        avg_box = [found_center[0]-found_std[0],found_center[0]+found_std[0],
                   found_center[1]-found_std[1],found_center[1]+found_std[1]]
        if just_return_weddell_low_pos:
            return found_lons, found_lats
    else:
        found_center = None

    if nearest_to_lat_lon is not None:
        index = dataset[param_name].sel(lats=nearest_to_lat_lon[0],lons=nearest_to_lat_lon[1],
                                        method='nearest')
    else:
        if mask_land is not None or mask_sea is not None:
            if mask_land is not None: geo_mask = mask_land; ignore_val = 1.0
            else:                     geo_mask = mask_sea;  ignore_val = 0.0
            data = dataset[param_name].where(geo_mask != ignore_val)
        elif avg_box_north_of_antarctic_coast:
            lon_grid, lat_grid = meshgrid(dataset[param_name]['lons'],dataset[param_name]['lats'])
            geo_mask = lat_grid > circumant_lats[(floor(lon_grid) + 180).astype(int)]
            data = dataset[param_name].where(geo_mask)
        else:
            data = dataset[param_name]

        if abs_val: data = xr.ufuncs.fabs(data)  # maintains lazy evaluation of Dask array

        data = data.sel(lons=slice(avg_box[0],avg_box[1]),lats=slice(avg_box[3],avg_box[2]))

        # note: Dask array computation/conversion triggered here by .compute()
        #       could also keep as Dask array using .persist(), but not sure if this has any advantages
        if min_not_mean:   index = data.min(dim=['lats','lons'],keep_attrs=True,skipna=True).compute()
        elif max_not_mean: index = data.max(dim=['lats','lons'],keep_attrs=True,skipna=True).compute()
        else:              index = data.mean(dim=['lats','lons'],keep_attrs=True,skipna=True).compute()

    if create_climo or create_climo_iqr:
        if param_name in climo_abs_val: index = abs(index)
        if climo_years is None: index_trimmed = index
        else:                   index_trimmed = index.loc[str(climo_years[0]):str(climo_years[1])]
        if create_climo:
            climo = index_trimmed.groupby('doy').mean(dim='time')
            climo_std = index_trimmed.groupby('doy').std(dim='time')
            climo_series = climo.to_pandas()
            climo_std_series = climo_std.to_pandas()
        elif create_climo_iqr:
            climo = index_trimmed.groupby('doy').median(dim='time')
            climo_iqr_25 = index_trimmed.groupby('doy').reduce(stats.iqr,rng=(25,50))
            climo_iqr_75 = index_trimmed.groupby('doy').reduce(stats.iqr,rng=(50,75))
            climo_series = climo.to_pandas()
            climo_iqr_25_series = climo_iqr_25.to_pandas()
            climo_iqr_75_series = climo_iqr_75.to_pandas()
        if make_year is not None:
            ref_datetime = datetime(make_year-1,12,31)
            new_index = array([timedelta(days=int(doy)) + timedelta(hours=int((doy-floor(doy))*24))
                               for doy in climo_series.index]) + ref_datetime
            climo_series.index = new_index
            if create_climo:
                climo_std_series.index = new_index
            elif create_climo_iqr:
                climo_iqr_25_series.index = new_index
                climo_iqr_75_series.index = new_index
        if create_climo:
            return climo_series, climo_std_series
        elif create_climo_iqr:
            return climo_series, climo_iqr_25_series, climo_iqr_75_series

    # convert from xarray to Pandas (because rolling() operations buggy in xarray)
    index = index.to_pandas()
    if make_year is not None:
        new_index = array([dt.replace(year=make_year) for dt in index.index])
        index.index = new_index

    if rm_window is not None:
        if rm_min is None: rm_min = int(rm_window/2)
        index_filtered = index.rolling(window=rm_window,min_periods=rm_min,center=rm_center).mean()
    else:
        index_filtered = None

    if annual_mean_years is not None:
        index_for_annual_mean = index.loc[str(annual_mean_years[0]):str(annual_mean_years[1])]
    else:
        index_for_annual_mean = index
    index_annual_mean = index_for_annual_mean.resample('AS',loffset=timedelta(6 * 365.24 / 12)).mean()

    return index, index_filtered, index_annual_mean, avg_box, found_center


############# MISCELLANEOUS - OUTWARD-FACING FUNCTIONS ################


def load_bathy(data_dir):
    """ Load bathymetry. Script "inspired" by an example somewhere on the internet.
    
    Note on updating ETOPO1 data:
        - download in ArcGIS ASCII format from here: https://maps.ngdc.noaa.gov/viewers/wcs-client/
        - convert to .npy format using these commands:
             etopo = np.loadtxt(bathymetry_file, skiprows=5)
             np.save('/path/to/data/folder/etopo1_weddell.npy',etopo)
        - move into /Data/Bathymetry/ folder
    """
    bathymetry_file = data_dir + 'Bathymetry/etopo1_weddell.asc'
    bathymetry_file_np = data_dir + 'Bathymetry/etopo1_weddell.npy'
    topo_file = open(bathymetry_file, 'r')
    ncols = int(topo_file.readline().split()[1])
    nrows = int(topo_file.readline().split()[1])
    xllcorner = float(topo_file.readline().split()[1])
    yllcorner = float(topo_file.readline().split()[1])
    cellsize = float(topo_file.readline().split()[1])
    topo_file.close()
    etopo = load(bathymetry_file_np)
    dres = 5
    etopo[:nrows + 1, :] = etopo[nrows + 1::-1, :]
    etopo = etopo[::dres, ::dres]
    lons = arange(xllcorner, xllcorner + cellsize * ncols, cellsize)[::dres]
    lats = arange(yllcorner, yllcorner + cellsize * nrows, cellsize)[::dres]
    return lons,lats,etopo


def load_isd_station(data_dir,station_number,start_year=1950,end_year=datetime.now().year):
    """ Load sub-daily meteorological station data from NOAA NCEI Integrated Surface Database (ISD) ISD-Lite files.

    Reference: https://www.ncdc.noaa.gov/isd

    Args:
        data_dir: directory path
        station_number: six-digit integer station number, likely five-digit WMOID with trailing zero appended
        start_year: first year of data
        end_year: last year of data

    Returns:
        isd_station: dict with <<params>> as keys to Pandas Series with Datetime index, where <<params>> are:
            temp: air temperature (°C)
            dpt: dewpoint temperature (°C)
            mslp: mean sea level pressure (hPa)
            wdir: direction from which the wind is blowing (angular degrees from 0 to 359; likely in intervals of 10;
                                                            calm winds show 0)
            ws: wind speed (m/s)
            sky: sky condition - cloud coverage in oktas (0-8) or other specifier (9-19);
                 see ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-lite/isd-lite-format.pdf
            precip_1: liquid precipitation measured over a 1-hour period (m)
            precip_6: liquid precipitation measured over a 6-hour period (m)
                note: trace precipitation was encoded as -1 and is treated here as zero
        NOTE: missing values are NaN; regularity of time index is not guaranteed (it's whatever is in the data file)

    """
    isd_station = {'temp':pd.Series(),'dpt':pd.Series(),'mslp':pd.Series(),'wdir':pd.Series(),'ws':pd.Series(),
                   'sky':pd.Series(),'precip_1':pd.Series(),'precip_6':pd.Series()}
    for year in range(start_year,end_year + 1):
        filename = '{0}-99999-{1}'.format(station_number,year)
        try:
            dataframe = pd.read_csv(data_dir + filename,delim_whitespace=True,header=None,na_values=-9999,
                                    parse_dates=[[0,1,2,3]],index_col=0)
        except FileNotFoundError:
            continue
        isd_station['temp'] = isd_station['temp'].append(dataframe[4] / 10.0)
        isd_station['dpt'] = isd_station['dpt'].append(dataframe[5] / 10.0)
        isd_station['mslp'] = isd_station['mslp'].append(dataframe[6] / 10.0)
        isd_station['wdir'] = isd_station['wdir'].append(dataframe[7])
        isd_station['ws'] = isd_station['ws'].append(dataframe[8] / 10.0)
        isd_station['sky'] = isd_station['sky'].append(dataframe[9])
        precip_1_without_trace = dataframe[10].copy()
        precip_1_without_trace[precip_1_without_trace == -1.0] = 0.0
        precip_6_without_trace = dataframe[11].copy()
        precip_6_without_trace[precip_6_without_trace == -1.0] = 0.0
        isd_station['precip_1'] = isd_station['precip_1'].append(precip_1_without_trace / (10.0 * 1000))
        isd_station['precip_6'] = isd_station['precip_6'].append(precip_6_without_trace / (10.0 * 1000))

    return isd_station


def load_reader_station(data_dir,filename):
    """ Load monthly mean quality-controlled surface meteorological data from BAS READER met dataset.

    Data provenance:
        https://legacy.bas.ac.uk/met/READER/data.html

    Cite as: '... Reference Antarctic Data for Environmental Research (READER, www.antarctica.ac.uk/met/READER)
                  [Turner et al., 2004] archive ...'
        Turner, J., et al. (2004), The SCAR READER project: Toward a high-quality database of mean Antarctic
        meteorological observations, J. Clim., 17, 2890–2898.

    Antarctic station locations:
        WMOid 89512: Novolazarevskaya (Novolazarevskaja): 70.8°S, 11.8°E, elevation 119 m
        WMOid 89002: Neumayer: 70.7°S, 8.4°W, elevation 50 m
        WMOid 89532: Showa (Syowa): 69.0°S, 39.6°E, elevation 21 m
        WMOid 89022: Halley: 75.5°S, 26.4°W, elevation 30 m
        WMOid 88903: Grytviken: 54.3°S, 36.5°W, elevation 3 m

    """
    dataframe = pd.read_csv(data_dir+filename,delim_whitespace=True,index_col=0,header=None,skiprows=1,na_values='-')
    return climate_index_DataFrame_to_Series(dataframe).resample('MS').mean()  # to retain NaN values


############# AUXILIARY (INTERNAL) FUNCTIONS ################


def sea_ice_filename(sat_name,date,nimbus5_dir,dmsp_dir,dmsp_nrt_dir,amsre_dir,amsr2_dir):
    """ Returns full path (directory + filename) of a sea ice data file and checks for existence.

    Arguments:
        sat_name: 'amsr' (meaning AMSR2 or AMSR-E), 'amsr2', 'amsre', 'dmsp' (meaning NSIDC NRT CDR, NSIDC CDR, or
                  GSFC Merged), 'dmsp_nrt', 'dmsp_cdr_or_gsfc', or 'nimbus5'
        date: tuple (YYYY,MM,DD)
        data directories: full paths to data folders

    Returns [filepath, exists], where:
        filepath: string of directory + filename
        exists: True or False (does the file exist?)
    """
    sat_abbrevs = ['n05','n07','f08','f11','f13','f17','f18_nrt',
                   'ame','am2']
    sat_start_dates = [(1972,12,12),(1978,11,1),(1987,7,9),(1991,12,3),(1995,10,1),(2008,1,1),(2018,1,1),
                       (2002,6,1),(2012,7,4)]
    sat_end_dates = [(1977,5,11),(1987,7,8),(1991,12,2),(1995,9,30),(2007,12,31),(2017,12,31),tt.now(),
                     (2011,10,4),tt.now()]   # note: okay if end date, e.g. tt.now(), is beyond actual end date

    # if specified satellite name ambiguous, figure out exactly which record to use
    if sat_name is 'amsr':
        if tt.is_time_in_range(sat_start_dates[sat_abbrevs.index('ame')], sat_end_dates[sat_abbrevs.index('ame')], date):
            sat_name = 'amsre'
        elif tt.is_time_in_range(sat_start_dates[sat_abbrevs.index('am2')], sat_end_dates[sat_abbrevs.index('am2')], date):
            sat_name = 'amsr2'
        else:
            raise ValueError('Satellite name given as AMSR but given date not within AMSR-E or AMSR2 date ranges.')
    elif sat_name is 'dmsp':
        if tt.is_time_in_range(sat_start_dates[sat_abbrevs.index('n07')], sat_end_dates[sat_abbrevs.index('f17')], date):
            sat_name = 'dmsp_cdr_or_gsfc'
        elif tt.is_time_in_range(sat_start_dates[sat_abbrevs.index('f18_nrt')], sat_end_dates[sat_abbrevs.index('f18_nrt')], date):
            sat_name = 'dmsp_nrt'
        else:
            sat_name = 'dmsp_cdr_or_gsfc'   # throwaway, since it will return exists = False regardless
            # raise ValueError('Satellite name given as DMSP but given date not within NRT or CDR/GSFC date ranges.')

    # construct filepath and and check for existence
    if sat_name is 'nimbus5':
        filename_part1 = 'ESMR-'
        filename_part2 = '.tse.00.h5'
        date_365 = tt.convert_date_to_365(date)
        filepath = nimbus5_dir + filename_part1 + '{0[0]}{1:03d}'.format(date, date_365) + filename_part2
        if not tt.is_time_in_range(sat_start_dates[sat_abbrevs.index('n05')], sat_end_dates[sat_abbrevs.index('n05')], date):
            exists = False
        else:
            exists = os.path.isfile(filepath)
    elif sat_name is 'dmsp_cdr_or_gsfc':
        filename_part1 = 'seaice_conc_daily_sh_'
        filename_part2 = '_v03r01.nc'
        sat_abbrev = 'NAN' # default value to create a meaningless filename for dates outside DMSP range
        for sat in range(sat_abbrevs.index('n07'),sat_abbrevs.index('f17') + 1):
            if tt.is_time_in_range(sat_start_dates[sat], sat_end_dates[sat], date):
                sat_abbrev = sat_abbrevs[sat]
        filepath = dmsp_dir + filename_part1 + sat_abbrev + '_' + '{0[0]}{0[1]:02d}{0[2]:02d}'.format(date) + filename_part2
        if not tt.is_time_in_range(sat_start_dates[sat_abbrevs.index('n07')], sat_end_dates[sat_abbrevs.index('f17')], date):
            exists = False
        else:
            exists = os.path.isfile(filepath)
    elif sat_name is 'dmsp_nrt':
        filename_part1 = 'seaice_conc_daily_icdr_sh_'
        filename_part2 = '_v01r00.nc'
        sat_abbrev = 'NAN' # default value to create a meaningless filename for dates outside DMSP range
        for sat in range(sat_abbrevs.index('f18_nrt'),sat_abbrevs.index('f18_nrt') + 1):
            if tt.is_time_in_range(sat_start_dates[sat], sat_end_dates[sat], date):
                sat_abbrev = sat_abbrevs[sat][:-4]   # to strip away '_nrt' suffix on abbreviation
        filepath = dmsp_nrt_dir + filename_part1 + sat_abbrev + '_' + '{0[0]}{0[1]:02d}{0[2]:02d}'.format(date) + filename_part2
        if not tt.is_time_in_range(sat_start_dates[sat_abbrevs.index('f18_nrt')], sat_end_dates[sat_abbrevs.index('f18_nrt')], date):
            exists = False
        else:
            exists = os.path.isfile(filepath)
    elif sat_name is 'amsre':
        filename_part1 = 'asi-s6250-'
        filename_part2 = '-v5.h5'
        filepath = amsre_dir + filename_part1 + '{0[0]}{0[1]:02d}{0[2]:02d}'.format(date) + filename_part2
        if not tt.is_time_in_range(sat_start_dates[sat_abbrevs.index('ame')], sat_end_dates[sat_abbrevs.index('ame')], date):
            exists = False
        else:
            exists = os.path.isfile(filepath)
    elif sat_name is 'amsr2':
        filename_part1 = 'asi-AMSR2-s6250-'
        filename_part2 = '-v5.h5'
        filepath = amsr2_dir + filename_part1 + '{0[0]}{0[1]:02d}{0[2]:02d}'.format(date) + filename_part2
        if not tt.is_time_in_range(sat_start_dates[sat_abbrevs.index('am2')], sat_end_dates[sat_abbrevs.index('am2')], date):
            exists = False
        else:
            exists = os.path.isfile(filepath)
    else:
        raise ValueError('Given satellite name does match those hard-coded in function.')

    return [filepath, exists]


def load_amsr_grid(grid_file,area_file,regrid_to_25km=False):
    """ Processes and returns AMSR-E/AMSR2 polar stereographic lat/lon grid and pixel areas (in km^2).
    
    If regrid_to_25km is True, returns mean lat/lon of blocks of 16 grid cells from 6.25 km grid.
    
    Note: pixel area files are from NSIDC polar stereographic tool website:
        http://nsidc.org/data/polar-stereo/tools_geo_pixel.html.

    """
    with h5py.File(grid_file,'r') as grid:
        grid_dict = {'lats':grid['Latitudes'].value, 'lons':grid['Longitudes'].value}
    grid_dict['lons'] = gt.convert_360_lon_to_180(grid_dict['lons'])
    areas_flat = fromfile(area_file,dtype=int32) / 1000
    areas = reshape(areas_flat,(1328,1264))
    areas = flipud(areas)
    grid_dict['areas'] = areas
    if regrid_to_25km is not True:
        return grid_dict
    else:
        old_h = shape(grid_dict['areas'])[0]
        old_w = shape(grid_dict['areas'])[1]
        grid_dict['lons'] = grid_dict['lons'].reshape([old_h,old_w//4,4]).mean(2).T.reshape(old_w//4,old_h//4,4).mean(2).T
        grid_dict['lats'] = grid_dict['lats'].reshape([old_h,old_w//4,4]).mean(2).T.reshape(old_w//4,old_h//4,4).mean(2).T
        grid_dict['areas'] = grid_dict['areas'].reshape([old_h,old_w//4,4]).sum(2).T.reshape(old_w//4,old_h//4,4).sum(2).T
        return grid_dict


def load_nsidc_ps_25km_grid(grid_dir):
    """ Processes and returns NSIDC 25 km polar stereographic lat/lon grid and pixel areas (in km^2).

    Applicable to Nimbus-5 and SMMR/DMSP datasets.

    Further information here: http://nsidc.org/data/polar-stereo/tools_geo_pixel.html

    Lat/lon and area files downloaded from:
    ftp://sidads.colorado.edu/pub/DATASETS/brightness-temperatures/polar-stereo/tools/geo-coord/grid/

    """
    area_file = grid_dir + 'pss25area_v3.dat'
    lat_file = grid_dir + 'pss25lats_v3.dat'
    lon_file = grid_dir + 'pss25lons_v3.dat'

    grid_dict = {}
    lats_flat = fromfile(lat_file, dtype=int32) / 100000
    grid_dict['lats'] = reshape(lats_flat,(332,316))
    lons_flat = fromfile(lon_file, dtype=int32) / 100000
    grid_dict['lons'] = reshape(lons_flat,(332,316))
    areas_flat = fromfile(area_file, dtype=int32) / 1000
    grid_dict['areas'] = reshape(areas_flat,(332,316))
    return grid_dict
