# -*- coding: utf-8 -*-

from numpy import *
from datetime import datetime, timedelta
from dateutil.relativedelta import *
import os
import re
import codecs
import pandas as pd
import scipy.io.netcdf as spnc
from ecmwfapi import ECMWFDataServer

import time_tools as tt
import geo_tools as gt
import download_file as df


def argo_gdac(start_date,end_date,lat_range,lon_range,save_to_root,
              overwrite_global_index=True,overwrite_profs=False,bypass_download=False,
              only_download_wmoids=[]):
    """ Downloads Argo float profiles from US-GODAE GDAC.

    Args:
        start_date, end_date: datetime tuples, e.g. (Y,M,D) or (Y,M,D,H) or... etc.
        lat_range, lon_range: list-pairs (i.e. [min,max]) of lats from -90 to 90 or lons from -180 to 180 or 0 to 360
            note: to search over all longitudes, use [-180,180], [0,360], [0,0], or [lon,same_lon]... all work!
            note: when lat/lon unavailable for a profile (e.g. no position fix or under ice), last valid lat/lon for
                the float in question will be referenced
        save_to_root: path of main Argo data directory of interest
        only_download_wmoids: [] to download all
                              e.g. [5904468, 5904471, ...] to only download new profiles for specified WMOids
    """
    save_to_meta = save_to_root + 'Meta/'
    save_to_profiles = save_to_root + 'Profiles/'
    url_root = 'http://www.usgodae.org/ftp/outgoing/argo/'
    global_index_filename = 'ar_index_global_prof.txt'
    local_index_filename = 'ar_index_local_prof.txt'  # index of locally downloaded profiles
    url_profiles_root = url_root + 'dac/'

    # download most recent global profile list and parse columns
    df.single_file(url_root,global_index_filename,save_to_meta,ftp_root=False,overwrite=overwrite_global_index,verbose=True)
    data_frame = pd.read_csv(save_to_meta + global_index_filename,header=8,low_memory=False)
    global_profile_list = data_frame.values

    # identify profiles meeting argument criteria
    num_profs = len(global_profile_list)
    prof_matches = zeros(num_profs, dtype=bool)
    float_number_regexp = re.compile('[a-z]*/[0-9]*/profiles/[A-Z]*([0-9]*)_[0-9]*[A-Z]*.nc')
    last_valid_position_float = int(float_number_regexp.findall(global_profile_list[0,0])[0])
    last_valid_position = [global_profile_list[0,2],global_profile_list[0,3]]
    for n in range(num_profs):
        current_float = int(float_number_regexp.findall(global_profile_list[n,0])[0])
        # accommodate profiles with missing lat/lon data (set as 99999.000)
        if global_profile_list[n,2] == 99999.000 or global_profile_list[n,3] == 99999.000 \
                or global_profile_list[n,2] == -999.000 or global_profile_list[n,3] == -999.000:
            if current_float == last_valid_position_float:
                assumed_prof_position = last_valid_position
            else:
                continue # in effect, leave prof_matches[n] = False
                ### original solution was the following: raise AssertionError('Profile has invalid lat/lon and is unusable because no prior valid lat/lon for this float, {0}.'.format(current_float))
        else:
            assumed_prof_position = [global_profile_list[n,2],global_profile_list[n,3]]
            last_valid_position = assumed_prof_position
            last_valid_position_float = current_float
        # skip profiles with missing timestamps
        if isnan(global_profile_list[n,1]):
            continue  # in effect, leave prof_matches[n] = False
        # finally, if profile has valid position and timestamp, then check against args
        if tt.is_time_in_range(start_date,end_date,tt.convert_14_to_tuple(global_profile_list[n,1])):
            if gt.geo_in_range(assumed_prof_position[0],assumed_prof_position[1],lat_range,lon_range):
                prof_matches[n] = True
    print('>>> Number of Argo profiles on GDAC meeting criteria = ',sum(prof_matches))

    # using profile matches, create index of local float profile metadata (same format as global index)
    # add columns for float number, profile number, profile status (R, D), profile suffix (D = descending profile)
    matching_profs = where(prof_matches)[0]
    local_profile_list = global_profile_list[matching_profs,:]
    num_profs = len(local_profile_list)

    # download necessary profiles to local
    if not bypass_download:
        if len(only_download_wmoids) is not 0:
            only_download_wmoids = [str(selected_wmoid) for selected_wmoid in only_download_wmoids]
            trim_local_profile_list_indices = []
            starting_dir = os.getcwd()
            os.chdir(save_to_profiles)
            existing_prof_files = os.listdir()
        prof_file_regexp = re.compile('[a-z]*/[0-9]*/profiles/([A-Z]*[0-9]*_[0-9]*[A-Z]*.nc)')
        prof_path_regexp = re.compile('([a-z]*/[0-9]*/profiles/)[A-Z]*[0-9]*_[0-9]*[A-Z]*.nc')
        for i, global_prof_index in enumerate(matching_profs):
            prof_file = prof_file_regexp.findall(global_profile_list[global_prof_index,0])[0]
            prof_path = prof_path_regexp.findall(global_profile_list[global_prof_index,0])[0]
            if len(only_download_wmoids) is not 0:
                if all([selected_wmoid not in prof_file for selected_wmoid in only_download_wmoids]):
                    if prof_file in existing_prof_files: trim_local_profile_list_indices.append(i)
                    continue
                print('dlp.argo_gdac() is downloading ' + prof_file)
                trim_local_profile_list_indices.append(i)
            df.single_file(url_profiles_root + prof_path,prof_file,save_to_profiles,ftp_root=False,overwrite=overwrite_profs,verbose=False)
            df.how_far(i,matching_profs,0.01)
        if len(only_download_wmoids) is not 0:
            matching_profs = matching_profs[trim_local_profile_list_indices]
            local_profile_list = local_profile_list[trim_local_profile_list_indices,:]
            num_profs = len(local_profile_list)
            os.chdir(starting_dir)

    # re-process local profile index
    float_wmoid_regexp = re.compile('[a-z]*/([0-9]*)/profiles/[A-Z]*[0-9]*_[0-9]*[A-Z]*.nc')
    float_profile_filename_regexp = re.compile('[a-z]*/[0-9]*/profiles/([A-Z]*[0-9]*_[0-9]*[A-Z]*.nc)')
    float_profile_mode_regexp = re.compile('[a-z]*/[0-9]*/profiles/([A-Z]*)[0-9]*_[0-9]*[A-Z]*.nc')
    float_profile_num_regexp = re.compile('[a-z]*/[0-9]*/profiles/[A-Z]*[0-9]*_([0-9]*)[A-Z]*.nc')
    float_wmoids = [int(float_wmoid_regexp.findall(local_profile_list[n,0])[0]) for n in range(num_profs)]
    float_profile_filenames = [float_profile_filename_regexp.findall(local_profile_list[n,0])[0] for n in range(num_profs)]
    float_profile_modes = [float_profile_mode_regexp.findall(local_profile_list[n,0])[0] for n in range(num_profs)]
    float_profile_nums = [int(float_profile_num_regexp.findall(local_profile_list[n, 0])[0]) for n in range(num_profs)]
    float_position_flags = [0 for n in range(num_profs)]
    local_profile_list = hstack((vstack(float_wmoids),vstack(float_profile_filenames),vstack(float_profile_modes),
                                 vstack(float_position_flags),local_profile_list))

    # sort profile index by WMOid + profile number (e.g. 7900093 is completely out of order)
    sort_param = array(float_wmoids) + array(float_profile_nums) / 10000
    local_profile_list = local_profile_list[argsort(sort_param)]

    # flat and interpolate between missing positions
    # note: ignores lat/lon of additional profiles when NUM_PROF > 1
    # note: will likely fail if first or last profiles in the index have bad positions
    currently_interpolating = 0
    previous_prof_wmoid = local_profile_list[0,0]
    bad_starting_position = 0
    starting_position = [0, 0]  # [lat,lon]
    ending_position = [0, 0]
    interp_profile_indices = []
    datetime_stamps = []
    for p in range(num_profs):
        if p > 1: previous_prof_wmoid = local_profile_list[p-1, 0]
        current_prof_wmoid = local_profile_list[p,0]
        profile_file = spnc.netcdf_file(save_to_profiles + local_profile_list[p,1], 'r', mmap=False)
        profile_mode = str(profile_file.variables['DATA_MODE'][0])[2]
        local_profile_list[p,2] = profile_mode  # R, D, or A (adjusted real-time)
        profile_lat_given = profile_file.variables['LATITUDE'][0]
        local_profile_list[p,6] = profile_lat_given
        profile_lon_given = profile_file.variables['LONGITUDE'][0]
        local_profile_list[p,7] = profile_lon_given
        profile_position_qc = int(str(profile_file.variables['POSITION_QC'][0])[2])
        profile_time = tt.convert_tuple_to_datetime(tt.convert_14_to_tuple(local_profile_list[p,5]))
        profile_number = profile_file.variables['CYCLE_NUMBER'][0]
        profile_file.close()

        if current_prof_wmoid != previous_prof_wmoid and currently_interpolating == 1:
            interp_profile_indices.append(p)
            for n, pint in enumerate(interp_profile_indices[1:-1]):
                local_profile_list[pint, 3] = 9  # 'ETHAN_POSITION_QC' of 9 = bad, failed interpolation attempt
                                                 # (reached last of float's profiles without finding a good position)
                local_profile_list[pint, 6] = NaN
                local_profile_list[pint, 7] = NaN
            currently_interpolating = 0  # reinitialize tracker and counter variables
            bad_starting_position = 0
            starting_position = [0, 0]
            ending_position = [0, 0]
            interp_profile_indices = []
            datetime_stamps = []
        if gt.geo_in_range(profile_lat_given,profile_lon_given,[-90,90],[-180,180]) \
            and (profile_position_qc == 1 or profile_position_qc == 2):
            if currently_interpolating == 0:
                local_profile_list[p,3] = 1  # 'ETHAN_POSITION_QC' of 1 = likely good
            elif currently_interpolating == 1: # here ends the interpolated track
                local_profile_list[p, 3] = 1  # 'ETHAN_POSITION_QC' of 1 = likely good
                currently_interpolating = 0
                if bad_starting_position == 0:
                    ending_position = [profile_lat_given,profile_lon_given]
                    interp_profile_indices.append(p)
                    datetime_stamps.append(profile_time)
                    if len(interp_profile_indices) > 2:
                        interp_positions = gt.great_circle_interp(starting_position,ending_position,datetime_stamps)
                        for n, pint in enumerate(interp_profile_indices[1:-1]):
                            local_profile_list[pint, 3] = 2  # 'ETHAN_POSITION_QC' of 2 = interpolated; assumed under ice
                            local_profile_list[pint, 6] = interp_positions[n][0]
                            local_profile_list[pint, 7] = interp_positions[n][1]
                    else: # weird case of float's first profile with position flag '8', second profile with '1', and
                          #     same positions listed for both (e.g. 5901722)
                        local_profile_list[p-1, 3] = 9  # 'ETHAN_POSITION_QC' of 9 = bad
                        local_profile_list[p-1, 6] = NaN
                        local_profile_list[p-1, 7] = NaN
                    starting_position = [0, 0] # reinitialize tracker and counter variables
                    ending_position = [0, 0]
                    interp_profile_indices = []
                    datetime_stamps = []
                elif bad_starting_position == 1:
                    bad_starting_position = 0
        elif profile_number == 1 and current_prof_wmoid != previous_prof_wmoid and profile_position_qc == 8 \
                and gt.geo_in_range(profile_lat_given,profile_lon_given,[-90,-50],[-180,180]):
            # special case where float's first profile is under ice, and thus was marked '8' (interp'd)
            #       with lat/lon likely from deployment location
            # note: criterion of profile number = 1 used to avoid floats that drifted into download lat/lon box while
            #       under ice (i.e. first profile downloaded was marked '8' with GDAC-interp'd lat/lon)
            currently_interpolating = 1
            starting_position = [local_profile_list[p, 6], local_profile_list[p, 7]]
            bad_starting_position = 0
            interp_profile_indices = [p]
            datetime_stamps = [profile_time]
            local_profile_list[p, 3] = 2  # 'ETHAN_POSITION_QC' of 2 = under-ice first profile, lat/lon from deployment
        elif current_prof_wmoid == previous_prof_wmoid \
                and (profile_position_qc == 9 or (profile_position_qc == 8
                      and gt.geo_in_range(profile_lat_given,profile_lon_given,[-90,-50],[-180,180]))):
            if currently_interpolating == 0:
                currently_interpolating = 1
                if local_profile_list[p-1, 3] == 1: # good starting position
                    starting_position = [local_profile_list[p-1,6],local_profile_list[p-1,7]]
                    bad_starting_position = 0
                    interp_profile_indices = [p-1, p]
                    datetime_stamps = [tt.convert_tuple_to_datetime(tt.convert_14_to_tuple(local_profile_list[p-1,5]))]
                    datetime_stamps.append(profile_time)
                    local_profile_list[p, 3] = 0  # 'ETHAN_POSITION_QC' of 0 = pending interpolation attempt
                else: # bad starting position
                    bad_starting_position = 1
                    local_profile_list[p, 3] = 9  # 'ETHAN_POSITION_QC' of 9 = bad, failed interpolation attempt
                    local_profile_list[p, 6] = NaN
                    local_profile_list[p, 7] = NaN
            elif currently_interpolating == 1:
                if bad_starting_position == 0:
                    interp_profile_indices.append(p)
                    datetime_stamps.append(profile_time)
                    local_profile_list[p, 3] = 0  # 'ETHAN_POSITION_QC' of 0 = pending interpolation attempt
                elif bad_starting_position == 1:
                    local_profile_list[p, 3] = 9  # 'ETHAN_POSITION_QC' of 9 = bad, failed interpolation attempt
                    local_profile_list[p, 6] = NaN
                    local_profile_list[p, 7] = NaN
        else:
            if currently_interpolating == 0:
                local_profile_list[p, 3] = 9  # 'ETHAN_POSITION_QC' of 9 = bad, for many possible reasons
                local_profile_list[p, 6] = NaN
                local_profile_list[p, 7] = NaN
            elif currently_interpolating == 1:
                local_profile_list[p, 3] = 9  # 'ETHAN_POSITION_QC' of 9 = bad, for many possible reasons
                local_profile_list[p, 6] = NaN
                local_profile_list[p, 7] = NaN
                interp_profile_indices.append(p)
                for n, pint in enumerate(interp_profile_indices[1:-1]):
                    local_profile_list[pint, 3] = 9  # 'ETHAN_POSITION_QC' of 9 = bad, failed interpolation attempt
                                                                                # (ended on a bad lat/lon)
                    local_profile_list[pint, 6] = NaN
                    local_profile_list[pint, 7] = NaN
                currently_interpolating = 0  # reinitialize tracker and counter variables
                bad_starting_position = 0
                starting_position = [0, 0]
                ending_position = [0, 0]
                interp_profile_indices = []
                datetime_stamps = []
        df.how_far(p,range(num_profs),0.01)

    # save updated local profile index
    savetxt(save_to_meta + local_index_filename, local_profile_list, fmt='%i,%s,%s,%i,%s,%i,%f,%f,%s,%s,%s,%i')


def argo_soccom(save_to_root,overwrite_profs=True):
    """ Downloads and processes SOCCOM float profiles in text format from FloatViz FTP server.

    Args:
        save_to_root: path of main Argo data directory of interest
    """
    save_to_floats = save_to_root + 'SOCCOM_HiResQC_ftp_' + datetime.today().strftime('%Y-%m-%d') + '/'
    os.mkdir(save_to_floats)
    ftp_root = 'ftp.mbari.org'
    url_root = 'pub/SOCCOM/FloatVizData/HRQC/'

    df.all_files(ftp_root,url_root,save_to_floats,overwrite=overwrite_profs)

    # do a find-and-replace on data files to remove whitespace between some column names
    for data_filename in os.listdir(save_to_floats):
        orig_file_as_list = codecs.open(save_to_floats + data_filename,'rb',encoding='latin-1').readlines()
        new_file_as_list = []
        for line in orig_file_as_list:
            first_edit = line.replace('Lon [°E]', 'Lon[°E]')
            second_edit = first_edit.replace('Lat [°N]', 'Lat[°N]')
            new_file_as_list.append(second_edit)
        out_file = codecs.open(save_to_floats + data_filename,'wb',encoding='latin-1')
        out_file.writelines(new_file_as_list)
        out_file.close()


def amsr(which_amsr, start_date, end_date, save_to, get_pdfs=True, overwrite=False, convert=False, conversion_script_dir=None):
    """ Downloads AMSR-E or AMSR2 sea ice concentration product.

    Converts data from HDF4 to HDF5 format by calling df.convert_to_hdf5() if 'convert'
    is True, then deletes original HDF4 file.

    AMSR-2:
        AMSR2 6.25 km daily sea ice concentration product is ARTIST Sea Ice (ASI)
        algorithm from 89 GHz channel, a preliminary data product that uses the
        AMSR-E calibrations. Consider switching to JAXA GCOM-W1 AMSR2 sea ice
        product when "research" calibrated version becomes available, or NSIDC
        DAAC validated versions (supposedly in late 2016).

        Example file path: http://www.iup.uni-bremen.de:8084/amsr2data/asi_daygrid_swath/s6250/2015/aug/Antarctic/asi-AMSR2-s6250-20150801-v5.hdf

        Note that 3.125 km gridded ARTIST AMSR2 is available from the following
        link, but the lower 6.25 km resolution is used here for consistency with
        AMSR-E products: ftp://ftp-projects.zmaw.de/seaice/AMSR2/

    AMSR-E:
        AMSR-E 6.25 km daily sea ice concentration product is ARTIST Sea Ice (ASI)
        algorithm from 89 GHz channel.

        Example file path: http://iup.physik.uni-bremen.de:8084/amsredata/asi_daygrid_swath/l1a/s6250/2011/oct/Antarctic/asi-s6250-20111004-v5.hdf

        Another option for AMSR-E is the 12.5 km v3 NSIDC product available here:
        http://nsidc.org/data/AE_SI12

        It seems that the 6.25 km ASI product is also available at the following link,
        but no 3.125 km product is available: ftp://ftp-projects.zmaw.de/seaice/AMSR-E_ASI_IceConc/

    SSMIS product from University of Bremen on 6.25 km grid to bridge gap between AMSR-E and AMSR2:
        SSMIS interim: http://iup.physik.uni-bremen.de:8084/ssmisdata/asi_daygrid_swath/s6250/
    
    Required data acknowledgement: Spreen et al. (2008), doi:10.1029/2005JC003384
    Optional data acknowledgement (for AMSR2): Beitsch et al. (2014), doi:10.3390/rs6053841
    
    Args:
        which_amsr: if 1, download AMSR-E; if 2, download AMSR2
        start_date and end_date: (Y,M,D), with start/end inclusive
        save_to: directory path
        get_pdfs: download image files
    Returns:
        None
    Raises:
        No handled exceptions
    
    """
    if which_amsr == 2:
        url_part1 = 'http://www.iup.uni-bremen.de:8084/amsr2data/asi_daygrid_swath/s6250/'
        url_part2 = '/Antarctic/'
        filename_part1 = 'asi-AMSR2-s6250-'
        filename_part2 = '-v5.hdf'
    elif which_amsr == 1:
        url_part1 = 'http://iup.physik.uni-bremen.de:8084/amsredata/asi_daygrid_swath/l1a/s6250/'
        url_part2 = '/Antarctic/'
        filename_part1 = 'asi-s6250-'
        filename_part2 = '-v5.hdf'
    filename_part2_pdf1 = '-v5_nic.pdf'
    filename_part2_pdf2 = '-v5_visual.pdf'
    months = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']

    starting_dir = os.getcwd()
    os.chdir(save_to)
    existing_files = os.listdir()
    os.chdir(starting_dir)

    all_dates = tt.dates_in_range(start_date, end_date)
    for index, d in enumerate(all_dates):
        url_dir = url_part1 + str(d[0]) + '/' + months[d[1]-1] + url_part2
        filename = filename_part1 + '{0[0]}{0[1]:02d}{0[2]:02d}'.format(d) + filename_part2
        new_filename = filename.split('.')[0] + '.h5'
        if (new_filename not in existing_files) or (new_filename in existing_files and overwrite is True):
            df.single_file(url_dir, filename, save_to, overwrite)
        if convert:
            df.convert_to_hdf5(conversion_script_dir, filename, save_to, save_to, overwrite=overwrite, delete_original=True)
        if get_pdfs:
            pdf1name = filename_part1 + '{0[0]}{0[1]:02d}{0[2]:02d}'.format(d) + filename_part2_pdf1
            pdf2name = filename_part1 + '{0[0]}{0[1]:02d}{0[2]:02d}'.format(d) + filename_part2_pdf2
            df.single_file(url_dir, pdf1name, save_to, overwrite)
            df.single_file(url_dir, pdf2name, save_to, overwrite)
        df.how_far(index,all_dates,0.01)


def dmsp_nrt(start_date, end_date, save_to, overwrite=False):
    """ Downloads NSIDC 25 km preliminary Near Real-Time (NRT) sea ice concentration product.

    NSIDC's v1 daily SSMIS product on 25 km grid in netCDF-4 (HDF5) format. Product derived from 3 channels. Data files
    contain the following:
    - NRT CDR (Climate Data Record) product based on DMSP SSMIS currently from 2016-01-01 to present, using purely
      automated application and merging of the NASA Team (NT) and Bootstrap (BT) algorithms.
    (The NRT product does not contain Goddard Merged fields.)

    Information: https://nsidc.org/data/g10016

    Example file path: ftp://sidads.colorado.edu/pub/DATASETS/NOAA/G10016/south/daily/2016/seaice_conc_daily_icdr_sh_f17_20160101_v01r00.nc

    Expert guidance on the related CDR record:
    https://climatedataguide.ucar.edu/climate-data/sea-ice-concentration-noaansidc-climate-data-record

    Required data acknowledgement given in full under 'Citing This Data' here: http://dx.doi.org/10.7265/N5FF3QJ6.

    """
    ftp_root = 'sidads.colorado.edu'
    url_root = 'pub/DATASETS/NOAA/G10016/south/daily/'

    filename_part1 = 'seaice_conc_daily_icdr_sh_'
    filename_part2 = '_v01r00.nc'

    sat_abbrevs = ['f17','f18']
    sat_start_dates = [(2016,1,1),(2016,4,1)]
    sat_end_dates = [(2016,3,30),tt.now()]

    all_dates = tt.dates_in_range(start_date, end_date)
    for index, d in enumerate(all_dates):
        if not tt.is_time_in_range(sat_start_dates[0],sat_end_dates[-1],d):
            raise ValueError('Given date range exceeds hard-coded satellite date ranges.')
        for sat in range(0,len(sat_abbrevs)):
            if tt.is_time_in_range(sat_start_dates[sat], sat_end_dates[sat], d):
                sat_abbrev = sat_abbrevs[sat]
        filename = filename_part1 + sat_abbrev + '_' + '{0[0]}{0[1]:02d}{0[2]:02d}'.format(d) + filename_part2

        starting_dir = os.getcwd()
        try:
            if starting_dir is not save_to:
                os.chdir(save_to)
            if filename not in os.listdir() or (filename in os.listdir() and overwrite is True):
                df.single_file(url_root + '{0[0]}/'.format(d), filename, save_to, ftp_root=ftp_root, overwrite=False, auth=None)
        finally:
            os.chdir(starting_dir)
        df.how_far(index, all_dates, 0.1)


def dmsp_v3(start_date, end_date, save_to, overwrite=False):
    """ Downloads NSIDC 25 km sea ice concentration product.

    NSIDC's v3 r1 daily SMMR + SSM/I + SSMIS product on 25 km grid in netCDF-4 (HDF5) format. Product derived from
    3 channels. Data files contain the following:
    - CDR (Climate Data Record) product based on DMSP SSM/I and SSMIS from 1987-07-09 onwards, using purely automated
      application and merging of the NASA Team (NT) and Bootstrap (BT) algorithms.
    - GSFC (NASA Goddard Space Flight Center) merged product based on the above, plus Nimbus-7 SMMR from 1978-11-01
      onwards (every other day). Some manual quality control, interpolation, and editing has been conducted (but without
      provenance), meaning that GSFC is a higher-quality but less uniform record than CDR. In any case, CDR excludes
      the SMMR period (as of now) due to "data quality issues of the input brightness temperatures" but also
      because "full provenance and documentation of the SMMR brightness temperatures and processing methodology
      ... cannot be assured."

    Information: https://nsidc.org/data/g02202

    Example file path: ftp://sidads.colorado.edu/pub/DATASETS/NOAA/G02202_V3/south/daily/1978/seaice_conc_daily_sh_n07_19781101_v03r01.nc

    Expert guidance on these records:
    https://climatedataguide.ucar.edu/climate-data/sea-ice-concentration-noaansidc-climate-data-record

    Required data acknowledgement given in full under 'Citing This Data' here: http://dx.doi.org/10.7265/N59P2ZTG.

    """
    ftp_root = 'sidads.colorado.edu'
    url_root = 'pub/DATASETS/NOAA/G02202_V3/south/daily/'

    filename_part1 = 'seaice_conc_daily_sh_'
    filename_part2 = '_v03r01.nc'

    sat_abbrevs = ['n07','f08','f11','f13','f17']
    sat_start_dates = [(1978,11,1),(1987,7,9),(1991,12,3),(1995,10,1),(2008,1,1)]
    sat_end_dates = [(1987,7,8),(1991,12,2),(1995,9,30),(2007,12,31),(2017,12,31)]

    all_dates = tt.dates_in_range(start_date, end_date)

    starting_dir = os.getcwd()
    if starting_dir is not save_to:
        os.chdir(save_to)
    dir_contents = os.listdir()

    for index, d in enumerate(all_dates):
        print(d) ### FOR TESTING
        if not tt.is_time_in_range(sat_start_dates[0],sat_end_dates[-1],d):
            raise ValueError('Given date range exceeds hard-coded satellite date ranges.')
        for sat in range(0,len(sat_abbrevs)):
            if tt.is_time_in_range(sat_start_dates[sat], sat_end_dates[sat], d):
                sat_abbrev = sat_abbrevs[sat]
        filename = filename_part1 + sat_abbrev + '_' + '{0[0]}{0[1]:02d}{0[2]:02d}'.format(d) + filename_part2

        if filename not in dir_contents or (filename in dir_contents and overwrite is True):
            # if tt.is_time_in_range((1986,9,25),(1987,1,1),d):   # misplaced files -- but fixed now
            #     df.single_file(url_root + '1987/',filename,save_to,ftp_root=ftp_root,
            #                    overwrite=False,auth=None)
            df.single_file(url_root + '{0[0]}/'.format(d), filename, save_to, ftp_root=ftp_root,
                           overwrite=False, auth=None)

        df.how_far(index, all_dates, 0.1)

    os.chdir(starting_dir)


def nimbus5(start_date, end_date, save_to, convert=False, conversion_script_dir=None):
    """ Downloads Nimbus-5 sea ice concentration product.

    Unzips files first. Converts data from HDF4 to HDF5 format by calling df.convert_to_hdf5()
    if 'convert' is True, then deletes original HDF4 file.

    NSIDC's v1 Nimbus-5 daily ESMR product on 25 km grid in compressed HDF4 format. Product based on
    a single channel (19 GHz), which is less accurate than SMMR and SSM/I products from after 1976.

    Information: http://nsidc.org/data/NSIDC-0009

    IMPORTANT NOTE: Downloading batch data via HTTPS requires login to EarthData. To do this, one must create an
    account: https://urs.earthdata.nasa.gov/users/new
    
    ... and then create a .netrc file via the command line using the following process:
        cd $HOME
        rm -f .netrc
        touch .netrc
        echo 'machine urs.earthdata.nasa.gov login [USERNAME] password [PASSWORD]' >> .netrc
                note: replace with your username and password
        chmod 0600 .netrc

    Example file path: https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0009_esmr_seaice/south/daily00/ESMR-1972346.tse.00.gz

    Required data acknowledgement given in full here: http://dx.doi.org/10.5067/W2PKTWMTY0TP.

    """
    url_dir = 'https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0009_esmr_seaice/south/daily00/'
    filename_part1 = 'ESMR-'
    filename_part2 = '.tse.00.gz'
    filename_part2_uncompressed = '.tse.00.hdf'
    filename_part2_uncompressed_converted = '.tse.00.h5'

    all_dates = tt.dates_in_range(start_date, end_date)
    for index, d in enumerate(all_dates):
        date_365 = tt.convert_date_to_365(d)
        filename = filename_part1 + '{0[0]}{1:03d}'.format(d,date_365) + filename_part2
        intermediate_filename = filename_part1 + '{0[0]}{1:03d}'.format(d, date_365) + filename_part2_uncompressed
        new_filename = filename_part1 + '{0[0]}{1:03d}'.format(d,date_365) + filename_part2_uncompressed_converted

        starting_dir = os.getcwd()
        try:
            if starting_dir is not dir:
                os.chdir(save_to)
            if new_filename not in os.listdir():
                df.single_file(url_dir, filename, save_to, overwrite=False, auth=None)
                df.un_gzip(save_to, filename, append_extension='.hdf', remove_compressed_file=True)
                df.convert_to_hdf5(conversion_script_dir, intermediate_filename, save_to, save_to, overwrite=False,delete_original=True)
        finally:
            os.chdir(starting_dir)
        df.how_far(index, all_dates, 0.1)


def ecmwf(date_range='1979-01-01/to/2017-08-31',area='-40/-90/-90/90',type='an',step='0',time='00/06/12/18',
          params=['msl','t2m','skt'],output_filename=None):
    """ Submits MARS request to retrieve ERA-Interim reanalysis fields as netCDF file.

    Arguments:
        date_range: for daily fields, format as, e.g., '1979-01-01/to/2017-08-31'
                    for monthly means of daily means, use [datetime(start_yr,start_mo,1),datetime(end_yr,end_mo,1)]
        area: subsetting area, format '-40/-90/-90/90' (N/W/S/E)
        type: 'an' for analysis or 'fc' for forecast
        step: '0' for analysis only, '6/12' or '3/6/9/12' for 6-hourly or 3-hourly forecasts from 0000 and 1200 UTC
              or None for monthly means (regardless, it will be ignored)
        time: analysis times, e.g. '00/06/12/18' for all analyses, or '00/12' if retrieving forecasts only
              or None for monthly means (regardless, it will be ignored)
        params: parameter abbreviations, to be translated into GRIB and Table 2 codes - see below for those available
                note: to find new codes, use parameter database: http://apps.ecmwf.int/codes/grib/param-db/
                      or use web interface and check "View the MARS request"
        output_filename: desired path + filename including '.nc' extension, to save locally
                         or None to save to temporary storage; download from: http://apps.ecmwf.int/webmars/joblist/
                note: if not downloading locally, cancel call using Ctrl-C after "Request is queued" appears
                      (otherwise file will be deleted almost instantly from ECMWF servers)

    None: cancelling call (Ctrl-C) after "Request is queued" appears is fine. It will prevent local download, though.

    Note: private login key required. See documentation for instructions on creating local login key.

    Note: file size limit is probably 20 GB. Check here: https://software.ecmwf.int/wiki/display/WEBAPI/News+feed

    Limited web API access:
        http://apps.ecmwf.int/datasets/data/interim-full-daily/levtype=sfc/
        http://apps.ecmwf.int/datasets/data/interim-full-moda/levtype=sfc/

    Documentation:
        https://software.ecmwf.int/wiki/display/WEBAPI/Access+ECMWF+Public+Datasets
        https://software.ecmwf.int/wiki/display/WEBAPI/Python+ERA-interim+examples
        https://software.ecmwf.int/wiki/display/UDOC/MARS+user+documentation
        https://software.ecmwf.int/wiki/display/UDOC/MARS+keywords
        http://apps.ecmwf.int/codes/grib/param-db

    Reference: Dee et al. 2011

    """
    param_codes = ''
    for param_idx, param in enumerate(params):
        # analysis parameters
        if   param == 't2m':  param_codes += '167.128'  # 2 metre temperature (K)
        elif param == 'sst':  param_codes +=  '34.128'  # Sea surface temperature (K)
        elif param == 'skt':  param_codes += '235.128'  # Skin temperature (K)
        elif param == 'd2m':  param_codes += '168.128'  # 2 metre dewpoint temperature (K)
        elif param == 'msl':  param_codes += '151.128'  # Mean sea level pressure (Pa)
        elif param == 'sp':   param_codes += '134.128'  # Surface pressure (Pa)
        elif param == 'u10':  param_codes += '165.128'  # 10 metre U wind component (m/s)
        elif param == 'v10':  param_codes += '166.128'  # 10 metre V wind component (m/s)
        elif param == 'si10': param_codes += '207.128'  # 10 metre wind speed (m/s) [NOTE: in monthly means only]
        # forecast parameters (* indicates accumulated field; note downward fluxes are positive)
        elif param == 'sf':   param_codes += '144.128'  # Snowfall (m of water equivalent) *
        elif param == 'sshf': param_codes += '146.128'  # Surface sensible heat flux (J/m^2) *
        elif param == 'slhf': param_codes += '147.128'  # Surface latent heat flux (J/m^2) *
        elif param == 'ssr':  param_codes += '176.128'  # Surface net solar radiation [shortwave] (J/m^2) *
        elif param == 'str':  param_codes += '177.128'  # Surface net thermal radiation [longwave] (J/m^2) *
        elif param == 'strd': param_codes += '175.128'  # Surface thermal radiation [longwave] downwards (J/m^2) *
        elif param == 'e':    param_codes += '182.128'  # Evaporation (m of water equivalent) *
        elif param == 'tp':   param_codes += '228.128'  # Total precipitation (m) *
        elif param == 'iews': param_codes += '229.128'  # Instantaneous eastward turbulent surface stress (N/m^2)
        elif param == 'inss': param_codes += '230.128'  # Instantaneous northward turbulent surface stress (N/m^2)
        if param_idx < len(params)-1: param_codes += '/'

    retrieve_dict = {
        "class":"ei",
        "dataset":"interim",
        "expver":"1",
        "format":"netcdf",
        "grid":"0.75/0.75",
        "levtype":"sfc",
        "param":param_codes,
        "type":type,
        'area':area,
        "target":output_filename,
        "use":'frequent',
    }

    # monthly means of daily means
    if len(date_range) == 2:
        retrieve_dict['stream'] = 'moda'
        final_date_range = ''
        working_month = date_range[0]
        while working_month < date_range[1]:
            final_date_range += working_month.strftime('%Y%m%d')
            final_date_range += '/'
            working_month += relativedelta(months=+1)
        final_date_range += date_range[1].strftime('%Y%m%d')
        retrieve_dict['date'] = final_date_range

    # daily fields
    else:
        retrieve_dict['stream'] = 'oper'
        retrieve_dict['date'] = date_range
        retrieve_dict['step'] = step
        retrieve_dict['time'] = time

    server = ECMWFDataServer()
    server.retrieve(retrieve_dict)


def isd_station(station_number, start_year, end_year, save_to, overwrite=True):
    """ Download sub-daily meteorological station data from NOAA NCEI Integrated Surface Database (ISD) ISD-Lite
        space-delimited annual data files.

    Args:
        station_number: six-digit integer station number, likely five-digit WMOID with trailing zero appended
        start_year: first year of met data
        end_year: last year of met data
        save_to: directory path
        overwrite: overwrite existing files?

    Data provenance and information:
        ISD homepage: https://www.ncdc.noaa.gov/isd
        root data directory: ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-lite
        info on file format: ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-lite/isd-lite-format.pdf
        brief technical document: ftp://ftp.ncdc.noaa.gov/pub/data/noaa/isd-lite/isd-lite-technical-document.pdf
        station numbers can be found using: https://www.ncdc.noaa.gov/homr/#ncdcstnid=30103999&tab=MSHR
        Antarctic station locations can be found at: http://nsidc.org/data/docs/daac/nsidc0190_surface_obs/spatial.html

    Citation (assumed, not given):
        Smith et al. (2011), BAMS, "The Integrated Surface Database: Recent developments and partnerships."
            doi:10.1175/2011BAMS3015.1

    Specific Antarctic station notes:
        WMOid 89512 (station number 895120) - Novolazarevskaja Station (70.7678°S, 11.8317°E) - 1973-2019
            http://www.aari.aq/stations/lazarev/lazarev_en.html
            https://www.ncdc.noaa.gov/homr/#ncdcstnid=30103999&tab=MSHR
        WMOid 89001 (station number 890010) - SANAE SAF-Base (70.3°S, 2.35°W) - 1973-1994
        WMOid 89004 (station number 890040) - SANAE AWS (71.7°S, 2.8°W) - 1997-2019
        WMOid 89002 (station number 890020) - Neumayer Station (70.667°S, 8.25°W) - 1981-2019
        WMOid 89504 (station number 895040) - Troll in Antarktis (72.017°S, 2.383°W) - 1994-2019
        WMOid 89514 (station number 895140) - Maitri (70.767°S, 11.75°E) - 1990-2019
        WMOid 89524 (station number 895240) - Asuka Japan-Base (71.533°S, 24.133°E) - 1987-1997
        WMOid 89003 (station number 890030) - Halvfarryggen (71.15°S, 6.683°W) - 2009-2017?

    """

    for year in range(start_year,end_year+1):
        df.single_file('pub/data/noaa/isd-lite/{0}/'.format(year),'{0}-99999-{1}.gz'.format(station_number,year),
                       save_to,ftp_root='ftp.ncdc.noaa.gov',overwrite=overwrite,verbose=True)
        df.un_gzip(save_to,'{0}-99999-{1}.gz'.format(station_number,year),
                   remove_compressed_file=True,overwrite=overwrite)
