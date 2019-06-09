# -*- coding: utf-8 -*-

import sys
from numpy import *
from scipy import stats
from scipy import ndimage
from scipy import interpolate
from scipy import constants
import pandas as pd
from datetime import datetime
import gsw
from geographiclib.geodesic import Geodesic, Math
geod = Geodesic.WGS84
import shapefile

import time_tools as tt
import load_product as ldp


def coriolis(lat):
    """ Calculates Coriolis frequency (or Coriolis parameter) f at a given latitude (in degrees, -90 to 90).
    """
    omega = 7.2921e-5    # Earth's rotation rate (rad/s)
    f = 2 * omega * sin(2*pi*lat/360)
    return f


def convert_360_lon_to_180(lons):
    """ Converts any-dimension array of longitudes from 0 to 360 to longitudes from -180 to 180.
    """
    lons = array(lons)
    outside_range = lons > 180
    lons[outside_range] = lons[outside_range] - 360
    return lons


def distance_between_two_coors(lat1,lon1,lat2,lon2):
    """ Returns distance between (lat1,lon1) and (lat2,lon2) in meters.
    
    Negative numbers required for SH latitudes, but this can handle any longitudes (0 to 360, -180 to 180), even if
    the given lat/lon pairs use different systems!
    
    """
    return geod.Inverse(lat1,lon1,lat2,lon2)['s12']


def geo_in_range(lat,lon,lat_range,lon_range):
    """ Is (lat,lon) within the lat/lon ranges specified? Returns True or False.

    Notes:
        - lat_range and lon_range must be LISTS, not TUPLES
        - comparison is inclusive (e.g. <=) not exclusive (e.g. <)
        - returns False for lats, lons equal to NaN or 99999.0
        - longitude quirks:
            - longitudes from -180 to 180 as well as 0 to 360 can be handled
            - searches within longitude range eastward from lon_range[0] to lon_range[1]
            - treats longitude ranges of [0,360], [0,0], [120,120], [-120,-120], for example, as spanning entire globe
            - cannot handle searching for a single longitude, e.g. [-20,-20] is interpreted as spanning entire globe

    """

    if lat == 99999.0 or lon == 99999.0 or isnan(lat) or isnan(lon):
        return False

    if 180 < lon <= 360:
        lon = lon - 360
    if 180 < lon_range[0] <= 360:
        lon_range[0] = lon_range[0] - 360
    if 180 < lon_range[1] <= 360:
        lon_range[1] = lon_range[1] - 360

    assert all([-90 <= lt <= 90 for lt in [lat, lat_range[0], lat_range[1]]]) and lat_range[0] <= lat_range[1]
    assert all([-180 <= ln <= 180 for ln in [lon, lon_range[0], lon_range[1]]])

    if lat_range[0] <= lat <= lat_range[1]:
        if lon_range[0] == lon_range[1]:
            return True
        if lon_range[0] > lon_range[1]:
            lon = lon + 360
            lon_range[1] = lon_range[1] + 360
        if lon_range[0] <= lon <= lon_range[1]:
            return True
        else:
            return False
    else:
        return False


def vert_prof_eval(profile_data,param_abbrev,z_or_z_range,z_coor='depth',interp_method='linear',extrap='NaN',
                  avg_method='interp',avg_spacing=0.1,avg_nan_tolerance=0.0,verbose_warn=True,verbose_error=True):
    """ Compute interpolated value at a depth/depths OR average value within range of depths for a vertical profile.

    NOTE: requires positive, monotonically-increasing vector of depths or pressures.

    Args:
        profile_data: dict with <<param_abbrev>> as key to a dict with 'data' and 'pres'/'depth' as keys to 1-D arrays
                      (e.g. this_float_data['profiles'][72])
        param_abbrev: string corresponding to parameter key in profile_data
                      (e.g. 'psal')
        z_or_z_range: three options:
            [a] single z value (to interpolate single value) [NOTE: array(scalar) will return a scalar, not an array)]
            [b] array or list of z values (to interpolate multiple values)
            [c] 1x2 tuple of (ztop,zbottom) where ztop < zbottom (to compute average value within range)
        z_coor: 'depth' or 'pres' (remember to give z_or_z_range in meters or decibars accordingly)
        interp_method: evaluate profile data using 'linear', 'nearest', or 'cubic' interpolation ('linear' recommended)
        extrap: extrapolate above/below z range via 'NaN', 'nearest', or 'extrap'
        avg_method: 'simple' for simply averaging all values found within the z_range
                    'interp' for interpolating data to regular spacing (determined by 'spacing') and then averaging
        avg_spacing: approximate spacing in meters or decibars of interpolated values for averaging
                     (relevant only if z_or_z_range is a tuple ([c]) and avg_method == 'interp')
                     (approximate in order to keep spacing perfectly even between upper and lower z values)
        avg_nan_tolerance: print error and return NaN if NaN fraction of original or interpolated data values
                                        in z range is > nan_tolerance
                           (float between 0.0 and 1.0)
                           (note: relevant only if z_or_z_range is a tuple ([c]))
        verbose_warn: print warnings
        verbose_error: print fatal errors (recommend keeping this True)

    Returns:
        None if error encountered
        computed value/values (which can be NaN) if successful

    """
    if param_abbrev not in profile_data.keys():
        if verbose_error: print('Error from geo_tools.vert_prof_eval(): profile does not include parameter of interest.')
        return None
    if 'data' not in profile_data[param_abbrev].keys() or z_coor not in profile_data[param_abbrev].keys():
        if verbose_error: print('Error from geo_tools.vert_prof_eval(): profile is missing a data-vector.')
        return None
    if len(profile_data[param_abbrev]['data']) == 0:
        if verbose_error: print('Error from geo_tools.vert_prof_eval(): profile is missing data within data-vector.')
        return None
    if not all(diff(profile_data[param_abbrev][z_coor]) > 0):
        if verbose_error: print('Error from geo_tools.vert_prof_eval(): depth or pressure vector is not monotonically '
                                'increasing. It could be backwards, jumbled, or incorrectly signed (should be positive).')
        return None

    # evaluate data at z value/values
    if not isinstance(z_or_z_range,tuple):
        return profile_interp(profile_data[param_abbrev]['data'],profile_data[param_abbrev][z_coor],z_or_z_range,
                              method=interp_method,out_of_bounds=extrap)

    # compute average within range (tuple) of z values
    else:
        if avg_method == 'simple':
            z_match = logical_and(profile_data[param_abbrev][z_coor] <= z_or_z_range[1],
                                  profile_data[param_abbrev][z_coor] >= z_or_z_range[0])
            if sum(z_match) == 0:
                if verbose_warn: print('Warning from geo_tools.vert_prof_eval(): no data within given depth range.')
                return NaN
            else:
                if sum(isnan(profile_data[param_abbrev]['data'][z_match])) / sum(z_match) > avg_nan_tolerance:
                    if verbose_warn: print('Warning from geo_tools.vert_prof_eval(): too many NaNs in given depth range.')
                    return NaN
                else:
                    return nanmean(profile_data[param_abbrev]['data'][z_match])
        elif avg_method == 'interp':
            z_eval = linspace(z_or_z_range[0],z_or_z_range[1],
                              int(ceil((z_or_z_range[1] - z_or_z_range[0]) / avg_spacing)))
            data_to_avg = profile_interp(profile_data[param_abbrev]['data'],profile_data[param_abbrev][z_coor],
                                         z_eval,method=interp_method,out_of_bounds=extrap)
            if isinstance(data_to_avg,float):
                if isnan(data_to_avg):
                    if verbose_warn: print('Warning from geo_tools.vert_prof_eval(): '
                                           'too little data; unable to interpolate.')
                    return NaN
            elif sum(isnan(data_to_avg)) / len(data_to_avg) > avg_nan_tolerance:
                if verbose_warn: print('Warning from geo_tools.vert_prof_eval(): too many NaNs in given depth range.')
                return NaN
            else:
                return nanmean(data_to_avg)


def vert_prof_even_spacing(profile_data,param_abbrev,z_coor='depth',spacing=0.1,interp_method='linear',extrap='NaN',
                           top=0.0,bottom='bottom',verbose_error=True):
    """ Interpolates vertical profile to even spacing. Helpful wrapper function for vert_prof_eval().

    Args:
        profile_data: dict with <<param_abbrev>> as key to a dict with 'data' and 'pres'/'depth' as keys to 1-D arrays
                      (e.g. this_float_data['profiles'][72])
        param_abbrev: string corresponding to parameter key in profile_data
                      (e.g. 'psal')
        z_coor: 'depth' or 'pres' (remember to give z values in meters or decibars accordingly)
        spacing: in meters or decibars (note: will start/end spacing inside range, e.g. given spacing of 0.25 and
                                        z-values from 5.1 to 1499.9, will return inclusive array from 5.25 to 1499.75;
                                        that said, will start/end spacing at given bounds if they line up with spacing)
        interp_method: see vert_prof_eval()
        extrap: see vert_prof_eval()
        top: <<scalar>> to start at given level or 'top' to start at uppermost measured level
        bottom: <<scalar>> to end at given level or 'bottom' to end at bottommost measured level
        verbose_error: print fatal errors (recommend keeping this True)

    Returns:
        z_vec, data_vec

    """
    if param_abbrev not in profile_data.keys():
        if verbose_error: print('Error from geo_tools.vert_prof_even_spacing(): profile does not include '
                                'parameter of interest.')
        return None
    if not all(diff(profile_data[param_abbrev][z_coor]) > 0):
        if verbose_error: print('Error from geo_tools.vert_prof_even_spacing(): depth or pressure vector is not '
                                'monotonically increasing. It could be backwards, jumbled, or incorrectly signed '
                                '(should be positive).')
        return None

    if top == 'top':
        top = profile_data[param_abbrev][z_coor][0]
    if bottom == 'bottom':
        bottom = profile_data[param_abbrev][z_coor][-1]

    z_vec = arange(0.0, bottom+spacing, spacing)
    z_vec = z_vec[logical_and(top <= z_vec, z_vec <= bottom)]
    data_vec = vert_prof_eval(profile_data,param_abbrev,z_vec,z_coor=z_coor,interp_method=interp_method,extrap=extrap,
                             verbose_error=verbose_error)
    return z_vec, data_vec


def vert_prof_running_mean(profile_data,param_abbrev,z_coor='depth',window=25.0,spacing=1.0,interp_method='linear',
                           extrap='NaN',top='top',bottom='bottom',verbose_error=True):
    """ Reduce noise of vertical profile using running mean with given window size.

    Args (see vert_prof_even_spacing() for those not described here):
        window: window period in meters or decibars (should be multiple of <<spacing>>)

    Returns:
        z_vec, data_vec

    """
    z_even, data_even = vert_prof_even_spacing(profile_data,param_abbrev,z_coor=z_coor,spacing=spacing,
                                               interp_method=interp_method,extrap=extrap,top=top,bottom=bottom,
                                               verbose_error=verbose_error)
    even = pd.DataFrame(data=data_even,index=z_even)
    window_in_indices = int(round(window/spacing))
    data_vec = even.rolling(window_in_indices,min_periods=0,center=True).mean().values.squeeze()
    return z_even, data_vec


def depth_at_which(profile_data,param_abbrev,value_attained,z_coor='depth',method='interp',top=0.0,bottom='bottom',
                   interp_initial_spacing=1.0,interp_final_spacing=0.01,verbose_warn=True,verbose_error=True):
    """ Estimate depth at which a given value is attained (intersected) in a vertical profile.

    Important notes on function behavior:
        Note that search direction is downwards from <<top>> pressure/depth level to <<bottom>> level.
        If parameter value at <<top>> is less than or equal to <<value_attained>>, function will search for first level
           at which <<value_attained>> is exceeded.
        If parameter value at <<top>> exceeds <<value_attained>>, function will search for first level at which
            parameter is less than <<value_attained>>.
        Function can also search for levels of max/min value between <<top>> and <<bottom>>.

    Args:
        profile_data: dict with <<param_abbrev>> as key to a dict with 'data' and 'pres'/'depth' as keys to 1-D arrays
                      (e.g. this_float_data['profiles'][72])
        param_abbrev: string corresponding to parameter key in profile_data
                      (e.g. 'sigma_theta')
        value_attained: three options for value of <<param_abbrev>> to search for:
            [a] scalar: search for this value
            [b] 'max': search for maximum value
            [c] 'min': search for minimum value
        z_coor: 'depth' or 'pres'
        method: 'actual' to choose measurement level preceding first measured level where value_attained is attained
                            (note that this will underestimate rather than overestimate the level)
                'interp' to use linear interpolation with 'nearest' interpolation to estimate exact level (recommended)
        top: <<scalar>> to start searching at given level or 'top' to start at uppermost measured level
        bottom: <<scalar>> to end searching at given level or 'bottom' to end at bottommost measured level
        interp_initial_spacing: spacing in meters/decibars used for interpolation during initial, coarse search
        interp_final_spacing: spacing in meters/decibars used for interpolation during final, fine search
                              (must be ≤ crit_interp_initial_spacing)
                              (note: these spacing args are only used if 'interp' selected for 'method')
        verbose_warn: print warnings
        verbose_error: print fatal errors (recommend keeping this True)

    Returns:
        level (depth in meters or pressure in decibars) at which <<value_attained>> attained
        NaN if <<value_attained>> is not attained between <<top>> and <<bottom>>
        None if error encountered

    """
    if param_abbrev not in profile_data.keys():
        if verbose_error: print('Error from geo_tools.depth_at_which(): this profile does not include given parameter.')
        return None
    if 'data' not in profile_data[param_abbrev].keys() or z_coor not in profile_data[param_abbrev].keys():
        if verbose_error: print('Error from geo_tools.depth_at_which(): this profile is missing data.')
        return None
    if not all(diff(profile_data[param_abbrev][z_coor]) > 0):
        if verbose_error: print('Error from geo_tools.depth_at_which(): depth or pressure vector is not monotonically '
                                'increasing. It could be backwards, jumbled, or incorrectly signed (should be positive).')
        return None
    if verbose_warn:
        if any(isnan(profile_data[param_abbrev]['data'])) or any(isnan(profile_data[param_abbrev][z_coor])):
            print('Warning from geo_tools.depth_at_which(): parameter, depth, or pressure vector contains NaNs.')

    # get search bounds
    if top == 'top':
        top = profile_data[param_abbrev][z_coor][0]
    if bottom == 'bottom':
        bottom = profile_data[param_abbrev][z_coor][-1]

    # determine whether parameter values are increasing or decreasing
    if value_attained != 'max' and value_attained != 'min':
        first_value = vert_prof_eval(profile_data,param_abbrev,top,z_coor=z_coor,interp_method='linear',
                                     extrap='nearest',verbose_warn=True,verbose_error=True)
        if first_value <= value_attained: expect_increasing = True
        else:                             expect_increasing = False

    # search for actual measurement levels
    if method == 'actual':
        levels_in_range_mask = logical_and(profile_data[param_abbrev][z_coor] >= top,
                                           profile_data[param_abbrev][z_coor] <= bottom)
        levels_in_range = profile_data[param_abbrev][z_coor][levels_in_range_mask]
        data_in_range   = profile_data[param_abbrev]['data'][levels_in_range_mask]

        if value_attained == 'max':
            attained_idx = argmax(data_in_range)
        elif value_attained == 'min':
            attained_idx = argmin(data_in_range)
        else:
            if expect_increasing:
                attained = (data_in_range >= value_attained)
            elif not expect_increasing:
                attained = (data_in_range <= value_attained)

            if sum(attained) == 0:
                return NaN
            else:
                attained_idx = argmax(attained) - 1  # note: np.argmax returns index of first 'True', or 0 if all False (!)
        if attained_idx == -1:
            return NaN
            # return profile_data[param_abbrev][z_coor][argmax(profile_data[param_abbrev][z_coor] >= top) - 1]
        else:
            return levels_in_range[attained_idx]

    # use interpolation to estimate depth of interest
    elif method == 'interp':
        # initial, coarse search for vicinity of depth
        lev_coarse, data_coarse = vert_prof_even_spacing(profile_data,param_abbrev,z_coor=z_coor,
                                                         spacing=interp_initial_spacing,interp_method='linear',
                                                         extrap='nearest',top=top,bottom=bottom,verbose_error=True)
        if value_attained == 'max':
            attained_idx_coarse = argmax(data_coarse)
        elif value_attained == 'min':
            attained_idx_coarse = argmin(data_coarse)
        else:
            if expect_increasing:
                attained = (data_coarse >= value_attained)
            elif not expect_increasing:
                attained = (data_coarse <= value_attained)

            if sum(attained) == 0:
                return NaN
            else:
                attained_idx_coarse = argmax(attained) - 1

        # final, fine search for depth
        if attained_idx_coarse == 0: top_idx_coarse = 0
        else:                        top_idx_coarse = attained_idx_coarse - 1
        if attained_idx_coarse == len(lev_coarse)-1: bottom_idx_coarse = len(lev_coarse)-1
        else:                                        bottom_idx_coarse = attained_idx_coarse + 1
        lev_fine, data_fine = vert_prof_even_spacing(profile_data,param_abbrev,z_coor=z_coor,
                                                     spacing=interp_final_spacing,interp_method='linear',
                                                     extrap='nearest',top=lev_coarse[top_idx_coarse],
                                                     bottom=lev_coarse[bottom_idx_coarse],verbose_error=True)
        if value_attained == 'max':
            attained_idx_fine = argmax(data_fine)
        elif value_attained == 'min':
            attained_idx_fine = argmin(data_fine)
        else:
            if expect_increasing:
                attained = (data_fine >= value_attained)
            elif not expect_increasing:
                attained = (data_fine <= value_attained)

            if sum(attained) == 0:
                return NaN
            else:
                attained_idx_fine = argmax(attained) - 1

        return lev_fine[attained_idx_fine]

    else:
        if verbose_error: print('Error from geo_tools.depth_at_which(): check argument passed for method.')


def mld(profile_data,ref_depth=10,ref_range_method='interp',ref_reject=False,sigma_theta_crit=0.03,crit_method='interp',
        crit_interp_initial_spacing=1.0,crit_interp_final_spacing=0.01,bottom_return='bottom',
        verbose_warn=True,verbose_error=True):
    """ Compute mixed layer depth (MLD) given a vertical profile of sigma-theta (potential density anomaly).
    
    Args:
        profile_data: dict with 'sigma_theta' as key to a dict with 'data' and 'depth' (!) as keys to 1-D arrays
                      (e.g. this_float_data['profiles'][72])
                      (note that a positive, monotonically increasing depth vector required, not pressure)
        ref_depth: three options for reference depth(s) in meters:
            [a] single scalar value at which sigma_theta evaluated using linear interp with 'nearest' extrapolation
            [b] range of values expressed as tuple of scalars: (upper,lower), where lower > upper
            [c] 'shallowest' (string), indicating the shallowest available measurement
        ref_range_method: if [b] above, calculate average in range using 'simple' or 'interp'? (see vert_prof_eval())
                         (for 'interp', linear interpolation with 'nearest' extrapolation used before averaging)
                         (if [b] not selected, this arg is ignored)
        ref_reject: False (default) or True (to return 'NaN' if ref_depth is [a] or [b] above and shallowest measurement
                    is above value for [a] or upper value for [b]
        sigma_theta_crit: density criteria in kg/m3 as scalar
        crit_method: how to select the MLD using the given criteria?
            'actual' to choose measurement depth preceding first measured depth where sigma_theta_crit is exceeded
                     (probably better to slightly underestimate MLD than overestimate it)
            'interp' to use linear interpolation with 'nearest' interpolation to estimate exact MLD (recommended)
        crit_interp_initial_spacing: spacing in meters used for interpolation during initial, coarse MLD search
        crit_interp_final_spacing: spacing in meters used for interpolation during final, fine MLD search
                                  (must be ≤ crit_interp_initial_spacing)
                                  (note: these spacing args are only used if 'interp' selected for 'crit_method')
        bottom_return: what to return if MLD not reached by bottom of profile
                      (note: warning will be printed if verbose_warn is True)
            'bottom' to return deepest measurement depth
            'NaN' to return NaN
        verbose_warn: print warnings
        verbose_error: print fatal errors (recommend keeping this True)

    Returns:
        MLD in meters if found
        NaN if MLD couldn't be found
        None if error encountered

    Common MLD criteria using sigma_theta:
        de Boyer Montégut et al. 2004 (for global ocean):
             0.03 kg/m3 from value at 10 m
            (authors note that 0.01 kg/m3 had been the 'often standard' criteria)
        Dong et al. 2008 (for Southern Ocean):
             0.03 kg/m3 (or temp criterion) from "near surface" value
            (authors say "0-20 m" or "20 m" but don't specify which to use, or whether to use average value)
        Wong and Riser 2011 (for under-ice Argo profiles off E. Antarctica):
             0.05 kg/m3 from the shallowest measurement

    """
    if 'sigma_theta' not in profile_data.keys():
        if verbose_error: print('Error from geo_tools.mld(): this profile does not include sigma_theta.')
        return None
    if 'data' not in profile_data['sigma_theta'].keys() or 'depth' not in profile_data['sigma_theta'].keys():
        if verbose_error: print('Error from geo_tools.mld(): this profile is missing data for sigma_theta.')
        return None
    if not all(diff(profile_data['sigma_theta']['depth']) > 0):
        if verbose_error: print('Error from geo_tools.mld(): depth vector is not monotonically increasing. It could be '
                                'backwards, jumbled, or incorrectly signed (should be positive).')
        return None

    if verbose_warn:
        if any(isnan(profile_data['sigma_theta']['data'])) or any(isnan(profile_data['sigma_theta']['depth'])):
            print('Warning from geo_tools.mld(): sigma-theta or depth vector contains NaNs.')

        if ref_depth == 'shallowest':
            if profile_data['sigma_theta']['depth'][0] >= 20: print('Warning from geo_tools.mld(): shallowest '
                                                                    'measurement is 20 m or deeper.')
        elif not isinstance(ref_depth,tuple):
            if profile_data['sigma_theta']['depth'][0] > ref_depth:
                if not ref_reject: print('Warning from geo_tools.mld(): '
                                         'reference depth is above shallowest measurement.')
                else:              return NaN
        elif not ref_reject:
            if profile_data['sigma_theta']['depth'][0] > ref_depth[1]:
                if not ref_reject: print('Warning from geo_tools.mld(): '
                                         'reference depth range is above shallowest measurement.')
                else:              return NaN

    if ref_depth == 'shallowest':
        rho_mld = sigma_theta_crit + profile_data['sigma_theta']['data'][0]
    elif not isinstance(ref_depth,tuple):
        rho_mld = sigma_theta_crit + vert_prof_eval(profile_data,'sigma_theta',ref_depth,z_coor='depth',
                                                    interp_method='linear',extrap='nearest',verbose_warn=True,
                                                    verbose_error=True)
    else:
        rho_mld = sigma_theta_crit + vert_prof_eval(profile_data,'sigma_theta',ref_depth,z_coor='depth',
                                                    interp_method='linear',extrap='nearest',avg_method=ref_range_method,
                                                    avg_spacing=0.1,verbose_warn=True,verbose_error=True)

    mld_found = depth_at_which(profile_data,'sigma_theta',rho_mld,z_coor='depth',method=crit_method,
                               top=0.0,bottom='bottom',interp_initial_spacing=crit_interp_initial_spacing,
                               interp_final_spacing=crit_interp_final_spacing,verbose_warn=True,verbose_error=True)

    if mld_found == None:
        if verbose_error: print('Error from geo_tools.mld(): unexpected error encountered at end of function.')
        return None
    elif isnan(mld_found) and bottom_return == 'bottom':
        return profile_data['sigma_theta']['depth'][-1]
    elif isnan(mld_found) and bottom_return == 'NaN':
        return NaN
    else:
        return mld_found


def destab(profile_data,to_depths,verbose_warn=True):
    """ Calculate convection resistance, i.e. buoyancy anomaly vertically integrated from the surface downwards, which
        represents the buoyancy loss required for convection to reach <<to_depth>>, as in de Lavergne et al. (2014),
        Fig. S3, Frajka-Williams et al. (2014), eq. 1, or **Bailey et al. (2005), p. 508**. Possibly dates to earlier
        work by Peter Rhines.

    Formula: (g/rho_0) * integral from 0 to <<to_depth>> of {sigma_theta(<<to_depth>>) - sigma_theta(z)} dz

    Args:
        profile_data: dict with 'sigma_theta' as key to a dict with 'data' and 'depth' as keys to 1-D arrays
                      (e.g. this_float_data['profiles'][72])
        to_depths: depth of convection in meters (scalar or 1-D array)
        verbose_warn: print warnings

    Returns:
        buoyancy_loss (in m^2 s^-2) or NaN if measurements not deep enough
                note: buoyancy flux has units m^2 s^-3, or total buoyancy per second

    """
    spacing = 0.1
    rho_0 = 1027.8
    if isscalar(to_depths): to_depth = array([to_depths])

    if max(to_depths) > profile_data['sigma_theta']['depth'][-1]:
        if verbose_warn: print('Warning from geo_tools.destab(): convection depth deeper than profile.')
        return NaN
    z_vec, rho_vec = vert_prof_even_spacing(profile_data,'sigma_theta',z_coor='depth',spacing=spacing,
                                            interp_method='linear',extrap='nearest',top=spacing,bottom=max(to_depths),
                                            verbose_error=True)
    if any(isnan(rho_vec)):
        if verbose_warn: print('Warning from geo_tools.destab(): NaNs in interpolated density profile. Check why.')
        return NaN

    buoyancy_loss = []
    for td_idx, to_depth in enumerate(to_depths):
        buoyancy_loss.append((constants.g / rho_0) * sum(rho_vec[-1] - rho_vec[z_vec <= to_depth]) * spacing)
    buoyancy_loss = array(buoyancy_loss)

    if buoyancy_loss.size == 1: return buoyancy_loss[0]
    else:                       return buoyancy_loss


def martinson(profile_data,metric='SD',to_depth=None,h_ice=None,tb_from_depth=None,sd_from_depth=None,sd_ref_psal=None,
              spacing=0.1,max_depth=500,sigma_i=30.0,verbose_warn=True,verbose_error=True):
    """ Martinson diagnostics for vertical profiles in Southern Ocean:
        custom 'thermal barrier' and 'salt deficit' calculations.

    Equations:
        see Wilson et al., 2019 (also, of course, Martinson 1990, Martinson and Iannuzzi 1998)

    Args:
        profile_data: dict with 'ptmp' as key to dict with 'data' and 'depth' as keys to 1-D arrays
                      (e.g. profile_data could be this_float_data['profiles'][72])
        metric: 'TB' (thermal barrier) or 'SD' (salt deficit)
        to_depth: None
                  or depth to which metric should be integrated
                    note: nearest value to interpolated depths is used, given <<spacing>>
        h_ice: None
               or equivalent ice thickness (m) (growth or melt) to which metric will be integrated;
                 the integration depth is returned
               note: if both <<to_depth>> and <<h_ice>> are None, will return vector of metric at each depth level
        tb_from_depth: None; or if metric is 'TB', start integrating from <<tb_from_depth>> instead of surface
                       example: use this to ignore warm summer mixed-layer
        sd_from_depth: None; or if metric is 'SD', start integrating from <<sd_from_depth>> instead of surface
                       example: use this to calculate freshwater anomaly of a subsurface layer
        sd_ref_psal: None; or if metric is 'SD' & h_ice is None, integrate from <<sd_ref_psal>>, not psal at <<to_depth>>
                     example: use this to track changes in 0-250 m freshwater anomaly over time, not absolute values
        spacing: spacing in meters for vertical interpolation
        max_depth: maximum depth to which these quantities will be computed
        sigma_i: negative freshwater flux (psu) from unit ice growth of salinity ~5 psu from seawater at ~35 psu
                 into a 100-m mixed layer, per Martinson 1990 and Martinson and Iannuzzi 1998

    Returns:
        option 1: single value
                  if h_ice is specified: depth (m) at which metric attains given value
                  if to_depth is specified:: given metric as equivalent ice thickness (m) (growth or melt)
        option 2: z_vec, prof
                  z_vec is vector of depth levels (m) with given <<spacing>>
                  prof is vertical profile of given metric (m ice equiv.), integrated down to each depth level

    """
    if 'ptmp' not in profile_data.keys() or 'psal' not in profile_data.keys():
        if verbose_error: print('Error from geo_tools.martinson(): profile does not include parameter of interest.')
        return None
    if to_depth is not None and to_depth >= max_depth:
        if verbose_error: print('Error from geo_tools.martinson(): parameter <<to_depth>> exceeds <<max_depth>>.')
        return None

    elif metric == 'SD':
        z_vec, psal_vec = vert_prof_even_spacing(profile_data,'psal',z_coor='depth',spacing=spacing,
                                                 interp_method='linear',extrap='nearest',top=0.0,bottom=max_depth,
                                                 verbose_error=True)
        if any(isnan(psal_vec)):
            if verbose_warn: print('Warning from geo_tools.martinson(): NaNs in interpolated profile. Check why.')
            return NaN

        if sd_from_depth is not None:
            from_depth_idx = abs(z_vec - sd_from_depth).argmin()
        else:
            from_depth_idx = 0

        if to_depth is not None and h_ice is None:
            to_depth_idx = abs(z_vec-to_depth).argmin()
            if to_depth_idx <= from_depth_idx and verbose_warn:
                print('Warning from geo_tools.martinson(): SD cannot be calculated because <<sd_from_depth>> is'
                      'deeper than <<to_depth>>.')
                return NaN
            if sd_ref_psal is not None:
                s_0 = sd_ref_psal
            else:
                s_0 = psal_vec[to_depth_idx]
            sd = trapz(s_0 - psal_vec[from_depth_idx:to_depth_idx + 1], dx=spacing) / sigma_i
            return sd
        elif to_depth is None and h_ice is not None:
            for td_idx, td in enumerate(z_vec):
                if td_idx <= from_depth_idx: continue
                s_0 = psal_vec[td_idx]
                sd = trapz(s_0 - psal_vec[from_depth_idx:td_idx + 1], dx=spacing) / sigma_i
                if sd >= h_ice: break
            if td_idx == len(z_vec)-1: return NaN
            else:                      return td
        elif to_depth is None and h_ice is None:
            sd = full(len(z_vec),NaN)
            for td_idx, td in enumerate(z_vec):
                if td_idx <= from_depth_idx: continue
                if sd_ref_psal is not None:
                    s_0 = sd_ref_psal
                else:
                    s_0 = psal_vec[td_idx]
                sd[td_idx] = trapz(s_0 - psal_vec[from_depth_idx:td_idx + 1], dx=spacing) / sigma_i
            return z_vec, sd

    if metric == 'TB':
        z_vec,ptmp_vec = vert_prof_even_spacing(profile_data,'ptmp',z_coor='depth',spacing=spacing,
                                                interp_method='linear',extrap='nearest',top=0.0,bottom=max_depth,
                                                verbose_error=True)
        if any(isnan(ptmp_vec)):
            if verbose_warn: print('Warning from geo_tools.martinson(): NaNs in interpolated profile. Check why.')
            return NaN

        t_f = gsw.t_freezing(gsw.SA_from_SP(34.4,0,profile_data['lon'],profile_data['lat']),0,1)
                      # freezing point of seawater at given lat/lon and 34.4 psu salinity
        rho_w = 1000  # density of water, kg m-3
        c_w = 4180    # specific heat capacity of seawater, J kg-1 K-1
        rho_i = 920   # sea-ice density, kg m-3
        L_i = 3.3e5   # latent heat of fusion of ice, J kg-1

        if tb_from_depth is not None:
            from_depth_idx = abs(z_vec - tb_from_depth).argmin()
        else:
            from_depth_idx = 0

        if to_depth is not None and h_ice is None:
            to_depth_idx = abs(z_vec-to_depth).argmin()
            if to_depth_idx <= from_depth_idx and verbose_warn:
                print('Warning from geo_tools.martinson(): TB cannot be calculated because <<tb_from_depth>> is'
                      'deeper than <<to_depth>>.')
                return NaN
            tb = trapz(ptmp_vec[from_depth_idx:to_depth_idx + 1] - t_f, dx=spacing) * rho_w * c_w / (rho_i * L_i)
            return tb
        elif to_depth is None and h_ice is not None:
            for td_idx, td in enumerate(z_vec):
                if td_idx <= from_depth_idx: continue
                tb = trapz(ptmp_vec[from_depth_idx:td_idx + 1] - t_f, dx=spacing) * rho_w * c_w / (rho_i * L_i)
                if tb >= h_ice: break
            if td_idx == len(z_vec)-1: return NaN
            else:                      return td
        elif to_depth is None and h_ice is None:
            tb = full(len(z_vec),NaN)
            for td_idx, td in enumerate(z_vec):
                if td_idx <= from_depth_idx: continue
                tb[td_idx] = trapz(ptmp_vec[from_depth_idx:td_idx + 1] - t_f, dx=spacing) * rho_w * c_w / (rho_i * L_i)
            return z_vec, tb


def great_circle_interp(start_position,end_position,scale_vector):
    """ Interpolates along a great circle track between two defined points.

    Adapted from my previous MATLAB script 'geotrack_interp'.

    Dimensions:
          J = 2 + N   (i.e. N plus the start/end points)
          N = number of values to interpolate to

    Inputs:
        'start_position' and 'end_position' are [lat,lon]
        'scale_vector' (length J) can be datetime instances, float timestamps, float unitless distances, etc.

    Output:
        'interp_positions'    array of size N x 2   [[lat1,lon1];[lat2,lon2];...]

    """

    if isinstance(scale_vector[0],datetime):
        scale_vector = [tt.convert_datetime_to_total_seconds(t) for t in scale_vector]

    interp_J = len(scale_vector)
    interp_N = interp_J - 2
    scale_vector = array(scale_vector)

    assert interp_N >= 1, 'Error: check size of input scaling array.'
    assert len(start_position) == 2 and len(end_position) == 2, 'Error: check size of start and end points.'
    assert scale_vector[interp_J-1] > scale_vector[0], 'Error: check that scale vector increases from start to end.'

    geod_inv = geod.Inverse(*start_position, *end_position)
    interp_arc_length = geod_inv['s12']
    interp_azimuth = geod_inv['azi1']

    interp_scale_factor = interp_arc_length / (scale_vector[interp_J-1] - scale_vector[0])
    interp_arc_distances = interp_scale_factor * (scale_vector[1:interp_J-1] - scale_vector[0])

    geod_path = geod.DirectLine(*start_position,interp_azimuth,interp_arc_length)
    interp_positions = []
    for pt in range(len(interp_arc_distances)):
        interp_positions.append([geod_path.Position(interp_arc_distances[pt])['lat2'],
                                 geod_path.Position(interp_arc_distances[pt])['lon2']])
    return interp_positions


def nan_fraction_domain(lons,lats,field_of_interest,circumant_lons,circumant_lats):
    """ Calculates fraction of grid cells north of the Antarctic coastline in 'field_of_interest' that are NaN.
    
    Notes: 1D vector 'circumant_lons' must be from -180 to 179 (inclusive) with a spacing of 1.0.
           2D grid 'lons' can be -180 to 180 or 0 to 360.
    
    Test: should return 0.1494
        sic_grid = sea_ice_grids['nimbus5']
        sic_field = ldp.load_nimbus5(sea_ice_data_avail['nimbus5'][(1972,12,13)][0])
        gt.nan_fraction_domain(sic_grid['lons'],sic_grid['lats'],sic_field,circumant_lons,circumant_lats)
    
    Baselines (NaN fraction > 0.045 serves as reasonable rejection criterion for all datasets below):
        Nimbus-5 25 km: perfect data returns 0.040659
        GSFC (DMSP) 25 km: perfect data returns 0.0424958
        AMSR-E/AMSR-2 regridded to 25 km: perfect data returns around 0.033-0.035 (variable)
    
    """
    nans_in_domain = isnan(field_of_interest[lats > circumant_lats[(convert_360_lon_to_180(floor(lons)) + 180).astype(int)]])
    return sum(nans_in_domain) / len(nans_in_domain)


def identify_polynyas_magic(sat_abbrev,date,sea_ice_grids,sea_ice_data_avail,circumant_lons,circumant_lats,
                            open_threshold=80,extent_threshold=1000,cutoff_lat=-56.0,nan_fraction_domain_thresh=0.045,
                            use_goddard=True,regrid_amsr_to_25km=True,identify_bad=False):
    """ 'Magic' wrapper for identify_polynyas(). Given date, satellite, thresholds, and all sea ice fields/grids,
        identifies polynyas and outputs relevant statistics.

    Args:
        sat_abbrev: 'nimbus5', 'dmsp' or 'gsfc', 'amsre', or 'amsr2'
        date: tuple format, e.g. (2002,9,26)
        sea_ice_grids, sea_ice_data_avail: created by load_product.sea_ice_data_prep()
        circumant_lons, circumant_lats: created from coastline shapefiles by establish_antarctic_sectors in main script
        open_threshold: 1-100, sea ice concentration below which a polynya may be identified
        extent_threshold: in km^2, minimum polynya extent to identify
        cutoff_lat: latitude north of which to ignore polynyas, e.g. -56.0 (for 56°S)
        use_goddard: see load_product.load_dmsp()
        regrid_amsr_to_25km: regrid AMSR fields to 25 km? ... see load_product.load_amsr()
        nan_fraction_domain_thresh: [0.045] threshold for good quality SIC data; determined empirically;
                                    see geo_tools.nan_fraction_domain() for details
        identify_bad: return polynya ID results regardless of whether SIC is good throughout domain
                      (i.e. bypass nan_fraction_domain_thresh)

    Returns:
        sat_string: nicely formatted string noting satellite instrument and resolution
        polynya_string: nicely formatted string noting polynya SIC and extent criteria
        filename_abbrev: same as sat_abbrev, except 'gsfc' if sat_abbrev is 'dmsp'
        sic_grid: from argument sea_ice_grids, for specified satellite
        sic_field: from load_product.<satellite load method>()
        polynya_stats, polynya_grid, open_ocean_grid: as returned by identify_polynyas(), modified as follows:
            - polynyas not matching argument criteria are labeled '0' in polynya_grid and deleted from polynya_stats
        polynya_grid_binary: like polynya_grid, except all polynyas labeled as '1' (cells otherwise labeled '0')
        error_code: 0 for good SIC field and polynya ID results
                        (returns all items as computed)
                    1 for fully bad SIC field and no polynya ID results
                        (returns None for SIC and polynya items, except polynya_string)
                    2 for partially bad SIC field and no polynya ID results
                        (returns SIC items, but returns None for polynya items, except polynya_string)
    """

    if not sea_ice_data_avail[sat_abbrev][date][1]:
        polynya_string = 'Note: nonexistent SIC field'
        error_code = 1
        return None,polynya_string,None,None,None,None,None,None,None,error_code
    elif sat_abbrev is 'nimbus5':
        sic_grid = sea_ice_grids['nimbus5']
        sic_field = ldp.load_nimbus5(sea_ice_data_avail[sat_abbrev][date][0])
        sat_string = 'NSIDC Nimbus-5 (25 km)'
    elif sat_abbrev is 'dmsp' or sat_abbrev is 'gsfc':
        sic_grid = sea_ice_grids['dmsp']
        sic_field = ldp.load_dmsp(sea_ice_data_avail[sat_abbrev][date][0],date,use_goddard=use_goddard)
        gsfc_sat_abbrevs = ['Nimbus-7 SMMR','DMSP-F8 SSM/I','DMSP-F11 SSM/I','DMSP-F13 SSM/I',
                            'DMSP-F17 SSMIS','DMSP-F18 SSMIS']
        gsfc_sat_start_dates = [(1978,11,1),(1987,7,9),(1991,12,3),(1995,10,1),(2008,1,1),(2017,3,1)]
        gsfc_sat_end_dates = [(1987,7,8),(1991,12,2),(1995,9,30),(2007,12,31),(2017,2,28),tt.now()]
        for s in range(len(gsfc_sat_abbrevs)):
            if tt.is_time_in_range(gsfc_sat_start_dates[s],gsfc_sat_end_dates[s],date):
                if s >= 5:
                    sat_string = 'NOAA/NSIDC NRT CDR ' + gsfc_sat_abbrevs[s] + ' (25 km)'
                    filename_abbrev = 'nrt_cdr'
                elif use_goddard or (not use_goddard and s == 0):
                    sat_string = 'Goddard Merged ' + gsfc_sat_abbrevs[s] + ' (25 km)'
                    filename_abbrev = 'gsfc'
                else:
                    sat_string = 'NOAA/NSIDC CDR ' + gsfc_sat_abbrevs[s] + ' (25 km)'
                    filename_abbrev = 'cdr'
    elif sat_abbrev is 'amsre':
        if regrid_amsr_to_25km:
            sic_grid = sea_ice_grids['amsre_25km']
        else:
            sic_grid = sea_ice_grids['amsre']
        sic_field = ldp.load_amsr(sea_ice_data_avail[sat_abbrev][date][0],regrid_to_25km=regrid_amsr_to_25km)
        if regrid_amsr_to_25km:
            sat_string = 'ASI AMSR-E (25 km, regridded from original 6.25 km)'
        else:
            sat_string = 'ASI AMSR-E (6.25 km)'
    elif sat_abbrev is 'amsr2':
        if regrid_amsr_to_25km:
            sic_grid = sea_ice_grids['amsr2_25km']
        else:
            sic_grid = sea_ice_grids['amsr2']
        sic_field = ldp.load_amsr(sea_ice_data_avail[sat_abbrev][date][0],regrid_to_25km=regrid_amsr_to_25km)
        if regrid_amsr_to_25km:
            sat_string = 'ASI AMSR2 (25 km, regridded from original 6.25 km)'
        else:
            sat_string = 'ASI AMSR2 (6.25 km)'
    else:
        raise Exception('Satellite abbreviation not recognized.')

    if sat_abbrev != 'dmsp' and sat_abbrev != 'gsfc':
        filename_abbrev = sat_abbrev

    # bad SIC grid... return here
    if sum(isnan(sic_field)) == size(sic_field):
        polynya_string = 'Note: nonexistent SIC field'
        error_code = 1
        return None, polynya_string, None, None, None, None, None, None, None, error_code

    # good SIC grid... continue with identification
    elif identify_bad \
            or nan_fraction_domain(sic_grid['lons'],sic_grid['lats'],sic_field,circumant_lons,circumant_lats) \
                    <= nan_fraction_domain_thresh:
        # FIXME: 0.045 nan_fraction_domain threshold doesn't apply for 6.25 km AMSR data... recompute or ditch it
        polynya_stats,polynya_grid,open_ocean_grid = identify_polynyas(sic_grid['lons'],sic_grid['lats'],
                                                                       sic_grid['areas'],sic_field,
                                                                       open_threshold=open_threshold)
        for polynya_index in reversed(range(len(polynya_stats))):
            if (polynya_stats[polynya_index]['total_extent'] < extent_threshold) \
                    or (polynya_stats[polynya_index]['centroid'][0] > cutoff_lat):
                polynya_grid[polynya_grid == polynya_stats[polynya_index]['polynya_ID']] = 0
                del polynya_stats[polynya_index]
        polynya_grid_binary = polynya_grid.copy()
        polynya_grid_binary[polynya_grid_binary != 0] = 1
        polynya_string = 'Polynya criteria: SIC <' + str(open_threshold) + '%, extent >' \
                         + str(extent_threshold) + ' km$^2$'
        error_code = 0
        return sat_string, polynya_string, filename_abbrev, sic_grid, sic_field, \
               polynya_stats, polynya_grid, polynya_grid_binary, open_ocean_grid, error_code

    # partially bad SIC grid (too many NaNs)... return SIC but no polynya IDs
    else:
        polynya_string = 'Note: too many bad grid cells for polynya identification'
        error_code = 2
        return sat_string, polynya_string, filename_abbrev, sic_grid, sic_field, None, None, None, None, error_code


def identify_polynyas(lons,lats,areas,sic,open_threshold=80):
    """ Identifies polynyas from gridded sea ice concentration field.

    Uses SciPy's scipy.ndimage.label() function. Very fast. This is a basic 'binary connected-component
    labeling' algorithm, a staple of mathematical morphology and image processing. Here, it groups
    contiguous grid cells that I've denoted as 'polynya candidates' using the 'open_threshold' criterion.
    A 'structuring element' defines allowed connections; here, we choose to include diagonals between grid
    cells, and thus choose a 3x3 matrix of ones as our structuring element. One could exclude diagonals
    from consideration by choosing [[0,1,0],[1,1,1],[0,1,0]] instead. Polynya statistics are subsequently
    calculated for each connected region. This implementation uses the following auxiliary (helper)
    function:
        - sea_ice_grid_polynya_statistics()
    
    Args:
        lons: 2D array of grid lons
        lats: 2D array of grid lats
        areas: 2D array of grid areas
        sic: 2D array of gridded sea ice concentration (0-100), in which contiguous grid cells are more or less
             contiguous in lat/lon space
        open_threshold: 1-100, sea ice concentration below which a polynya may be identified

    Returns:
        polynya_statistics: list with length = number of polynyas found; each list entry is a dict, with these keys:
            - 'cell_indices'      = unordered list of (i,j) tuples, one for each grid cell within the polynya
            - 'cell_latlons'      = unordered list of (lat,lon) tuples, "
            - 'cell_areas'        = unordered list of pixel areas [km^2]
            - 'cell_sics'         = unordered list of pixel sea ice concentrations (0% to open_threshold)
            - 'total_extent_under_threshold'    = open water extent of polynya [km^2]
            - 'total_open_area_under_threshold' = open water area of polynya (i.e. SUM[0.01*pixelsic*pixelarea]) [km^2]
                    NOTE: the above two metrics exclude isolated blocks of icy grid cells within the polynya (i.e. with
                          SIC >= threshold; thus 'open_extent' and 'open_area' could be smaller than the true extent
                          and open area
            - 'total_extent'      = open water extent of polynya, with icy interior patches filled in [km^2]
            - 'total_open_area'   = open water area (as computed above), with icy interior patches filled in [km^2]
            - 'total_grid_cells'  = total number of grid cells (i.e. pixels) contained within the polynya
            - 'centroid'          = (lat,lon) tuple of centroid (i.e. center of mass) of polynya
            - 'polynya_ID'        = NaN for now; meant to be used elsewhere for tracking polynyas through time
        polynya_grid: returns 2D array with shape = shape(sic); different numbers correspond to polynya labels
        open_ocean_grid: boolean grid with shape = shape(sic)
            True represents contiguous open water found
            False represents everything else
        
    """
    with errstate(invalid='ignore'):
        polynya_candidates = sic < open_threshold

    labeled_feature_grid, num_features = ndimage.label(polynya_candidates,
                                                       structure=array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))
    pixels_per_feature = [sum(labeled_feature_grid == n+1) for n in range(num_features)]
    open_ocean_feature_label = 1 + argmax(pixels_per_feature)
    open_ocean_grid = (labeled_feature_grid == open_ocean_feature_label)
    polynya_grid = copy(labeled_feature_grid)
    polynya_grid[polynya_grid == open_ocean_feature_label] = 0
    polynya_statistics = sea_ice_grid_polynya_statistics(polynya_grid,num_features,open_ocean_feature_label,
                                                         lons,lats,areas,sic)
    return polynya_statistics, polynya_grid, open_ocean_grid


def establish_coastline(coastline_filename_prefix):
    """ Exports lats/lons of Antarctic coastline.

    Data provenance: GSHHG coastline shapefiles downloaded from: https://www.soest.hawaii.edu/pwessel/gshhg/
    """

    sf = shapefile.Reader(coastline_filename_prefix)
    shapes = sf.shapes()
    antarctic_coast_coors = shapes[1].points[3:len(shapes[1].points)] + shapes[0].points[4:len(shapes[0].points)]
    antarctic_coast_coors = array(antarctic_coast_coors)
    circumant_lons = arange(-180,180)
    circumant_lats = interp(arange(-180,180),antarctic_coast_coors[:,0],antarctic_coast_coors[:,1])
    return circumant_lons, circumant_lats


############# AUXILIARY (INTERNAL) FUNCTIONS ################


def profile_interp(data,z_orig,z_interp,method='linear',out_of_bounds='NaN'):
    """ Wrapper method. Use 1-D interpolation method of choice to evaluate 1-D 'data' at value/values 'z_interp'.

    NOTE: Cubic interpolation doesn't actually work well on vertical profiles, especially near sharp corners. For most
          profiles with reasonably high resolution (< ~5 m spacing?), linear interp is probably a better choice.

    Args:
        method: 'linear'  for linear interpolation
                'nearest' for nearest-neighbor interpolation
                               (in which case 'nearest' and 'extrap' as args for 'out_of_bounds' are identical)
                'cubic'   for spline interpolation of the 3rd order
        out_of_bounds: 'NaN'     to return NaN for values above or below range of 'z_orig'
                       'nearest' to extrapolate using the uppermost/lowermost value of 'data'
                       'extrap'  to extrapolate using cubic interp
    """
    assert(all(diff(z_orig) > 0))

    if len(z_orig) <= 2 or len(data) <= 2: return NaN

    if out_of_bounds == 'NaN':
        interpolant = interpolate.interp1d(z_orig,data,kind=method,bounds_error=False,fill_value=NaN)
    elif out_of_bounds == 'nearest':
        interpolant = interpolate.interp1d(z_orig,data,kind=method,bounds_error=False,fill_value=(data[0],data[-1]))
    elif out_of_bounds == 'extrap':
        interpolant = interpolate.interp1d(z_orig,data,kind=method,bounds_error=False,fill_value='extrapolate')
    else:
        raise ValueError('Extrapolation method must be NaN, nearest, or cubic.')
    result = interpolant(z_interp)

    if result.size == 1: return asscalar(result)
    else:                return result


def sea_ice_grid_polynya_statistics(polynya_label_grid,num_features,open_ocean_label,lons,lats,areas,sic):
    """ Helper function for identify_polynyas(). Calculates statistics on polynyas found using SciPy ndimage.label().

        Returns:
            this_polynya_statistics: dictionary with key/value pairs as described above in identify_polynyas()

    """
    polynya_statistics = []
    polynya_counter = 0
    for p in range(num_features):
        if p+1 == open_ocean_label:
            continue
        else:
            polynya_counter += 1
            this_polynya_grid = (polynya_label_grid == p+1)
            this_polynya_statistics = {}
            this_polynya_cell_indices = [tuple(ind) for ind in list(array(where(polynya_label_grid == p+1)).T)]
            this_polynya_statistics['cell_indices'] = this_polynya_cell_indices
            this_polynya_statistics['cell_latlons'] = [(lats[this_polynya_cell_indices[c][0],
                                                            this_polynya_cell_indices[c][1]],
                                                        lons[this_polynya_cell_indices[c][0],
                                                            this_polynya_cell_indices[c][1]])
                                                       for c in range(len(this_polynya_cell_indices))]
            this_polynya_statistics['cell_areas'] = [areas[this_polynya_cell_indices[c][0],
                                                           this_polynya_cell_indices[c][1]]
                                                     for c in range(len(this_polynya_cell_indices))]
            this_polynya_statistics['cell_sics'] = [sic[this_polynya_cell_indices[c][0],
                                                        this_polynya_cell_indices[c][1]]
                                                    for c in range(len(this_polynya_cell_indices))]
            if mean(this_polynya_statistics['cell_sics']) == 0.0 and len(this_polynya_cell_indices) <= 20:
                # consider this a spurious polynya (small patches of exactly 0% SIC, e.g. on 8/31/1986, 9/28/1986)
                polynya_label_grid[polynya_label_grid == p+1] = 0
                continue
            this_polynya_statistics['total_extent_under_threshold'] = sum(this_polynya_statistics['cell_areas'])
            this_polynya_statistics['total_open_area_under_threshold'] = sum(0.01 * (100.0 -
                                                                     array(this_polynya_statistics['cell_sics'])) * \
                                                                     array(this_polynya_statistics['cell_areas']))
            this_polynya_grid_holes_filled = ndimage.binary_fill_holes(this_polynya_grid)
            filled_indices = where(this_polynya_grid_holes_filled)  # shape = (2,n_cells)
            this_polynya_statistics['total_extent'] = sum([areas[filled_indices[0][c],filled_indices[1][c]] for c in
                                                           range(sum(this_polynya_grid_holes_filled))])
            this_polynya_statistics['total_open_area'] = sum([0.01 * (100.0 - sic[filled_indices[0][c],
                                                                                  filled_indices[1][c]])
                                                              * areas[filled_indices[0][c], filled_indices[1][c]]
                                                              for c in range(sum(this_polynya_grid_holes_filled))])
            this_polynya_statistics['total_grid_cells'] = len(this_polynya_cell_indices)
            if sum(array(this_polynya_statistics['cell_latlons'])[:,1] > 150) > 0 \
                    and sum(array(this_polynya_statistics['cell_latlons'])[:,1] < -150) > 0:
                mean_lat = mean(array(this_polynya_statistics['cell_latlons'])[:,0])
                corrected_lons = convert_180_lon_to_360(array(this_polynya_statistics['cell_latlons'])[:,1])
                mean_lon = convert_360_lon_to_180(mean(corrected_lons))
                this_polynya_statistics['centroid'] = array([mean_lat,mean_lon])
            else:
                this_polynya_statistics['centroid'] = array(this_polynya_statistics['cell_latlons']).mean(0)
            this_polynya_statistics['polynya_ID'] = p+1
            polynya_statistics.append(this_polynya_statistics)
    return polynya_statistics