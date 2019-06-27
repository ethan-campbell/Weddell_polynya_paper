# -*- coding: utf-8 -*-

import numpy as np
from scipy import stats
from scipy import interpolate as spin
import pandas as pd
import os
import warnings
from datetime import datetime, timedelta
import calendar
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as pltick
import matplotlib.dates as mdates
import matplotlib.cbook as mcbook
import matplotlib.colors as mcolors
import matplotlib.legend as mlegend
from matplotlib import gridspec
from matplotlib.patches import Polygon
os.environ['PROJ_LIB'] = '/Applications/anaconda/share/proj'      # temporarily necessary for Basemap import
from mpl_toolkits.basemap import Basemap
from collections import OrderedDict
warnings.filterwarnings('ignore','.*is_string_like function.*')   # MatplotlibDeprecationWarning upon cmocean import
import cmocean
import gsw
from Circles.circles import circle   # from https://github.com/urschrei/Circles

import time_tools as tt
import geo_tools as gt
import load_product as ldp


def sea_ice_argo_spatial(data_dir,date,sic_grid,sic,float_data,plot_argo_locs_not_trajs,
                         title,save_as,results_dir,width,height,lat_center,lon_center,
                         open_sic=0,max_sic=100,which_ice_cmap=4,extend_cmap='neither',rasterized=True,
                         plot_floats=True,polynya_grid=None,label_traj_dates=True,
                         as_subplot=False,create_subplot=True,subplot_fig_size=(11,6),
                         first_subplot=False,last_subplot=False,which_subplot=[1,1,1],
                         subplot_add_colorbar=False,bathy_contours=np.arange(-3500,-100,500),
                         grid_lats=np.arange(-80,60,5),grid_lons=np.arange(-80,50,10),
                         subplot_lon_labels=[0,0,0,0],subplot_lat_labels=[0,0,0,0],subplot_labelsize=None,
                         cmap_bad_color='w',cmap_ocean_color='#5bcfff',grid_color='.2',continent_color='0.7',
                         boundary_width=2,coastline_width=1,pad=0.25,spacing=0.2,cbar_bottom=0.125,
                         return_basemap=False,return_pcolor=False,include_year_in_date=False,save_png=False):
    """ Plots Argo profile locations on regional map with background of sea ice concentration and bathymetry.
    """

    warnings.filterwarnings('ignore', category=mcbook.mplDeprecation)

    if plot_argo_locs_not_trajs is True:
        floats_linger = 14   # let floats linger on the map for N days after most recent profile (if no new profile)
        if as_subplot: cross_lonx = 100000/2; cross_laty = 100000/2; cross_width = 1.25; text_offset = 0.03
        else:          cross_lonx = 100000/4; cross_laty = 100000/4; cross_width = 2.5;  text_offset = 0.015

    if which_ice_cmap == 1:   ice_cmap = plt.cm.CMRmap
    elif which_ice_cmap == 2: ice_cmap = plt.cm.inferno
    elif which_ice_cmap == 3: ice_cmap = plt.cm.gist_ncar
    elif which_ice_cmap == 4: # custom colormap similar to cm.CMRmap:
        cmap_colors = ['#79CDFA','#79CDFA','#87C3EC','#628eac','#1E1952',
                       '#4630B8','#E85A33','#E1C047','#F2F1C4','#FBFBEE','#FFFFFF']
        ice_cmap = mcolors.LinearSegmentedColormap.from_list(name=None,colors=cmap_colors,N=250,gamma=1.3)
    elif which_ice_cmap == 5: # alternate version of cmap 4 above, with less vibrant ocean blue
        cmap_colors = ['#bce6fc','#bce6fc','#87C3EC','#628eac','#1E1952',
                       '#4630B8','#E85A33','#E1C047','#F2F1C4','#FBFBEE','#FFFFFF']
        ice_cmap = mcolors.LinearSegmentedColormap.from_list(name=None,colors=cmap_colors,N=250,gamma=1.3)

    if not as_subplot:
        fig, m = blank_inset_basemap(width,height,lat_center,lon_center,lon_labels=[0,0,0,1],lat_labels=[1,0,0,0],
                                     grid_lats=grid_lats,grid_lons=grid_lons,labelsize=subplot_labelsize)
    if as_subplot:
        if create_subplot:
            if first_subplot: master_fig = plt.figure(figsize=subplot_fig_size)
            plt.gcf().add_subplot(*which_subplot)
        master_fig, m = blank_inset_basemap(width,height,lat_center,lon_center,create_new_fig=False,
                                            lon_labels=subplot_lon_labels,lat_labels=subplot_lat_labels,
                                            grid_lats=grid_lats,grid_lons=grid_lons,
                                            labelsize=subplot_labelsize,grid_color=grid_color,
                                            fill_continent_color=continent_color,
                                            boundary_width=boundary_width,coastline_width=coastline_width)
    xlims = plt.gca().get_xlim()
    ylims = plt.gca().get_ylim()
    lonx, laty = m(sic_grid['lons'], sic_grid['lats'])
    sic_nan_masked = np.ma.masked_where(np.isnan(sic), sic)
    sic_lon_edge_to_center = 0.5*np.mean([np.mean(np.diff(lonx[0,:])), np.mean(np.diff(lonx[-1,:]))])
    sic_lat_edge_to_center = 0.5*np.mean([np.mean(np.diff(laty[0,:])),np.mean(np.diff(laty[-1,:]))])
    pcm = plt.pcolormesh(lonx-sic_lon_edge_to_center, laty-sic_lat_edge_to_center, sic_nan_masked,
                         cmap=ice_cmap, edgecolors='None', rasterized=rasterized, zorder=1, alpha=1.0,
                         vmin=open_sic, vmax=max_sic) # norm=mcolors.PowerNorm(gamma=0.4,vmin=open_sic,vmax=100,clip=False)
    pcm.cmap.set_over('w')
    pcm.cmap.set_bad(cmap_bad_color)
    pcm.cmap.set_under(cmap_ocean_color) # '#5bcfff' is sea blue (previously #f0ffff, very light blue)
    if not as_subplot:
        cbar = plt.colorbar(pad=0.05, shrink=0.65, format='%.0f%%', extend=extend_cmap)
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label('Sea ice concentration',size=12)
    if polynya_grid is not None:
        plt.contour(lonx,laty,polynya_grid,levels=[0.999],colors='#00FF00',linewidths=0.7,alpha=0.8, zorder=2)
    if len(bathy_contours) > 0:
        etopo_lons, etopo_lats, etopo = ldp.load_bathy(data_dir)
        retopolons, retopolats = m(*np.meshgrid(etopo_lons, etopo_lats))
        olevels = bathy_contours  # check etopo.ravel().min()
        m.contour(retopolons, retopolats, etopo, olevels, linewidths=0.5, linestyles='solid', colors='#808080',
                  alpha=0.5, zorder=3)

    if plot_floats:
        for f in range(len(float_data)):
            wmoid = float_data[f][0]
            float_lons = float_data[f][1]
            float_lats = float_data[f][2]
            position_flags = float_data[f][3]
            float_datetimes = float_data[f][4]
            float_dates = (float_data[f][4]/1000000).astype(int)
            prof_nums = float_data[f][5]
            date_int = tt.convert_tuple_to_8_int(date)

            if plot_argo_locs_not_trajs:
                if sum((float_dates - date_int) == 0) >= 1:
                    this_day_index = np.where((float_dates - date_int) == 0)[0][0]
                    lonx, laty = m(float_lons[this_day_index], float_lats[this_day_index])
                    if not (xlims[0] <= lonx <= xlims[1]) or not (ylims[0] <= laty <= ylims[1]): continue
                    if position_flags[this_day_index] == 1:
                        c='m'
                        edgecolor='k'
                    else:
                        c='#15178F'
                        edgecolor='k'
                    plt.plot([lonx-cross_lonx,lonx+cross_lonx],[laty,laty],color=c,linestyle='solid',linewidth=cross_width,zorder=4)
                    plt.plot([lonx,lonx],[laty-cross_laty,laty+cross_laty],color=c,linestyle='solid',linewidth=cross_width,zorder=4)
                    plt.scatter(lonx,laty,s=14,c=c,edgecolors=edgecolor,alpha=0.9,zorder=5)
                    if subplot_labelsize is None: float_fontsize = 8
                    else:                         float_fontsize = subplot_labelsize
                    plt.text(lonx + text_offset*width,laty - 3*text_offset*height,str(wmoid) + ' ('
                             + str(prof_nums[this_day_index]) + ')',fontsize=float_fontsize,color=c,clip_on=False,zorder=6)
                elif date_int > float_dates[0]:
                    recent_day_index = np.where((float_dates - date_int) < 0)[0][-1]
                    days_since_last_profile = tt.days_between(tt.convert_8_int_to_tuple(float_dates[recent_day_index]),date)
                    if days_since_last_profile <= floats_linger:
                        lonx, laty = m(float_lons[recent_day_index], float_lats[recent_day_index])
                        if not (xlims[0] <= lonx <= xlims[1]) or not (ylims[0] <= laty <= ylims[1]): continue
                        # alpha = (0.75-0.25) + 0.25*(1 - days_since_last_profile/floats_linger)
                        if position_flags[recent_day_index] == 1: c = 'm'
                        else: c = '#15178F'
                        plt.plot([lonx-cross_lonx,lonx+cross_lonx],[laty,laty],color=c,linestyle='solid',linewidth=cross_width,zorder=4)
                        plt.plot([lonx,lonx],[laty-cross_laty,laty+cross_laty],color=c,linestyle='solid',linewidth=cross_width,zorder=4)
                        # plt.scatter(lonx,laty,s=18,c=c,edgecolors='none',alpha=0.7,zorder=5)
            elif not plot_argo_locs_not_trajs:
                flonx,flaty = m(float_lons,float_lats)
                plt.plot(flonx[position_flags != 9],flaty[position_flags != 9],color='#15178F',linewidth=1.25,zorder=4)
                plt.scatter(flonx[position_flags == 2],flaty[position_flags == 2],s=10,c='m',edgecolors='none',zorder=5)
                plt.scatter(flonx[position_flags == 1],flaty[position_flags == 1],s=10,c='#15178F',edgecolors='none',
                            zorder=6)

                if len(float_data) == 1 or label_traj_dates == True:
                    datetime_tuples = [tt.convert_14_to_tuple(float_datetimes[n]) for n in range(len(float_datetimes))]
                    mo_yr_strings = [str(datetime_tuples[n][1]) + '/' + '{0:02d}'.format(datetime_tuples[n][0] - 2000)
                                     for n in range(len(datetime_tuples))]
                    unique_mo_yr_strings,unique_indices = np.unique(mo_yr_strings,return_index=True)
                    unique_indices = np.sort(unique_indices) # to undo undesired sort by 'unique'
                    mo_yr_strings_to_label = [mo_yr_strings[n] for n in unique_indices]
                    lonx_to_label = [flonx[n] for n in unique_indices]
                    laty_to_label = [flaty[n] for n in unique_indices]
                    for pt in np.arange(0,len(mo_yr_strings_to_label),6):
                        plt.text(lonx_to_label[pt] + 0.000625 * width,laty_to_label[pt] - 0.026 * height,
                                 mo_yr_strings_to_label[pt],fontsize=7,color='#15178F')
                    for pt in np.arange(3,len(mo_yr_strings_to_label),6):
                        plt.text(lonx_to_label[pt] + 0.000625 * width,laty_to_label[pt] + 0.017 * height,
                                 mo_yr_strings_to_label[pt],fontsize=7,color='#15178F')
                if len(float_data) != 1:
                    plt.annotate(str(wmoid),fontsize=10,color='#15178F',xy=(flonx[len(flonx) - 1], flaty[len(flaty) - 1]),
                                 xytext=(flonx[len(flonx) - 1] - 0.25 * width,flaty[len(flaty) - 1] + 0.2 * height),
                                 arrowprops=dict(arrowstyle='->',color='#15178F',alpha=0.5))
    if not as_subplot:
        if title is not None: plt.title(title,fontsize=16)
        plt.tight_layout()
        if save_png: plt.savefig(results_dir + save_as + '.png',dpi=150)
        else:        plt.savefig(results_dir + save_as + '.pdf')
        plt.close()
    elif as_subplot:
        if create_subplot:
            if subplot_labelsize is None: baseline_date_fontsize = 7
            else:                         baseline_date_fontsize = subplot_labelsize
            if include_year_in_date:
                day_string = '{0}-{1:02d}-{2:02d}'.format(*date)
                date_fontsize = baseline_date_fontsize + 1
            else:
                day_string = '{1}-{2:02}'.format(*date)
                date_fontsize = baseline_date_fontsize + 3
            plt.text(0.05,0.95,day_string,fontsize=date_fontsize,fontweight='bold',
                     horizontalalignment='left',verticalalignment='top',transform=plt.gca().transAxes)
        if not create_subplot and subplot_add_colorbar:  # deprecated?
            cbar = plt.gcf().colorbar(pcm,ticks=np.arange(open_sic,101,10),format='%.0f%%',extend=extend_cmap,
                                      orientation='vertical',fraction=0.03,aspect=30)
            cbar.ax.tick_params(labelsize=subplot_labelsize,left=True,right=False,labelleft=True,labelright=False)
            cbar.outline.set_linewidth(boundary_width)
        if last_subplot:
            plt.tight_layout(h_pad=pad,w_pad=pad,rect=(0.02,0.02,0.98,0.98))
            if subplot_add_colorbar:
                if spacing is not None: hspace = spacing*width/height
                else:                   hspace = None
                plt.gcf().subplots_adjust(bottom=0.05,wspace=spacing,hspace=hspace)
                cbar_ax = plt.gcf().add_axes([0.2,cbar_bottom,0.6,0.015])
                cbar = plt.gcf().colorbar(pcm,format='%.0f%%',extend=extend_cmap,orientation='horizontal',cax=cbar_ax)
                cbar.ax.tick_params(labelsize=subplot_labelsize+2)
                cbar.outline.set_linewidth(boundary_width)
            plt.savefig(results_dir + save_as + '.pdf')
            plt.close()
    if   return_basemap and not return_pcolor: return m
    elif return_basemap and return_pcolor:     return m, pcm


def section(wmoid,results_dir,save_as,float_data,params='all',depth_lim=(0,1700),fixed_ylim=True,vert_res=10,toi=None,
            mld=True,mld_ref_depth=10,mld_sigma_theta_crit=0.03,show_ice_bars=True,sea_ice_grids=None,
            sea_ice_data_avail=None,show_prof_bars=False,show_prof_ticks=True,add_date_bars=None,cmap_level_mods=None,
            cmap_color_mods=None,cmap_gamma_mods=None,cmap_freq_mods=None,trim_xlim=False,
            create_new_figs=True,new_figsize=(8,6),add_title=True,facecolor='k',grid=True,
            years_only=False,plot_ylabel=True,plot_xticklabels=True,plot_cbar=True,condensed_cbar_label=None,
            smaller_text=False,force_label_size=None,density_coor=False,density_lim=(27.75,27.85),density_power_scale=15,
            density_depth_contours=None,density_depth_labels=True,explicit_yticks=None,
            drift_temps=None,drift_temp_baseline=None,drift_temp_depth=None):
    """ Hydrographic depth section plots.

    Args:
        wmoid: int
        params: 'all' to plot standard set of parameters listed below, or list of specific param_abbrev (if available)
        depth_lim: tuple/list of bounding depths (ylim[1] will default to shallowest observation ≤ depth_lim[1])
        fixed_ylim: True or False (force depth range to <<depth_lim>> [True], or set to deepest observation [False])
        vert_res: vertical resolution of section (in meters); note that plot size scales inversely with this
        toi: None or tuple/list of bounding times of interest (in 14-digit integer format)
        mld: plot mixed-layer depth
        mld_ref_depth: see gt.mld() (applies only if 'mld' is True)
        mld_sigma_theta_crit: see gt.mld() (applies only if 'mld' is True)
        show_ice_bars: plot bars estimating when float was under sea ice
                       note: calculates average SIC within a box 2° longitude x 1° latitude around the given or
                             interpolated float location (uses AMSR if available, then GSFC)
        sea_ice_grids: created by ldp.sea_ice_data_prep(), only needed if show_ice_bars is True
        sea_ice_data_avail: created by ldp.sea_ice_data_prep(), only needed if show_ice_bars is True
        show_prof_bars: plot each profile as thin gray line on section
        show_prof_ticks: plot each profile as small tick on top x-axis
        add_date_bars: None or list of Datetimes to add vertical black bars, e.g. denoting start and end of some event
        cmap_level_mods: None or dict with param_abbrevs as keys to lists of colormap levels to replace defaults
        cmap_color_mods: None or dict with param_abbrevs as keys to lists of color sequences to replace defaults
        cmap_gamma_mods: None or dict with param_abbrevs as keys to colormap shift parameter to replace defaults
                         note: gamma=1.0 is even spacing; gamma>1.0 stretches colors upwards; gamma<1.0 downwards
        cmap_freq_mods: None or dict with param_abbrevs as keys to multiplier for adding additional color levels
                        between those specified in 'cmap_levels' (e.g. 3 for 3 levels between)
        trim_xlim: trim xlim (time axis) to match range of data for each parameter (otherwise trim to time range of
                   GDAC temperature data)
        create_new_figs: True to save each sections as an individual new figure (with dimensions of new_figsize)
                         False to plot each section to the currently active plot axes
                                (for this, pass a single param_abbrev)
        new_figsize: figure dimensions in inches: (width,height), e.g. (8,6)
                     note: only used if create_new_figs is True
        facecolor: 'k' or other color for plot background (i.e. where data is missing or invalid)
        grid: True or False to add faint x- and y-grid at locations of major time and depth/density ticks
        years_only: True or False (only label years on x-axis, instead of automatic month labeling)
        plot_ylabel: True or False
        plot_xticklabels: True or False
        plot_cbar: True or False
        condensed_cbar_label: None or string to replace default colorbar parameter label
        smaller_text: True or False (use smaller font sizes, e.g. for subplot)
        force_label_size: None or fontsize for labels (smaller_text should be True)
        density_coor: True or False (if True, use sigma_theta as y-coordinate; if False, use depth as y-coordinate)
        density_lim: tuple/list of bounding sigma_theta values (only used if density_coor is True)
        density_power_scale: power exponent to stretch deeper density levels / condense near-surface levels
        density_depth_contours: None or list of depths to contour and label when plotting with density y-coordinate
        density_depth_labels: False or True (label the depth contours described above)
                              NOTE: this requires manual input (click and Return) to position contour labels
        explicit_yticks: None or list/array of ytick locations (depths or sigma_theta values)
        drift_temps: None or dict containing float drift-depth temperature time series with keys 'datetime' and 'temp'
        drift_temp_baseline: None or potential temperature value to use as baseline from which to plot drift_temps
        drift_temp_depth: None or depth to use as baseline from which to plot drift_temps
    """

    if params == 'all':
        param_abbrevs = np.array(['ptmp','psal','Nsquared','PV','destab','Oxygen','OxygenSat','pHinsitu','Nitrate',
                                  'Chl_a'])
        # full list of parameters below; implement custom colormaps as needed:
        # param_abbrevs = np.array(['ptmp', 'psal', 'Nsquared', 'PV', 'destab', 'Oxygen', 'OxygenSat', 'Nitrate',
        #                           'Chl_a', 'pHinsitu', 'pH25C', 'TALK_LIAR', 'DIC_LIAR', 'pCO2_LIAR'])
    else:
        param_abbrevs = params

    cmap_levels = {}
    cmap_levels['ptmp'] = [0.0,0.2,0.4,0.6,0.8,1.0,1.2]
    cmap_levels['psal'] = [34.66,34.67,34.68,34.69,34.70]
    cmap_levels['Oxygen'] = [190,195,200,205,210,215,220]
    cmap_levels['OxygenSat'] = [54,55,56,57,58,59,60,61,62,63,64,65,70,80,90,100,110]
    cmap_levels['Nsquared'] = [0,5,10,15,20,25,50,100,500,1000]
    cmap_levels['PV'] = [0,5,10,25,50,100,250,500,1000,5000]
    cmap_levels['sigma_theta'] = [27.0,27.78,27.79,27.80,27.81,27.82,27.83,27.84,27.85]
    cmap_levels['destab'] = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7]   # same range but even spacing
    cmap_levels['pHinsitu'] = np.arange(7.84,8.16,0.04)
    cmap_levels['Nitrate'] = np.arange(20.0,34.01,1.0)
    cmap_levels['Chl_a_corr'] = np.arange(0.0,2.0,0.25)

    cmap_extend = {}
    cmap_extend['ptmp'] = 'both'
    cmap_extend['psal'] = 'both'
    cmap_extend['Oxygen'] = 'both'
    cmap_extend['OxygenSat'] = 'both'
    cmap_extend['Nsquared'] = 'both'
    cmap_extend['PV'] = 'both'
    cmap_extend['sigma_theta'] = 'both'
    cmap_extend['destab'] = 'max'
    cmap_extend['pHinsitu'] = 'both'
    cmap_extend['Nitrate'] = 'both'
    cmap_extend['Chl_a_corr'] = 'both'

    cmap_under_over = {}
    cmap_under_over['ptmp'] = ['#252766','#62001d']   # darker purple/blue, darker red
    cmap_under_over['psal'] = ['#271E6A','#fcf6d1']   # darker purple/blue, lighter cream
    cmap_under_over['Oxygen'] = ['0.9','#660000']     # light grey-white, darker version of 'maroon'
    cmap_under_over['Nsquared'] = ['#000099','0.3']   # darker version of 'blue', dark grey
    cmap_under_over['PV'] = ['#000099','0.3']         # same as above
    cmap_under_over['destab'] = [None,'#2d004e']      # darker version of 'indigo'

    cmap_colors = {}
    # useful resources for color picking: https://matplotlib.org/examples/color/named_colors.html
    #                                     http://www.color-hex.com/
    cmap_colors['ptmp'] = ['#353992','#7294C2','#A5C4DD','#F9FCCF','#F2CF85','#CB533B','#8D002A']
    cmap_colors['psal'] = ['#252A83','#22369C','#215091','#306489','#3F7687','#569487',
                           '#6EB380','#87C574','#B4D56D','#DCE184','#FAEDA3']
    cmap_colors['Oxygen'] = ['white','maroon']   # previously ended with 'teal', started with '0.7'
    cmap_colors['OxygenSat'] = ['0.7','white','maroon','teal']
    cmap_colors['Nsquared'] = ['blue','#ffe34c','firebrick']   # ffe34c is lighter version of 'gold'
    cmap_colors['PV'] = ['blue','gold','firebrick']
    cmap_colors['sigma_theta'] = ['seagreen','white','coral','0.2']
    cmap_colors['destab'] = ['yellow','#9366b4','indigo']  #9366b4 is lighter indigo
    cmap_colors['pHinsitu'] = ['green','white','red','blue','orange']
    cmap_colors['Nitrate'] = ['orange','blue','red','white','green']
    cmap_colors['Chl_a_corr'] = ['#11114e','white','palegoldenrod','#005000']

    cmap_gamma = {}
    cmap_gamma['ptmp'] = 0.9
    cmap_gamma['psal'] = 0.7
    cmap_gamma['Oxygen'] = 1.75
    cmap_gamma['OxygenSat'] = 0.5
    cmap_gamma['Nsquared'] = 1.1
    cmap_gamma['PV'] = 0.7
    cmap_gamma['sigma_theta'] = 0.6  # colorbar is reversed below
    cmap_gamma['destab'] = 0.7       # colorbar is reversed below
    cmap_gamma['pHinsitu'] = 1.0
    cmap_gamma['Nitrate'] = 1.0
    cmap_gamma['Chl_a_corr'] = 1.0

    cmap_freq = {}
    cmap_freq['ptmp'] = 2
    cmap_freq['psal'] = 3
    cmap_freq['Oxygen'] = 2
    cmap_freq['OxygenSat'] = 2
    cmap_freq['Nsquared'] = 1
    cmap_freq['PV'] = 1
    cmap_freq['sigma_theta'] = 4
    cmap_freq['destab'] = 2
    cmap_freq['pHinsitu'] = 8
    cmap_freq['Nitrate'] = 8
    cmap_freq['Chl_a_corr'] = 5

    if cmap_level_mods is not None:
        for param in cmap_level_mods.keys():
            cmap_levels[param] = cmap_level_mods[param]
    if cmap_color_mods is not None:
        for param in cmap_color_mods.keys():
            cmap_colors[param] = cmap_color_mods[param]
    if cmap_gamma_mods is not None:
        for param in cmap_gamma_mods.keys():
            cmap_gamma[param] = cmap_gamma_mods[param]
    if cmap_freq_mods is not None:
        for param in cmap_freq_mods.keys():
            cmap_freq[param] = cmap_freq_mods[param]

    prof_match = np.zeros(len(float_data['profiles'])).astype(bool)
    for p in np.arange(len(prof_match)):
        if toi is not None:
            if toi[0] <= float_data['profiles'][p]['datetime'] <= toi[1]:
                prof_match[p] = True
        else:
            prof_match[p] = True
    prof_indices_to_plot = np.where(prof_match)[0]

    if mld or show_ice_bars:
        datetime_coord_profs = []
        datetime_coord_as_tuples = []
        mld_data = []
        prof_lats = []
        prof_lons = []
        for pi in prof_indices_to_plot:
            datetime_tuple_format = tt.convert_14_to_tuple(float_data['profiles'][pi]['datetime'])
            datetime_coord_as_tuples.append(datetime_tuple_format)
            datetime_coord_profs.append(tt.convert_tuple_to_datetime(datetime_tuple_format))
            this_mld = gt.mld(float_data['profiles'][pi],ref_depth=mld_ref_depth,
                              sigma_theta_crit=mld_sigma_theta_crit,verbose_warn=False)
            if density_coor:
                # actually a density value:
                this_mld = gt.vert_prof_eval(float_data['profiles'][pi],'sigma_theta',this_mld,extrap='nearest')
                # convert to power-scaled density; ignore MLDs outside plotting range
                if this_mld < density_lim[0]: this_mld = np.NaN
                this_mld = (this_mld - density_lim[0])**density_power_scale
            mld_data.append(this_mld)
            prof_lats.append(float_data['profiles'][pi]['lat'])
            prof_lons.append(float_data['profiles'][pi]['lon'])

    def DatetimeToTimestampForInterp(dt):
        return calendar.timegm(dt.timetuple())

    if show_ice_bars:
        date_coord_daily = tt.dates_in_range(datetime_coord_as_tuples[0][0:3],datetime_coord_as_tuples[-1][0:3])
        datetime_coord_daily = [tt.convert_tuple_to_datetime(date_tuple) for date_tuple in date_coord_daily]
        timestamp_coord_daily = [DatetimeToTimestampForInterp(dt) for dt in datetime_coord_daily]
        timestamp_coord_profs = [DatetimeToTimestampForInterp(dt) for dt in datetime_coord_profs]
        specific_lat_coord_for_ice = np.interp(timestamp_coord_daily,timestamp_coord_profs,prof_lats)
        specific_lon_coord_for_ice = np.interp(timestamp_coord_daily,timestamp_coord_profs,prof_lons)
        lat_coord_for_ice = []
        lon_coord_for_ice = []
        for pos_idx in range(len(specific_lat_coord_for_ice)):
            lat_coord_for_ice.append([specific_lat_coord_for_ice[pos_idx] - 0.5,
                                      specific_lat_coord_for_ice[pos_idx] + 0.5])
            lon_coord_for_ice.append([specific_lon_coord_for_ice[pos_idx] - 1.0,
                                      specific_lon_coord_for_ice[pos_idx] + 1.0])
        sic_coord = ldp.sea_ice_concentration_along_track(date_coord_daily,lat_coord_for_ice,lon_coord_for_ice,
                                                          sea_ice_grids,sea_ice_data_avail)

    for param_index, param_abbrev in enumerate(param_abbrevs):
        param_skip = True
        for pi in prof_indices_to_plot:
            if param_abbrev in float_data['profiles'][pi].keys(): param_skip = False
        if param_skip: continue

        datetime_coord = []
        section_data = []
        if density_coor: depth_data = []
        obs_range = []
        for pi in prof_indices_to_plot:
            if param_abbrev in float_data['profiles'][pi].keys():
                if float_data['profiles'][pi][param_abbrev]['data'].size == 0: continue
                z_vec, data_vec = gt.vert_prof_even_spacing(float_data['profiles'][pi],param_abbrev,z_coor='depth',
                                                            spacing=vert_res,interp_method='linear',extrap='NaN',
                                                            top=depth_lim[0],bottom=depth_lim[1],verbose_error=True)
                if density_coor:
                    obs_param = data_vec
                    obs_depth, obs_sigma_theta \
                        = gt.vert_prof_even_spacing(float_data['profiles'][pi],'sigma_theta',z_coor='depth',
                                                    spacing=vert_res,interp_method='linear',extrap='NaN',
                                                    top=depth_lim[0],bottom=depth_lim[1],verbose_error=True)
                    obs_good_mask = ~np.logical_or(np.isnan(obs_param),np.isnan(obs_sigma_theta))
                    obs_sort_order = obs_sigma_theta[obs_good_mask].argsort()
                    sorted_sigma_theta = obs_sigma_theta[obs_good_mask][obs_sort_order]
                    sorted_param = obs_param[obs_good_mask][obs_sort_order]
                    sorted_depth = obs_depth[obs_good_mask][obs_sort_order]
                    z_vec = density_lim[0] \
                            + (np.arange(0, (density_lim[1]-density_lim[0])**density_power_scale,
                                         ((density_lim[1]-density_lim[0])**density_power_scale)/200)) \
                              ** (1.0/density_power_scale)
                    data_vec = gt.profile_interp(sorted_param,sorted_sigma_theta,z_vec,
                                                 method='linear',out_of_bounds='NaN')
                    depth_vec = gt.profile_interp(sorted_depth,sorted_sigma_theta,z_vec,
                                                  method='linear',out_of_bounds='NaN')
                    depth_data.append(depth_vec)

                    z_vec[z_vec < density_lim[0]] = np.NaN
                    z_vec = (z_vec - density_lim[0])**density_power_scale

                section_data.append(data_vec)
                datetime_coord.append(tt.convert_tuple_to_datetime(tt.convert_14_to_tuple(float_data['profiles']
                                                                                          [pi]['datetime'])))
                obs_range.append([np.min(z_vec[np.isfinite(data_vec)]),
                                  np.max(z_vec[np.isfinite(data_vec)])])
                param_name_for_cbar = float_data['profiles'][pi][param_abbrev]['name']
                param_units_for_cbar = float_data['profiles'][pi][param_abbrev]['units']

        section_data = np.ma.masked_invalid(np.array(section_data).T)
        if density_coor: depth_data = np.ma.masked_invalid(np.array(depth_data).T)

        if create_new_figs: plt.figure(figsize=new_figsize)

        specified_levels = np.array(cmap_levels[param_abbrev])
        more_levels = np.interp(np.arange(len(specified_levels),step=1.0/cmap_freq[param_abbrev]),
                                np.arange(len(specified_levels)),specified_levels,right=np.NaN)
        more_levels = more_levels[~np.isnan(more_levels)]

        N_colors = len(more_levels) - 1
        contourf_cmap = mcolors.LinearSegmentedColormap.from_list(name=None,colors=cmap_colors[param_abbrev],
                                                                  N=N_colors,gamma=cmap_gamma[param_abbrev])

        if param_abbrev in cmap_under_over:
            if cmap_under_over[param_abbrev][0] is not None: contourf_cmap.set_under(cmap_under_over[param_abbrev][0])
            if cmap_under_over[param_abbrev][1] is not None: contourf_cmap.set_over(cmap_under_over[param_abbrev][1])
        normalization = mcolors.BoundaryNorm(more_levels,ncolors=N_colors,clip=False)

        # set facecolor as black (or other given color)
        if density_coor: plt.gca().axhspan((density_lim[1]-density_lim[0])**density_power_scale,0,
                                           facecolor=facecolor,zorder=1)
        else:            plt.gca().axhspan(depth_lim[0],depth_lim[1],facecolor=facecolor,zorder=1)
        contour_handle = plt.contourf(datetime_coord,z_vec,section_data,
                                      vmin=np.min(more_levels),vmax=np.max(more_levels),
                                      levels=more_levels,norm=normalization,cmap=contourf_cmap,
                                      extend=cmap_extend[param_abbrev],zorder=2)

        if plot_cbar:
            if show_ice_bars: shrink_cbar = 1650/(1650+175)
            else:             shrink_cbar = 1.0
            if np.max(np.abs(specified_levels)) >= 1000: formatter = pltick.FuncFormatter(lambda x, p: format(x, ','))
            else:                                        formatter = None
            cbar = plt.colorbar(ticks=specified_levels,spacing='uniform',shrink=shrink_cbar,format=formatter)
            if condensed_cbar_label is not None: cbar_label = condensed_cbar_label
            else:                                cbar_label = '{0}\n({1})'.format(param_name_for_cbar,
                                                                                  param_units_for_cbar)
            if smaller_text:
                if force_label_size is not None:
                    cbar_labelsize = force_label_size - 1
                    cbar_titlesize = force_label_size
                else:
                    cbar_labelsize = 6
                    cbar_titlesize = 8
                cbar.ax.tick_params(labelsize=cbar_labelsize)
                cbar.set_label(label=cbar_label,rotation=90,labelpad=9,size=cbar_titlesize)
            else:
                cbar.set_label(label=cbar_label,rotation=90,labelpad=11) # subtracted 9
                cbar.ax.set_title(param_units_for_cbar,fontsize=8)
            if param_abbrev == 'destab' or param_abbrev == 'sigma_theta': cbar.ax.invert_yaxis()

        if show_prof_bars:
            for obs_idx, obs_dt in enumerate(datetime_coord):
                plt.plot([obs_dt,obs_dt],obs_range[obs_idx],color='0.5',linewidth=0.5,zorder=3)

        if add_date_bars is not None:
            for dt in add_date_bars:
                if not density_coor: plt.plot([dt,dt],[*depth_lim],color='0.2',linewidth=0.8,zorder=3)
                else:                plt.plot([dt,dt],[*(np.array(density_lim)-density_lim[0])**density_power_scale],
                                              color='0.2',linewidth=0.8,zorder=3)

        if (drift_temps is not None) and (not density_coor):  # sorry, can't plot drift temps in density space
            plt.plot(drift_temps['datetime'],
                     drift_temp_depth + (depth_lim[1]-depth_lim[0])*(drift_temp_baseline-drift_temps['temp']),
                     'k-',linewidth=0.01,alpha=0.5,zorder=4)

        if trim_xlim:
            plt.xlim([datetime_coord[0],datetime_coord[-1]])
        else:
            start_date = tt.convert_tuple_to_datetime(tt.convert_14_to_tuple(float_data['profiles'][0]['datetime']))
            end_date = tt.convert_tuple_to_datetime(tt.convert_14_to_tuple(float_data['profiles'][-1]['datetime']))
            plt.xlim([start_date,end_date])

        if mld or show_ice_bars:
            mld_xlim_mask = np.logical_and(np.array(datetime_coord_profs) >= datetime_coord[0],
                                           np.array(datetime_coord_profs) <= datetime_coord[-1])
        if mld:
            plt.plot(np.array(datetime_coord_profs)[mld_xlim_mask],np.array(mld_data)[mld_xlim_mask],
                     'w-',linewidth=1.0,zorder=4)

        if density_coor and density_depth_contours is not None:
            depth_contours = plt.contour(datetime_coord,z_vec,depth_data,levels=density_depth_contours,
                                         linewidths=0.5,alpha=0.75,colors='k',zorder=5)
            if force_label_size: depth_contour_fontsize = force_label_size-1
            else:                depth_contour_fontsize = 8
            if density_depth_labels:
                print('>>> Waiting for manual input.\n'
                      '>>> Click to position contours, hit Return when done.\n'
                      '>>> Note: do not change figure size.')
                clabels = plt.clabel(depth_contours,fmt='%d m',fontsize=depth_contour_fontsize,manual=True,
                                     inline=True,inline_spacing=25)  # removed zorder=5
                for label in clabels: label.set_rotation(0)


        if fixed_ylim and not density_coor: max_ylim = depth_lim[1]
        else:                               max_ylim = np.max(np.array(obs_range))
        if show_ice_bars:
            sic_xlim_mask = np.logical_and(np.array(datetime_coord_daily) >= datetime_coord[0],
                                           np.array(datetime_coord_daily) <= datetime_coord[-1])

            if not smaller_text: sic_norm = -35         # meters equivalent
            elif depth_lim[1] <= 300: sic_norm = -35    # hackish temp solution for upper-ocean-only sections
            else:                sic_norm = -175

            if not density_coor:
                sic_baseline = depth_lim[0]    # top of section (probably 0 m, but not necessarily)
            else:
                sic_norm = (sic_norm/1700)*((density_lim[1]-density_lim[0])**density_power_scale)
                sic_baseline = 0

            plt.gca().fill_between(np.array(datetime_coord_daily)[sic_xlim_mask],
                                   sic_baseline + (sic_norm * np.array(sic_coord)[sic_xlim_mask]),sic_baseline,
                                   color='k',linewidth=0,zorder=5) # or #8aacb8 for blue
            plt.plot([datetime_coord_daily[0],datetime_coord_daily[-1]],[sic_baseline,sic_baseline],'k-',linewidth=0.5)
            if not density_coor: plt.ylim([sic_baseline + 1.2*sic_norm,max_ylim])
            else:                plt.ylim(sic_baseline + 1.2*sic_norm,
                                          (density_lim[1]-density_lim[0])**density_power_scale)
        else:
            if not density_coor: plt.ylim([depth_lim[0],max_ylim])
            else:                plt.ylim(*(np.array(density_lim)-density_lim[0])**density_power_scale)
        plt.gca().invert_yaxis()
        if not density_coor:
            if explicit_yticks is not None: plt.yticks(explicit_yticks)
            plt.gca().get_yaxis().set_major_formatter(pltick.FuncFormatter(lambda x, loc: "{:,}".format(x)))
        else:
            plt.yticks((np.array(explicit_yticks)-density_lim[0])**density_power_scale)
            plt.gca().set_yticklabels(explicit_yticks)
        if show_ice_bars and not density_coor:  # NOTE: weird numbers display when using this in density coordinates
            current_yticks = plt.yticks()[0]
            plt.yticks([sic_baseline+sic_norm,sic_baseline,*current_yticks[1:]],
                       ['100%','0%',*["{:,}".format(yt) for yt in current_yticks[1:]]])
        if smaller_text:
            if force_label_size is not None: ysize = force_label_size
            else:                            ysize = 8
        else:                                ysize = None
        if not plot_ylabel: plt.gca().yaxis.set_ticklabels([])
        else:               plt.gca().tick_params(axis='y',which='major',labelsize=ysize)
        if plot_ylabel:
            if not density_coor: plt.ylabel('Depth (m)',size=ysize)
            else:                plt.ylabel(r'$\sigma_\theta$ (kg/m$^3$)',size=ysize)
            if show_ice_bars:
                plt.ylabel('Depth (m)      ',size=ysize)
                plt.text(-0.14,0.93,'SIC',fontsize=ysize,rotation=0,transform=plt.gca().transAxes,
                         horizontalalignment='right',verticalalignment='center')

        years = mdates.YearLocator()
        months = mdates.MonthLocator()
        if not years_only:
            xaxis_formatter = mdates.DateFormatter("%b")
            plt.gca().xaxis.set_major_locator(months)
            plt.gca().xaxis.set_major_formatter(xaxis_formatter)
        else:
            xaxis_formatter = mdates.DateFormatter("%Y")
            plt.gca().xaxis.set_major_locator(years)
            plt.gca().xaxis.set_major_formatter(xaxis_formatter)
            plt.gca().xaxis.set_minor_locator(months)
        plt.xticks(rotation=45)
        if not plot_xticklabels:
            plt.gca().xaxis.set_ticklabels([])
        elif force_label_size is not None:
            plt.gca().tick_params(axis='x',which='major',labelsize=force_label_size)

        if grid:
            plt.grid(which='major',axis='both',color='0.6',linewidth=0.25,alpha=0.6)
            plt.gca().set_axisbelow(False) # True (grid below all), 'line' (below lines), False (grid above all)

        if show_prof_ticks:
            top_xaxis = plt.gca().twiny()
            top_xaxis.set_xlim([datetime_coord[0],datetime_coord[-1]])
            top_xaxis.xaxis.set_ticks_position('top')
            top_xaxis.xaxis.set_tick_params(width=0.5)
            top_xaxis.set_xticks(datetime_coord)
            top_xaxis.xaxis.set_ticklabels([])

        if create_new_figs:
            if add_title: plt.title('Float {0}'.format(wmoid))
            plt.tight_layout()
            plt.savefig(results_dir + save_as + param_abbrev + '.pdf')
            plt.close()


def section_compiler(wmoids,data_dir,results_dir,save_as,float_data,params,figsize=(8.5,11),depth_lim=(0,1000),
                     fixed_ylim=True,mld=True,sea_ice_grids=None,sea_ice_data_avail=None,add_date_bars=None,
                     condensed_cbar_labels=None,width_ratios=None,height_ratios=None,all_trajs=None,
                     traj_plot_params=None,show_ice_bars=True,density_coor=False,density_lim=None,
                     density_power_scale=None,density_depth_contours=None,plot_title=True,force_label_size=None,
                     explicit_yticks=None,w_pad=0.0,drift_temps=None,drift_temp_baseline=None,drift_temp_depth=None,
                     years_only=None):
    """ Arrange multiple hydrographic sections and float trajectories on a single plot. Wrapper method for pt.section().
    """
    
    plt.figure(figsize=figsize)
    if all_trajs is not None:
        params = ['__trajectories__',*params]
        if condensed_cbar_labels is not None: condensed_cbar_labels = ['__trajectories__',*condensed_cbar_labels]
        first_param_idx = 1
    else:
        first_param_idx = 0
    subplot_grid = gridspec.GridSpec(len(params),len(wmoids),width_ratios=width_ratios,height_ratios=height_ratios)
    for float_idx, wmoid in enumerate(wmoids):
        for param_idx, param in enumerate(params):
            plt.subplot(subplot_grid[len(wmoids)*param_idx + float_idx])
            if param_idx == 0 and plot_title: plt.title('Float {0}'.format(wmoid),size=8,fontweight='bold')
            if param == '__trajectories__':
                argo_traj(data_dir,None,all_trajs[float_idx],*traj_plot_params[float_idx],label_dates=True,
                          save_as=None,label_placement=(0.04,-0.25),boundary_width=1,labelsize=4,
                          label_dates_12mo_only=True)
                if float_idx+1 == len(wmoids):
                    plt.gca().set_anchor('W')
                continue
            if param_idx == first_param_idx: ice_bars_yes_no = show_ice_bars; show_prof_ticks = True
            else:                            ice_bars_yes_no = False;         show_prof_ticks = False
            if float_idx == 0: plot_ylabel = True;  density_depth_labels = True
            else:              plot_ylabel = False; density_depth_labels = False
            if float_idx == len(wmoids)-1: plot_cbar = True
            else:                          plot_cbar = False
            if param_idx == len(params)-1: plot_xticklabels = True;  dd_contours = density_depth_contours
            else:                          plot_xticklabels = False; dd_contours = None
            if param == 'ptmp' and drift_temps is not None:
                dt = drift_temps[wmoid]; dtb = drift_temp_baseline; dtd = drift_temp_depth
            else:
                dt = None;               dtb = None;                dtd = None
            section(wmoid,None,None,float_data[float_idx],params=[param],depth_lim=depth_lim,fixed_ylim=fixed_ylim,
                    mld=mld,show_ice_bars=ice_bars_yes_no,sea_ice_grids=sea_ice_grids,sea_ice_data_avail=sea_ice_data_avail,
                    show_prof_ticks=show_prof_ticks,add_date_bars=add_date_bars,trim_xlim=False,create_new_figs=False,
                    plot_ylabel=plot_ylabel,plot_xticklabels=plot_xticklabels,years_only=years_only[float_idx],plot_cbar=plot_cbar,
                    condensed_cbar_label=condensed_cbar_labels[param_idx],smaller_text=True,density_coor=density_coor,
                    density_lim=density_lim,density_power_scale=density_power_scale,force_label_size=force_label_size,
                    density_depth_contours=dd_contours,density_depth_labels=density_depth_labels,
                    explicit_yticks=explicit_yticks,drift_temps=dt,drift_temp_baseline=dtb,drift_temp_depth=dtd)

    plt.tight_layout(h_pad=0.2,w_pad=w_pad) # can go negative if necessary
    plt.savefig(results_dir + save_as + '.pdf')
    plt.close()


def prof_locations_map(results_dir,data_dir,compiled_obs,map_dimensions,
                       toi_range=[datetime(1900,1,1),datetime.today()],bathy_cmap='Greys_r',
                       seasons=[[1,3],[4,6],[7,9],[10,12]],season_colors=['orange','cyan','orchid','lime'],
                       season_labels=['Jan-Mar','Apr-Jun','Jul-Sep','Oct-Dec'],
                       manual_list_of_types=None,manual_labels_for_types=None,
                       manual_markers_for_types=None,manual_marker_open_for_types=None,
                       grid_lats=np.arange(-80,60,5),grid_lons=np.arange(-80,50,10),
                       lon_labels=[0,0,1,0],lat_labels=[1,0,0,0],label_contours=False,
                       add_epoch_title=None,fontsize=5,fontsize_extra_for_epoch=2,
                       add_rect_patch=None,add_circ_patch=None,add_legend=False,legend_pos='outside_bottom',
                       create_new_fig=False,use_existing_basemap=None,return_basemap=False,save_as=None):
    """ Plot locations of hydrographic observations, as compiled by ldp.compile_hydrographic_obs(),
        by season and type (source) for a given epoch.

    Args:
        add_rect_patch: None or [lon_W,lon_E,lat_S,lat_N]
        add_circ_patch: None or [[lon_cent,lat_cent,radius_in_km], etc.], i.e. a list of params for multiple circles
        legend_pos: 'outside_bottom' or 'outside_right'

    """
    if use_existing_basemap is None:
        fig,m = bathy_basemap(data_dir,*map_dimensions,create_new_fig=create_new_fig,figsize=(9,9),
                              boundary_width=1,labelsize=fontsize,grid_color='.2',
                              grid_lats=grid_lats,grid_lons=grid_lons,force_lon_labels=lon_labels,force_lat_labels=lat_labels,
                              label_contours=label_contours,cmap=bathy_cmap)
    else:
        m = use_existing_basemap

    if add_rect_patch is not None:
        ap = add_rect_patch
        patch_lons = np.concatenate((np.linspace(ap[0],ap[1],100),np.linspace(ap[1],ap[1],100),
                                     np.linspace(ap[1],ap[0],100),np.linspace(ap[0],ap[0],100)))
        patch_lats = np.concatenate((np.linspace(ap[3],ap[3],100),np.linspace(ap[3],ap[2],100),
                                     np.linspace(ap[2],ap[2],100),np.linspace(ap[2],ap[3],100)))
        plonx,platy = m(patch_lons,patch_lats)
        patchxy = list(zip(plonx,platy))
        poly = Polygon(patchxy,facecolor='white',alpha=0.1)
        plt.gca().add_patch(poly)

    if add_circ_patch is not None:
        for circ in add_circ_patch:
            circle_tuples = circle(m,*circ)
            poly = Polygon(list(circle_tuples),facecolor='white',alpha=0.1)
            plt.gca().add_patch(poly)

    toi_mask_base = np.logical_and(np.array(compiled_obs['datetimes']) >= toi_range[0],
                                   np.array(compiled_obs['datetimes']) <= toi_range[1])
    dt_months = np.array([dt.month for dt in compiled_obs['datetimes']])

    if manual_list_of_types is None: obs_types = np.unique(compiled_obs['types'][toi_mask_base])
    else:                            obs_types = manual_list_of_types
    if manual_labels_for_types is None: obs_type_labels = obs_types
    else:                               obs_type_labels = manual_labels_for_types
    if manual_markers_for_types is None: obs_type_markers = ['o','s','^','v','<','>','p','*','+','d'] # etc.
    else:                                obs_type_markers = manual_markers_for_types
    if manual_marker_open_for_types is None: obs_type_markers_open = np.tile(False,len(obs_type_markers))
    else:                                    obs_type_markers_open = manual_marker_open_for_types

    for s_idx, season_months in enumerate(seasons):
        toi_mask = np.logical_and(toi_mask_base,np.logical_and(dt_months >= season_months[0],
                                                               dt_months <= season_months[1]))
        for t_idx, obs_type in enumerate(obs_types):
            final_mask = np.logical_and(toi_mask,np.array(compiled_obs['types']) == obs_type)
            if sum(final_mask) > 0:
                lonx,laty = m(np.array(compiled_obs['lons'])[final_mask],np.array(compiled_obs['lats'])[final_mask])
                if obs_type_markers_open[t_idx]: plt.scatter(lonx,laty,s=4.0,marker=obs_type_markers[t_idx],
                                                             facecolor='none',edgecolors=season_colors[s_idx],
                                                             linewidths=0.5)
                else:                            plt.scatter(lonx,laty,s=4.0,marker=obs_type_markers[t_idx],
                                                             facecolor=season_colors[s_idx],edgecolors='none')

    if add_epoch_title is not None:
        plt.text(0.05,0.95,add_epoch_title,color='w',fontsize=fontsize+fontsize_extra_for_epoch,fontweight='bold',
                 horizontalalignment='left',verticalalignment='top',transform=plt.gca().transAxes)

    if add_legend:
        for s_idx, season_months in enumerate(seasons):
            plt.plot([0,0],[np.nan,np.nan],lw=0,c=season_colors[s_idx],marker='o',ms=4,label=season_labels[s_idx])
        for t_idx, obs_type in enumerate(obs_types):
            if obs_type_markers_open[t_idx]:
                plt.plot([0,0],[np.nan,np.nan],lw=0,marker=obs_type_markers[t_idx],ms=4,
                         markerfacecolor='none',markeredgecolor='k',markeredgewidth=0.5,label=obs_type_labels[t_idx])
            else:
                plt.plot([0,0],[np.nan,np.nan],lw=0,marker=obs_type_markers[t_idx],ms=4,
                         markerfacecolor='k',markeredgecolor='none',label=obs_type_labels[t_idx])

        if legend_pos == 'outside_bottom':
            ncol = len(seasons)+len(obs_types)
            loc = 'upper right'
            bbox_to_anchor = [0.5,-0.05]
            handletextpad = 0.05
            columnspacing = 1.5
            labelspacing = None
        elif legend_pos == 'outside_right':
            ncol = 2
            loc = 'center left'
            bbox_to_anchor = [1.15,0.5]
            handletextpad = 0.25
            columnspacing = 1.5
            labelspacing = 1.5

        plt.legend(ncol=ncol,fontsize=fontsize,loc=loc,bbox_to_anchor=bbox_to_anchor,frameon=False,
                   handletextpad=handletextpad,columnspacing=columnspacing,labelspacing=labelspacing)

    if save_as is not None:
        plt.tight_layout()
        plt.savefig(results_dir + save_as + '.pdf')
        plt.close()
    elif return_basemap:
        return m


def era_field(data_dir,results_dir,save_as,data,datetime_range,width,height,lat_center,lon_center,bathy_contours=[],
              contour=True,contour_lims=None,n_contours=21,use_cmap=None,add_cbar=True,
              existing_canvas=None,return_pcm=False,
              add_wind_vectors=None,wind_vector_downsample=[5,2],wind_vector_scale=50,
              add_wind_vector_key=True,wind_vector_key=20,wind_vector_key_loc=[0.8,-0.25],wind_vector_key_fontsize=8,
              add_sic_contours=None,sic_contours=[50],
              add_date=None,date_string_loc=[0.05,0.95],date_string_size=8,
              date_string_valign='top',date_string_halign='left',average_daily=True,add_patch_lons_lats=None):
    """ Plotting routine for daily ECMWF fields.

    Args:
        data_dir: data directory (for bathymetry files)
        results_dir: None to plot on existing canvas, or directory to save plot
        save_as: None (to not save figure) or filename, without extension
        data: xarray DataArray containing reanalysis parameter, e.g. erai_daily['u10']
        datetime_range: single datetime to plot, or range of datetimes ([start,end]) to average for plot
            note: if <<average_daily>> is True, averages over all hours during a given day, from hour 0 to 23
        width:  Basemap plot width
        height: Basemap plot height
        lat_center: Basemap plot latitude center location
        lon_center: Basemap plot longitude center location
        bathy_contours: list of depths to add bathymetric contours
        contour: True or False to draw filled contour plot of <<data>>
        contour_lims: None or [min,max] to specify contour color limits
        n_contours: number of contour levels to plot, if contour_lims is specified (default = 21)
        add_cbar: plot colorbar? True or False
        existing_canvas: None or handle of Basemap instance (m) to plot onto
        return_pcm: return handle ('pcm') to pcolormesh of field
        add_wind_vectors: None or xarray DataArrays for [u,v] to plot wind vectors from fields
                          note: assumes lats and lons are same as for main 'data' DataArray above
        wind_vector_downsample: [i,j] to plot every ith u-wind and jth v-wind vector
        wind_vector_scale: length of wind vectors (larger numbers are smaller vectors)
        add_wind_vector_key: add quiver key next to plot, representing size of [N] m/s wind vector
        wind_vector_key: depict a [N] m/s vector key
        wind_vector_key_loc: location of wind vector in axes coordinates from bottom left (x, y)
        wind_vector_key_fontsize: fontsize of 'N m/s' key text
        add_sic_contours: None or list of [sic_grid['lons'],sic_grid['lats'],sic_field]
        sic_contours: [50] or list of other SIC % levels to contour
        add_date: None or formatted date string to add as text
        date_string_loc: location in axes coordinates (x,y) from bottom left for date string
        date_string_size: fontsize for date string
        date_string_valign: vertical alignment of date string location ('top' or 'bottom')
        date_string_halign: horizontal alignment of date string location ('left' or 'right')
        average_daily: if True, ignores hour value(s) of <<datetime_range>> and averages over day
                       if False, keeps hour value(s) of <<datetime_range>>
        add_patch_lons_lats: None or box coordinates to plot as shaded patch: [lon_W,lon_E,lat_S,lat_N]

    """
    if contour_lims is not None: contour_levs = np.linspace(contour_lims[0],contour_lims[1],n_contours)
    else:                        contour_levs = None

    if not isinstance(datetime_range,list) and not isinstance(datetime_range,tuple):
        dtr = [datetime_range,datetime_range]   # i.e. if only single datetime specified
    else:
        dtr = datetime_range
    if average_daily:   # if not average_daily, interpret datetime_range exactly and slice accordingly
        dtr[0] = datetime(dtr[0].year,dtr[0].month,dtr[0].day,0)
        dtr[1] = datetime(dtr[1].year,dtr[1].month,dtr[1].day,23,59,59)
    data = data.sel(time=slice(*dtr)).mean(dim='time',keep_attrs=True)

    if add_wind_vectors is not None:
        u_data = add_wind_vectors[0].sel(time=slice(*dtr)).mean(dim='time',keep_attrs=True)
        v_data = add_wind_vectors[1].sel(time=slice(*dtr)).mean(dim='time',keep_attrs=True)

    if existing_canvas is None:
        fig, m = lambert_basemap(width,height,lat_center,lon_center,
                                 boundary_width=1,lon_labels_on_top=True,resolution='i')
    else:
        fig = plt.gcf()
        m = existing_canvas

    lon_grid, lat_grid = np.meshgrid(data['lons'],data['lats'])
    rlons, rlats = m(lon_grid, lat_grid)

    if contour:
        if use_cmap is None:
            if contour_lims is not None:
                if contour_lims[1] == abs(contour_lims[0]): cmap = 'PRGn'
                else:                                       cmap = 'viridis'
            else:
                cmap = 'viridis'
        else:
            cmap = use_cmap
        with warnings.catch_warnings():  # ignore Dask true_divide warning upon evaluating data
            warnings.simplefilter('ignore')
            pcm = m.contourf(rlons,rlats,data,levels=contour_levs,cmap=cmap,extend='both')

        if add_cbar:
            cbar = plt.colorbar()
            cbar.set_label('{0} ({1})'.format(data.attrs['long_name'],data.attrs['units']),rotation=-90,labelpad=15)

    if add_wind_vectors:
        [i,j] = wind_vector_downsample
        Q = plt.quiver(rlons[::j,::i],rlats[::j,::i],u_data[::j,::i],v_data[::j,::i],
                       units='width',scale=wind_vector_scale,width=0.01,zorder=10)
        if add_wind_vector_key: plt.quiverkey(Q,*wind_vector_key_loc,wind_vector_key,
                                              r'{0} '.format(wind_vector_key) + r'm s$^{-1}$',
                                              fontproperties={'size':wind_vector_key_fontsize})

    if add_sic_contours is not None:
        sic_lonx,sic_laty = m(add_sic_contours[0],add_sic_contours[1])
        plt.contour(sic_lonx,sic_laty,add_sic_contours[2],levels=sic_contours,
                    colors='k',linewidths=0.5,alpha=0.8,zorder=5)

    if len(bathy_contours) > 0:
        etopo_lons,etopo_lats,etopo = ldp.load_bathy(data_dir)
        retopolons,retopolats = m(*np.meshgrid(etopo_lons,etopo_lats))
        olevels = bathy_contours  # check etopo.ravel().min()
        m.contour(retopolons,retopolats,etopo,olevels,linewidths=0.5,linestyles='solid',colors='#808080',
                  alpha=0.5,zorder=4)

    if add_patch_lons_lats is not None:
        pll = add_patch_lons_lats
        patch_lons = np.concatenate((np.linspace(pll[0],pll[1],100),np.linspace(pll[1],pll[1],100),
                                     np.linspace(pll[1],pll[0],100),np.linspace(pll[0],pll[0],100)))
        patch_lats = np.concatenate((np.linspace(pll[3],pll[3],100),np.linspace(pll[3],pll[2],100),
                                     np.linspace(pll[2],pll[2],100),np.linspace(pll[2],pll[3],100)))
        plonx,platy = m(patch_lons,patch_lats)
        patchxy = list(zip(plonx,platy))
        poly = Polygon(patchxy,facecolor='white',alpha=0.25,zorder=3)
        plt.gca().add_patch(poly)

    if add_date is not None:
        plt.text(*date_string_loc,add_date,fontsize=date_string_size,fontweight='bold',
                 horizontalalignment=date_string_halign,verticalalignment=date_string_valign,
                 transform=plt.gca().transAxes)

    if save_as is not None:
        plt.savefig(results_dir + save_as + '.pdf')
        plt.close()

    if return_pcm and contour:
        return pcm


############# AUXILIARY (INTERNAL) FUNCTIONS ################


def lambert_basemap(width,height,lat_center,lon_center,boundary_width=2,create_new_fig=True,figsize=None,resolution='i',
                    draw_grid=True,lon_labels_on_top=False,grid_color='0.2',meridians=np.arange(-80,50,20)):
    """ Creates basic figure on a Lambert azimuthal equal-area projection. 
    """
    warnings.filterwarnings('ignore', category=mcbook.mplDeprecation)

    if create_new_fig: fig = plt.figure(figsize=figsize)
    else:              fig = plt.gcf()

    m = Basemap(width=width, height=height, resolution=resolution, projection='laea', lat_ts=lat_center,
                lat_0=lat_center, lon_0=lon_center)
    m.drawcoastlines(color='k')
    m.drawmapboundary(linewidth=boundary_width)
    m.fillcontinents()
    if draw_grid:
        if lon_labels_on_top: lon_labels = [0,0,1,0]
        else:                 lon_labels = [0,0,0,1]
        m.drawmeridians(meridians, linewidth=0.5, color=grid_color, labels=lon_labels)
        m.drawmeridians(np.arange(-80, 50, 10), linewidth=0.5, color=grid_color)
        m.drawparallels(np.arange(-80, 60,  5), linewidth=0.5, color=grid_color, labels=[1, 0, 0, 0])

    return fig, m


def bathy_basemap(data_dir,width,height,lat_center,lon_center,create_new_fig=True,figsize=None,
                  labelsize=None,lon_labels_on_top=False,force_lon_labels=None,force_lat_labels=[1,0,0,0],
                  boundary_width=2,grid_color='0.2',grid_lats=np.arange(-80,60,5),grid_lons=np.arange(-80,50,10),
                  label_contours=False,cmap=cmocean.cm.deep_r,bathy_alpha=1.0):
    """ Draws bathymetry on LAEA (Lambert) basemap.

    Currently the figure parameters are not entirely defined through arguments above. This could be remedied.

    """
    
    fig, m = lambert_basemap(width,height,lat_center,lon_center,create_new_fig=create_new_fig,figsize=figsize,
                             draw_grid=False,boundary_width=boundary_width)

    lons, lats, etopo = ldp.load_bathy(data_dir)
    rlons, rlats = m(*np.meshgrid(lons, lats))
    olevels = np.arange(-7000, 760, 750)  # check etopo.ravel().min()
    cf = m.contourf(rlons, rlats, etopo, olevels, cmap=cmap, alpha=bathy_alpha, zorder=1)
    if label_contours:
        if cmap == 'Greys_r': contour_line_cmap = 'Greys_r'
        else:                 contour_line_cmap = None
        co = m.contour(rlons,rlats,-1*etopo,[2500,3250,4000,4750],linewidths=0.0,alpha=0.5,cmap=contour_line_cmap,zorder=1)
        print('>>> Waiting for manual input.\n'
              '>>> Click to position contours, hit Return when done.\n'
              '>>> Note: do not change figure size.')
        if cmap == 'Greys_r': clabel_single_color = 'k'  # or change to None to use reversed grayscale cmap
        else:                 clabel_single_color = 'w'
        plt.clabel(co,colors=clabel_single_color,fmt='%d',fontsize=labelsize-1,manual=True,inline=True)
    if lon_labels_on_top: lon_labels = [0, 0, 1, 0]
    else:                 lon_labels = [0, 0, 0, 1]
    lat_labels = [1,0,0,0]
    if force_lon_labels is not None: lon_labels = force_lon_labels
    if force_lat_labels is not None: lat_labels = force_lat_labels
    m.drawmeridians(grid_lons,color=grid_color,linewidth=0.5,labels=lon_labels,fontsize=labelsize,zorder=2)
    m.drawparallels(grid_lats,color=grid_color,linewidth=0.5,labels=lat_labels,fontsize=labelsize,zorder=2)

    return fig, m


def blank_inset_basemap(width,height,lat_center,lon_center,create_new_fig=True,
                        boundary_width=2,coastline_width=1,lon_labels=[0,0,0,0],lat_labels=[0,0,0,0],
                        grid_lats=np.arange(-80,60,5),grid_lons=np.arange(-80,50,10),labelsize=None,grid_color='0.2',
                        fill_continent_zorder=None,lat_lon_line_zorder=None,fill_continent_color='0.8',
                        resolution='i'):
    """ Creates figure with regional Lambert basemap, to be filled elsewhere with a plot.
    """
    
    if create_new_fig: fig = plt.figure(figsize=(9, 7))
    else:              fig = plt.gcf()

    m = Basemap(width=width, height=height, resolution=resolution, projection='laea', lat_ts=lat_center,
                lat_0=lat_center, lon_0=lon_center)
    m.drawcoastlines(linewidth=coastline_width,color='k',zorder=fill_continent_zorder)
    m.drawmapboundary(linewidth=boundary_width, fill_color='#f0ffff')
    m.fillcontinents(color=fill_continent_color,zorder=fill_continent_zorder)
    if create_new_fig:
        print('Temporary warning from pt.blank_inset_basemap(): '
              'explicit setting of lat/lon labels is turned off; make sure this is okay')
        # lon_labels = [0, 0, 0, 1]
        # lat_labels = [1, 0, 0, 0]
    m.drawmeridians(grid_lons, color=grid_color, linewidth=0.5, labels=lon_labels, fontsize=labelsize,
                    zorder=lat_lon_line_zorder)
    m.drawparallels(grid_lats, color=grid_color, linewidth=0.5, labels=lat_labels, fontsize=labelsize,
                    zorder=lat_lon_line_zorder)

    return fig, m