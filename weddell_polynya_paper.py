# -*- coding: utf-8 -*-

print('Script is loading dependencies.')

# external imports
import os
from numpy import *
import scipy.io as sio
import scipy.interpolate as spin
from scipy import stats
import pandas as pd
import pandas.plotting._converter as pandacnv   # FIXME: only necessary due to Pandas 0.21.0 bug with Datetime plotting
pandacnv.register()                             # FIXME: only necessary due to Pandas 0.21.0 bug with Datetime plotting
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cbook as mcbook
import matplotlib.dates as mdates
import matplotlib.ticker as pltick
import matplotlib.legend as mlegend
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Polygon, Rectangle, ConnectionPatch
os.environ['PROJ_LIB'] = '/Applications/anaconda/share/proj'    # temporarily necessary for Basemap import
from mpl_toolkits.basemap import Basemap
from matplotlib import gridspec
from datetime import datetime, timedelta
from collections import OrderedDict
from Circles.circles import circle   # from https://github.com/urschrei/Circles
import pickle
import warnings
import time
import gsw

# import custom functions
import download_product as dlp
import load_product as ldp
import time_tools as tt
import plot_tools as pt
import geo_tools as gt

# custom settings
set_printoptions(threshold=100)    # speeds up printing of large dicts containing NumPy arrays during debugging
plt.ion()                          # necessary for interactive contour label placement

# prettier font for plots
# note: before uncommenting, have to 'install' Helvetica using fondu (see instructions at https://goo.gl/crpbt2)
# mpl.rc('font',**{'family':'sans-serif','sans-serif':'Helvetica'})

####################################################################################################

# download data
download_argo = False
download_soccom = False
download_amsr2 = False
download_amsre = False
download_dmsp_nrt = False
download_dmsp_v3 = False
download_nimbus5 = False
download_ecmwf = False
download_isd = False

# data preparation/processing routines
argo_cal_5904468 = False
process_ecmwf = False
load_erai_land_sea_mask = True      # always keep this True!

# analyze data and generate paper figures
plot_fig_1 = False                  # note: this routine requires manual input for contour labels
plot_fig_2_ED_figs_3_4 = False
plot_fig_3 = False
plot_fig_4_ED_figs_6_7 = False
plot_fig_5_ED_fig_9_ED_table_1 = False
plot_ED_fig_1 = False
plot_ED_figs_2_8 = False
plot_ED_fig_5 = False

# directory for plotting output
current_results_dir = 'Results/'

# directory for h4toh5 executable
script_dir = os.getcwd() + '/'

# root directory for data files
data_dir = 'Data/'

# sub-directories for data files
argo_gdac_dir = data_dir + 'Argo/'
soccom_dir = argo_gdac_dir + 'SOCCOM/'
uw_o2_dir = argo_gdac_dir + 'UW-O2/'
shipboard_dir = data_dir + 'Shipboard/'
wod_dir = shipboard_dir + 'WOD/'
waghc_dir = shipboard_dir + 'WAGHC2017/'
amsr2_dir = data_dir + 'Sea ice concentration/AMSR2/'
amsre_dir = data_dir + 'Sea ice concentration/AMSR-E/'
dmsp_nrt_dir = data_dir + 'Sea ice concentration/DMSP_NRT/'
dmsp_v3_dir = data_dir + 'Sea ice concentration/DMSP_v3/'
nimbus5_dir = data_dir + 'Sea ice concentration/Nimbus-5/'
amsr_gridfile = data_dir + 'Sea ice concentration/AMSR_grid/LongitudeLatitudeGrid-s6250-Antarctic.h5'
amsr_areafile = data_dir + 'Sea ice concentration/AMSR_grid/pss06area_v3.dat'
nsidc_ps25_grid_dir = data_dir + 'Sea ice concentration/NSIDC_polar_stereo_25km_grid/'
coastline_filename_prefix = data_dir + 'GSHHG coast shapefiles/l/GSHHS_l_L5'
climate_indices_dir = data_dir + 'Climate indices/'
reanalysis_dir = data_dir + 'Reanalysis/'
era_new_dir = data_dir + 'Reanalysis/ECMWF_Weddell_unprocessed/'
era_custom_dir = data_dir + 'Reanalysis/ECMWF_Weddell_processed/'
era_processed_gwk_moore_dir = data_dir + 'Reanalysis/ECMWF_processed_GWKMoore/'
isd_dir = data_dir + 'ISD station records/'
reader_dir = data_dir + 'READER station records/'

# sub-directories for serialized ("pickled") processed data
figure_pickle_dir = data_dir + 'Processed_pickle_archives/'
argo_index_pickle_dir = argo_gdac_dir + 'Argo_index_pickles/'

############################################ DATA DOWNLOAD ##########################################################

print('Script is starting.')

# download Argo data from GDAC and update pickles
if download_argo:
    # note: set overwrite_global_index to True when checking for new/updated profiles
    dlp.argo_gdac((1990,1,1),(2018,10,1),[-80,-55],[-70,50],argo_gdac_dir,overwrite_global_index=False,
                  overwrite_profs=False,bypass_download=False,
                  only_download_wmoids=[])
    argo_gdac_index = ldp.argo_gdac_load_index(argo_gdac_dir)
    pickle.dump(argo_gdac_index,open(argo_index_pickle_dir + 'argo_gdac_index.pickle','wb'))

# correct float 5904468 for salinity drift
# note: run this after downloading new Argo data; see #FIXME notes in ldp.argo_float_data()
if argo_cal_5904468:
    argo_gdac_index = pickle.load(open(argo_index_pickle_dir + 'argo_gdac_index.pickle','rb'))
    argo_soccom_index = pickle.load(open(argo_index_pickle_dir + 'argo_soccom_index.pickle','rb'))

    # export calibration based on average salinity between 1600-1700 m
    gdac_data = ldp.argo_float_data(5904468,argo_gdac_dir,argo_gdac_index,argo_soccom_index,
                                    prof_nums='all',verbose=False,use_unadjusted=False,correct_5904468_interim=False)
    gdac_num_profs = len(gdac_data['profiles'])
    gdac_prof_nums = array([gdac_data['profiles'][p]['prof_num'] for p in range(gdac_num_profs)])
    gdac_time_coord = array([tt.convert_tuple_to_datetime(tt.convert_14_to_tuple(gdac_data['profiles'][p]['datetime'])) 
                            for p in range(gdac_num_profs)])
    gdac_sal_series = array([gt.vert_prof_eval(gdac_data['profiles'][p],'psal',(1600,1700),z_coor='depth',
                                               interp_method='linear',extrap='NaN',avg_method='interp',avg_spacing=1.0,
                                               avg_nan_tolerance=1.0) for p in range(gdac_num_profs)])
    last_good_sal = gdac_sal_series[where(gdac_prof_nums == 83)[0][0]]
    prof_idx_mask_for_trend = logical_and(gdac_prof_nums >= 83, gdac_prof_nums <= 118)
    prof_idx_mask_to_cal = logical_and(gdac_prof_nums >= 84, gdac_prof_nums <= 118)
    
    [cal_slope,cal_intercept] = stats.linregress(mdates.date2num(gdac_time_coord[prof_idx_mask_for_trend]),
                                                 gdac_sal_series[prof_idx_mask_for_trend])[0:2]
    sal_trend = cal_intercept + cal_slope * mdates.date2num(gdac_time_coord)
    sal_deltas = -1 * (sal_trend - sal_trend[where(gdac_prof_nums == 83)[0][0]])
        # note: to not zero the trend at its leftmost y-value, subtract last_good_sal instead (a subjective choice)
    sal_deltas_within_cal_period_only = zeros(gdac_num_profs)
    sal_deltas_within_cal_period_only[prof_idx_mask_to_cal] = sal_deltas[prof_idx_mask_to_cal]
    pickle.dump([gdac_prof_nums,sal_deltas_within_cal_period_only],
                open(argo_index_pickle_dir + 'argo_5904468_cal.pickle','wb'))

# download and/or process SOCCOM data and update pickles
# note: it is preferred to download quarterly SOCCOM snapshots, which have a DOI, rather than the latest data
#       so this offers two options:
#           1) to just process already-downloaded quarterly DOI snapshots, uncomment the ldp.argo_soccom() routine
#           2) to download near-real-time FTP files and process them, uncomment the dlp.argo_soccom() routine
if download_soccom:
    # dlp.argo_soccom(argo_gdac_dir,overwrite_profs=True)   # OPTION 2
    ldp.argo_soccom(soccom_dir)                             # OPTION 1
    argo_soccom_index = ldp.argo_soccom_load_index(soccom_dir,uw_o2_dir,verbose=True)
    pickle.dump(argo_soccom_index,open(argo_index_pickle_dir + 'argo_soccom_index.pickle','wb'))

# download AMSR2 sea ice data and convert from HDF4 to HDF5
if download_amsr2:
    dlp.amsr(2,(2012,7,4),(2019,2,26),amsr2_dir,get_pdfs=False,overwrite=False,convert=True,
             conversion_script_dir=script_dir)

# download AMSR-E sea ice data and convert from HDF4 to HDF5
if download_amsre:
    dlp.amsr(1,(2002,6,1),(2011,10,4),amsre_dir,get_pdfs=False,overwrite=False,convert=True,
             conversion_script_dir=script_dir)

# download NSIDC v1 NRT CDR sea ice data
# note: check FTP link for new data availability; update satellite abbreviations/dates accordingly within functions,
#       including gt.identify_polynyas_magic()
if download_dmsp_nrt:
    dlp.dmsp_nrt((2018,1,1),(2019,2,26),dmsp_nrt_dir,overwrite=True)

# download NSIDC v3 GSFC Merged/CDR sea ice data
# note: check FTP link for new data availability; update satellite abbreviations/dates accordingly within functions,
#       including gt.identify_polynyas_magic()
# note: if a new year of data is released, delete the corresponding NSIDC NRT CDR data
if download_dmsp_v3:
    dlp.dmsp_v3((1978,11,1),(2017,12,31),dmsp_v3_dir,overwrite=False)

# download and unzip Nimbus-5 sea ice data
# note: this requires an EarthData login username and password and a local key; see dlp.nimbus5() for instructions
if download_nimbus5:
    dlp.nimbus5((1972,12,12),(1977,5,11),nimbus5_dir,convert=True,conversion_script_dir=script_dir)

# download station meteorological data from Integrated Surface Database
if download_isd:
    dlp.isd_station(895120,1973,2019,isd_dir,overwrite=True) # Novolazarevskaya
    dlp.isd_station(890010,1973,1994,isd_dir,overwrite=True) # SANAE SAF
    dlp.isd_station(890040,1997,2019,isd_dir,overwrite=True) # SANAE AWS
    dlp.isd_station(890020,1981,2019,isd_dir,overwrite=True) # Neumayer
    dlp.isd_station(895140,1990,2019,isd_dir,overwrite=True) # Maitri

# submit MARS request for ERA-Interim reanalysis fields
# note: submit one at a time; cancel using Ctrl-C (or "stop" button) immediately after seeing "Request is queued"
#       then download using Chrome from: http://apps.ecmwf.int/webmars/joblist/
#       and save using filenames in comments in folder 'ECMWF_Weddell_unprocessed'
#       then run 'process_ecmwf' routine below
if download_ecmwf:
    which_to_download = 1  # change to submit one at a time (see note above) - recommend order 3, 4, 1, 2

    # daily fields
    if which_to_download == 1:     # analysis; save as 'erai_daily_weddell.nc'
        dlp.ecmwf(date_range='1979-01-01/to/2018-12-31',area='-40/-90/-90/90',output_filename=None,type='an',
                  step='0',time='00/06/12/18',params=['msl','sst','skt','t2m','d2m','u10','v10'])
    elif which_to_download == 2:   # forecast; save as 'erai_daily_weddell_forecast.nc'
        dlp.ecmwf(date_range='1979-01-01/to/2018-12-31',area='-40/-90/-90/90',output_filename=None,type='fc',
                  step='6/12',time='00/12',params=['sf','sshf','slhf','ssr','str','strd','e','tp','iews','inss'])
    # monthly means
    elif which_to_download == 3:   # analysis; save as 'erai_monthly_mean_weddell.nc'
        dlp.ecmwf(date_range=[datetime(1979,1,1),datetime(2018,12,1)],area='-40/-90/-90/90',output_filename=None,
                  type='an',step=None,time=None,params=['msl','sp','sst','skt','t2m','u10','v10','si10'])
    elif which_to_download == 4:   # forecast; save as 'erai_monthly_mean_weddell_forecast.nc'
        dlp.ecmwf(date_range=[datetime(1979,1,1),datetime(2018,12,1)],area='-40/-90/-90/90',output_filename=None,
                  type='fc',step=None,time=None,params=['iews','inss'])

# process newly downloaded ECMWF reanalysis files (calculate derived quantities, de-accumulate, and re-export)
# note: once finished, manually delete unprocessed files and any processed chunks
if process_ecmwf:
    for filename in os.listdir(path=era_new_dir):
        if filename == '.DS_Store': continue
        ldp.load_ecmwf(era_new_dir,filename,export_to_dir=era_custom_dir,verbose=True)

# load ERA-Interim land-sea mask
if load_erai_land_sea_mask:
    erai_mask = ldp.load_ecmwf_mask(reanalysis_dir,'erai_land_sea_mask.nc')

###################################### ANALYSIS ROUTINES ######################################################

# Fig. 1. Polynyas of 1974, 2016 and 2017 in relation to profiling float trajectories near Maud Rise.
if plot_fig_1:
    # NOTE: bathymetry contours require manual input to select locations (click, then Return)
    plot_fig_1a = True
    plot_fig_1b = True
    plot_fig_1c = True
    use_fig_1a_pickle = True  # must be False after changing polynya dates for contour (not a big slowdown)

    # establish Antarctic coastline
    circumant_lons,circumant_lats = gt.establish_coastline(coastline_filename_prefix)

    # load sea ice concentration metadata
    [sea_ice_grids,sea_ice_data_avail,sea_ice_all_dates] = ldp.sea_ice_data_prep(nimbus5_dir,dmsp_v3_dir,dmsp_nrt_dir,
                                                                                 amsre_dir,amsr2_dir,amsr_gridfile,
                                                                                 amsr_areafile,nsidc_ps25_grid_dir)

    # plot parameters
    open_sic_plotting = 50  # SIC to be plotted as open
    open_sic_polynya = 50   # SIC threshold for polynyas
    extent_threshold_weddell = 30000  # minimum polynya extent in km^2 to identify/plot
    map_params_weddell = [1700000,1800000,-67.0,0]  # width,height,lat_center,lon_center
    map_params_floats_2016 = [475000,520000,-65.25,4.25]
    map_params_floats_2017 = [465000,520000,-65.25,2.75]
    labelsize = 5

    station_names = ['Novolazarevskaya Station']
    station_locs = [[11.8,-70.8]]  # lon, lat
    station_colors = ['w']
    station_markers = ['^']

    motoi_fuji_loc = [1.2,-66.5]  # lon, lat of R/V Fuji MLS observation on 1974-02-27 from Motoi et al. 1987

    mr_center = [-65.0,3.0]  # summit of Maud Rise (65.0°S, 3.0°E)
    mr_hydro_radius = 250  # search radius (km) from Maud Rise

    mr_bathy_contours = arange(-3250,760,750)
    ocean_color = '#bce6fc'        # light blue
    ocean_color_light = '#ddf2fd'  # lighter version of the above
    polynya_2016_date = (2016,8,6)
    polynya_color_2016 = '#ffae1a'
    polynya_color_2016_light = '#ffce75'
    polynya_color_2016_dark = '#AA7411'
    polynya_line_2016 = '-'
    polynya_2017_date_sic = (2017,11,21)
    polynya_2017_date_string = '2017-11-21'
    polynya_2017_date = (2017,9,25)
    polynya_color_2017 = '#0000cd'
    polynya_color_2017_medium = '#4c4cdc'
    polynya_color_2017_light = '#8484e7'
    polynya_color_2017_dark = '#000089'
    polynya_line_2017 = '-'
    polynya_1974_date = (1974,10,12)
    polynya_color_1974 = 'maroon'
    polynya_line_1974 = '-'
    float_wmoids = [5903616,5904468,5904471]
    float_markers = ['^','o','s']
    float_lines = [(0,(1,1)),'-.','--']
    float_line_colors = ['0.2','k','k']
    float_start_colors = ['lime','lime','lime']
    float_start_sizes = [10,10,10]
    polynya_marker_sizes = [None,25,25]

    toi_5903616 = [20111218000000,20160603000000]
    toi_5904468a = [20150118000000,20170110000000]
    toi_5904468b = [20161231000000,20180509000000]
    toi_5904471a = [20141220000000,20170110000000]
    toi_5904471b = [20161231000000,20180623000000]
    float_polynya_plot_dates = [polynya_2016_date,polynya_2017_date]
    polynya_colors = [polynya_color_2016,polynya_color_2017]

    # establish figure
    if plot_fig_1a or plot_fig_1b or plot_fig_1c:
        fig = plt.figure(figsize=(6.4,2.5))
        subplot_grid_left \
            = gridspec.GridSpec(1,3,width_ratios=[1.1*map_params_weddell[0]*map_params_floats_2016[1]/map_params_weddell[1],
                                                  map_params_floats_2016[0],map_params_floats_2017[0]],wspace=0.20)
        subplot_grid_right \
            = gridspec.GridSpec(1,3,width_ratios=[1.1*map_params_weddell[0]*map_params_floats_2016[1]/map_params_weddell[1],
                                                  map_params_floats_2016[0],map_params_floats_2017[0]],wspace=0.05)

    # 1974, 2016, and 2017 polynyas
    if plot_fig_1a:
        ax1 = plt.subplot(subplot_grid_left[0])

        sic_grid = sea_ice_grids['amsr2']
        sic_field = ldp.load_amsr(sea_ice_data_avail['amsr2'][polynya_2017_date_sic][0],regrid_to_25km=False)
        m1,pcm = pt.sea_ice_argo_spatial(data_dir,polynya_2017_date_sic,sic_grid,sic_field,None,None,None,None,None,
                                         *map_params_weddell,plot_floats=False,polynya_grid=None,open_sic=0,
                                         rasterized=True,as_subplot=True,create_subplot=False,
                                         subplot_add_colorbar=False,
                                         bathy_contours=[],subplot_lon_labels=[0,0,1,0],subplot_lat_labels=[1,0,0,0],
                                         grid_lats=arange(-80,60,5),grid_lons=arange(-80,50,10),
                                         subplot_labelsize=labelsize,grid_color='0.7',
                                         which_ice_cmap=5,cmap_bad_color=ocean_color,cmap_ocean_color=ocean_color,
                                         continent_color='0.7',boundary_width=0.5,coastline_width=0.5,
                                         return_basemap=True,return_pcolor=True)

        sic_grid = sea_ice_grids['nimbus5']
        polynya_grid_1974 = gt.identify_polynyas_magic('nimbus5',polynya_1974_date,sea_ice_grids,sea_ice_data_avail,
                                                       circumant_lons,circumant_lats,open_threshold=open_sic_polynya,
                                                       extent_threshold=extent_threshold_weddell,identify_bad=True)[7]
        m1.contour(*m1(sic_grid['lons'],sic_grid['lats']),polynya_grid_1974,levels=[0.999],
                   colors=polynya_color_1974,linestyles=polynya_line_1974,linewidths=1.0,alpha=0.9,zorder=3)

        sic_grid = sea_ice_grids['amsr2']
        if use_fig_1a_pickle:
            [polynya_grid_2016,polynya_grid_2017] = pickle.load(open(figure_pickle_dir + 'fig_1a','rb'))
        else:
            polynya_grid_2016 = gt.identify_polynyas_magic('amsr2',polynya_2016_date,sea_ice_grids,sea_ice_data_avail,
                                                           circumant_lons,circumant_lats,
                                                           open_threshold=open_sic_polynya,
                                                           extent_threshold=extent_threshold_weddell,identify_bad=True,
                                                           regrid_amsr_to_25km=False)[7]
            polynya_grid_2017 = gt.identify_polynyas_magic('amsr2',polynya_2017_date,sea_ice_grids,sea_ice_data_avail,
                                                           circumant_lons,circumant_lats,
                                                           open_threshold=open_sic_polynya,
                                                           extent_threshold=extent_threshold_weddell,identify_bad=True,
                                                           regrid_amsr_to_25km=False)[7]
            pickle.dump([polynya_grid_2016,polynya_grid_2017],open(figure_pickle_dir + 'fig_1a','wb'))
        m1.contour(*m1(sic_grid['lons'],sic_grid['lats']),polynya_grid_2017,levels=[0.999],
                   colors=polynya_color_2017_dark,linestyles=polynya_line_2017,linewidths=0.5,alpha=0.9,zorder=3)
        m1.contourf(*m1(sic_grid['lons'],sic_grid['lats']),polynya_grid_2017,levels=[0.999,1.1],
                    colors=polynya_color_2017,alpha=0.25,zorder=3)
        m1.contour(*m1(sic_grid['lons'],sic_grid['lats']),polynya_grid_2016,levels=[0.999],
                   colors=polynya_color_2016,linestyles=polynya_line_2016,linewidths=0.5,alpha=0.9,zorder=3)
        m1.contourf(*m1(sic_grid['lons'],sic_grid['lats']),polynya_grid_2016,levels=[0.999,1.1],
                    colors=polynya_color_2016,alpha=0.25,zorder=3)

        m1.scatter(*m1(*station_locs[0]),s=14,c=station_colors[0],marker=station_markers[0],
                   edgecolors='k',linewidths=0.25,zorder=3)
        m1.scatter(*m1(*motoi_fuji_loc),s=17,c='maroon',marker='*',
                   edgecolors='maroon',linewidths=0.25,zorder=3)
        c1, = plt.plot(0,NaN,ls=polynya_line_1974,color=polynya_color_1974,linewidth=1.0,alpha=0.8,
                       label='{0}-{1:02d}-{2:02d}'.format(*polynya_1974_date))  # dummy handles for legend
        c4, = plt.plot([0,0],[NaN,NaN],c='maroon',marker='*',ms=5,mec='maroon',mew=0.25,ls='none',
                       label=r'R/V $\it{Fuji}$, 1974-02-27')
        c5, = plt.plot([0,0],[NaN,NaN],c=station_colors[0],marker=station_markers[0],ms=3,mec='k',mew=0.25,ls='none',
                       label=station_names[0])
        leg1 = ax1.legend(handles=[c4,c1],ncol=2,loc='upper center',bbox_to_anchor=(0.5,-0.010),
                          fontsize=labelsize,columnspacing=1.5,handletextpad=0.6,handlelength=1.5,frameon=False)
        leg2 = mlegend.Legend(ax1,[c5],[station_names[0]],ncol=1,fontsize=labelsize,
                              columnspacing=1.5,handletextpad=0.6,handlelength=1.5,frameon=False)
        leg1._legend_box._children.append(leg2._legend_box._children[1])
        leg1._legend_box.align = 'center'
        m1_ax = plt.gca()
        plt.gca().set_anchor('C')

        plt.text(0.96,0.97,polynya_2017_date_string,color='0.2',size=labelsize+1,fontweight='bold',
                 horizontalalignment='right',verticalalignment='top',transform=ax1.transAxes)

    # 2016 polynya
    if plot_fig_1b:
        ax2 = plt.subplot(subplot_grid_right[1])
        argo_gdac_index = pickle.load(open(argo_index_pickle_dir + 'argo_gdac_index.pickle','rb'))
        float_data_all = []
        for wmoid in float_wmoids:
            this_float_meta = ldp.argo_gdac_float_meta(argo_gdac_index['local_prof_index'],wmoid)
            if wmoid == 5903616: toi_mask = logical_and(this_float_meta['prof_datetimes'] >= toi_5903616[0],
                                                        this_float_meta['prof_datetimes'] <= toi_5903616[1])
            if wmoid == 5904468: toi_mask = logical_and(this_float_meta['prof_datetimes'] >= toi_5904468a[0],
                                                        this_float_meta['prof_datetimes'] <= toi_5904468a[1])
            if wmoid == 5904471: toi_mask = logical_and(this_float_meta['prof_datetimes'] >= toi_5904471a[0],
                                                        this_float_meta['prof_datetimes'] <= toi_5904471a[1])
            float_data_all.append([wmoid,this_float_meta['prof_lons'][toi_mask],this_float_meta['prof_lats'][toi_mask],
                                   this_float_meta['prof_position_flags'][toi_mask],
                                   this_float_meta['prof_datetimes'][toi_mask]])
        _,m2 = pt.bathy_basemap(data_dir,*map_params_floats_2016,create_new_fig=False,
                                labelsize=labelsize,boundary_width=0.5,lon_labels_on_top=True,
                                grid_color='0.4',label_contours=True,cmap='Greys_r',bathy_alpha=0.6,
                                grid_lats=arange(-70,-60,2),grid_lons=arange(-80,50,5))
        for f in range(len(float_data_all)):
            zob = f * 7  # zorder baseline
            mk = float_markers[f]
            fl = float_line_colors[f]
            fls = float_lines[f]
            start_mark = float_markers[f]
            start_color = float_start_colors[f]
            start_size = float_start_sizes[f]
            polynya_marker_size = polynya_marker_sizes[f]

            wmoid = float_data_all[f][0]
            lons = float_data_all[f][1]
            lats = float_data_all[f][2]
            position_flags = float_data_all[f][3]
            datetimes = array([datetime(*tt.convert_14_to_tuple(dt)[0:3]) for dt in float_data_all[f][4]])

            lonx,laty = m2(lons,lats)
            plt.plot(lonx[position_flags != 9],laty[position_flags != 9],color=fl,lw=0.75,ls=fls,zorder=zob+3)
            plt.scatter(lonx[0],laty[0],s=start_size,c=start_color,marker=start_mark,edgecolors='none',zorder=zob+6)

            if wmoid == 5904468 or wmoid == 5904471:
                date_int = tt.convert_tuple_to_8_int(polynya_2016_date) * 1000000
                if sum((float_data_all[f][4] - date_int) == 0) >= 1:
                    polynya_prof_idx = where((float_data_all[f][4] - date_int) == 0)[0][0]
                else:
                    polynya_prof_idx = where((float_data_all[f][4] - date_int) < 0)[0][-1]
                polynya_lon = lons[polynya_prof_idx]
                polynya_lat = lats[polynya_prof_idx]
                plt.scatter(*m2(polynya_lon,polynya_lat),s=polynya_marker_size,c=polynya_color_2016,marker='*',
                            edgecolor='k',linewidths=0.5,zorder=zob+6)

        c1, = plt.plot([0,0],[NaN,NaN],lw=0.75,ls=float_lines[0],c=float_line_colors[0],label='{0}'.format(float_wmoids[0]))
        c2, = plt.plot([0,0],[NaN,NaN],lw=0.75,ls=float_lines[1],c=float_line_colors[1],label='{0}'.format(float_wmoids[1]))
        c3, = plt.plot([0,0],[NaN,NaN],lw=0.75,ls=float_lines[2],c=float_line_colors[2],label='{0}'.format(float_wmoids[2]))
        leg1 = ax2.legend(handles=[c1,c2,c3],ncol=3,loc='upper center',bbox_to_anchor=(1.025,-0.010),
                          fontsize=labelsize,columnspacing=2.5,handletextpad=0.6,handlelength=2.25,frameon=False,
                          markerscale=1,scatterpoints=1)
        c4, = plt.plot([0,0],[NaN,NaN],c=polynya_color_2016_light,ls='-',lw=3,
                       label='{0}-{1:02d}-{2:02d}'.format(*polynya_2016_date),
                       marker='*',ms=5,markerfacecolor=polynya_color_2016,markeredgecolor='k',markeredgewidth=0.5)
        c5, = plt.plot([0,0],[NaN,NaN],c=polynya_color_2017_light,ls='-',lw=3,
                       label='{0}-{1:02d}-{2:02d}'.format(*polynya_2017_date),
                       marker='*',ms=5,markerfacecolor=polynya_color_2017,markeredgecolor='k',markeredgewidth=0.5)
        leg2 = mlegend.Legend(ax2,[c4,c5],['{0}-{1:02d}-{2:02d}'.format(*polynya_2016_date),
                                           '{0}-{1:02d}-{2:02d}'.format(*polynya_2017_date)],ncol=2,fontsize=labelsize,
                              columnspacing=3.0,handletextpad=0.6,handlelength=2.25,frameon=False)
        leg1._legend_box._children.append(leg2._legend_box._children[1])
        leg1._legend_box.align = 'center'

        sic_grid = sea_ice_grids['amsr2']
        sic_field = ldp.load_amsr(sea_ice_data_avail['amsr2'][polynya_2016_date][0],regrid_to_25km=False)
        m2.contour(*m2(sic_grid['lons'],sic_grid['lats']),sic_field,levels=[open_sic_plotting+1],
                   colors=polynya_color_2016_dark,linewidths=0.5,alpha=0.9,zorder=4)
        sic_field = ma.masked_where(sic_field > open_sic_plotting,sic_field)
        m2.contourf(*m2(sic_grid['lons'],sic_grid['lats']),sic_field,levels=[0,open_sic_plotting-1],
                    colors=polynya_color_2016,alpha=0.40,zorder=3)

        plt.text(0.96,0.97,'2011–2016',color='w',size=labelsize+1,fontweight='bold',
                 horizontalalignment='right',verticalalignment='top',transform=ax2.transAxes)

    # 2017 polynya
    if plot_fig_1c:
        ax3 = plt.subplot(subplot_grid_right[2])
        argo_gdac_index = pickle.load(open(argo_index_pickle_dir + 'argo_gdac_index.pickle','rb'))
        float_data_all = []
        for wmoid in float_wmoids:
            this_float_meta = ldp.argo_gdac_float_meta(argo_gdac_index['local_prof_index'],wmoid)
            if wmoid == 5903616: toi_mask = logical_and(this_float_meta['prof_datetimes'] >= toi_5903616[0],
                                                        this_float_meta['prof_datetimes'] <= toi_5903616[1])  # dummy
            if wmoid == 5904468: toi_mask = logical_and(this_float_meta['prof_datetimes'] >= toi_5904468b[0],
                                                        this_float_meta['prof_datetimes'] <= toi_5904468b[1])
            if wmoid == 5904471: toi_mask = logical_and(this_float_meta['prof_datetimes'] >= toi_5904471b[0],
                                                        this_float_meta['prof_datetimes'] <= toi_5904471b[1])
            float_data_all.append([wmoid,this_float_meta['prof_lons'][toi_mask],this_float_meta['prof_lats'][toi_mask],
                                   this_float_meta['prof_position_flags'][toi_mask],
                                   this_float_meta['prof_datetimes'][toi_mask]])
        _,m3 = pt.bathy_basemap(data_dir,*map_params_floats_2017,create_new_fig=False,
                                labelsize=labelsize,boundary_width=0.5,lon_labels_on_top=True,
                                grid_color='0.4',label_contours=False,cmap='Greys_r',bathy_alpha=0.6,
                                grid_lats=arange(-70,-60,2),grid_lons=arange(-80,50,5),force_lat_labels=[0,0,0,0])
        for f in range(len(float_data_all)):
            if f == 0: continue

            zob = f * 7  # zorder baseline
            mk = float_markers[f]
            fl = float_line_colors[f]
            fls = float_lines[f]
            start_mark = float_markers[f]
            start_color = float_start_colors[f]
            start_size = float_start_sizes[f]
            polynya_marker_size = polynya_marker_sizes[f]

            wmoid = float_data_all[f][0]
            lons = float_data_all[f][1]
            lats = float_data_all[f][2]
            position_flags = float_data_all[f][3]
            datetimes = array([datetime(*tt.convert_14_to_tuple(dt)[0:3]) for dt in float_data_all[f][4]])

            lonx,laty = m3(lons,lats)
            plt.plot(lonx[position_flags != 9],laty[position_flags != 9],color=fl,lw=0.75,ls=fls,zorder=zob+3)
            plt.scatter(lonx[0],laty[0],s=start_size-6,c=start_color,marker=start_mark,edgecolors='none',zorder=zob+6)

            if wmoid == 5904468 or wmoid == 5904471:
                date_int = tt.convert_tuple_to_8_int(polynya_2017_date) * 1000000
                if sum((float_data_all[f][4] - date_int) == 0) >= 1:
                    polynya_prof_idx = where((float_data_all[f][4] - date_int) == 0)[0][0]
                else:
                    polynya_prof_idx = where((float_data_all[f][4] - date_int) < 0)[0][-1]
                polynya_lon = lons[polynya_prof_idx]
                polynya_lat = lats[polynya_prof_idx]
                plt.scatter(*m3(polynya_lon,polynya_lat),s=polynya_marker_size,c=polynya_color_2017_medium,marker='*',
                            edgecolor='k',linewidths=0.5,zorder=zob+6)

        sic_grid = sea_ice_grids['amsr2']
        sic_field = ldp.load_amsr(sea_ice_data_avail['amsr2'][polynya_2017_date][0],regrid_to_25km=False)
        m3.contour(*m3(sic_grid['lons'],sic_grid['lats']),sic_field,levels=[open_sic_plotting+1],
                   colors=polynya_color_2017_dark,linewidths=0.5,alpha=0.70,zorder=4)
        sic_field = ma.masked_where(sic_field > open_sic_plotting,sic_field)
        m3.contourf(*m3(sic_grid['lons'],sic_grid['lats']),sic_field,levels=[0,open_sic_plotting-1],
                    colors=polynya_color_2017,alpha=0.30,zorder=3)

        plt.text(0.04,0.97,'2017–2018',color='w',size=labelsize+1,fontweight='bold',
                 horizontalalignment='left',verticalalignment='top',transform=ax3.transAxes)

    # inset box on first subplot
    if plot_fig_1a and plot_fig_1b and plot_fig_1c:
        slon = 3.25  # switchover longitude for right frame representing Fig. 1b and left frame representing Fig. 1c

        # representing Fig. 1b: top, right, bottom
        top_lons,top_lats = m2(linspace(0,map_params_floats_2016[0],100),
                               linspace(map_params_floats_2016[1],map_params_floats_2016[1],100),inverse=True)  # left to right
        right_lons,right_lats = m2(linspace(map_params_floats_2016[0],map_params_floats_2016[0],100),
                                   linspace(map_params_floats_2016[1],0,100),inverse=True)   # downwards
        bottom_lons,bottom_lats = m2(linspace(map_params_floats_2016[0],0,100),
                                     linspace(0,0,100),inverse=True) # right to left
        patch_lons = concatenate((top_lons[top_lons >= slon],right_lons,bottom_lons[bottom_lons >= slon]))
        patch_lats = concatenate((top_lats[top_lons >= slon],right_lats,bottom_lats[bottom_lons >= slon]))
        plonx,platy = m1(patch_lons,patch_lats)
        m1_ax.plot(plonx,platy,c='k',lw=0.5,ls='-',alpha=0.7,zorder=2)

        # representing Fig. 1c: bottom, left, top
        top_lons,top_lats = m3(linspace(0,map_params_floats_2017[0],100),
                               linspace(map_params_floats_2017[1],map_params_floats_2017[1],100),inverse=True)  # left to right
        bottom_lons,bottom_lats = m3(linspace(map_params_floats_2017[0],0,100),
                                     linspace(0,0,100),inverse=True) # right to left
        left_lons,left_lats = m3(linspace(0,0,100),
                                 linspace(0,map_params_floats_2017[1],100),inverse=True)   # upwards
        patch_lons = concatenate((bottom_lons[bottom_lons <= slon],left_lons,top_lons[top_lons <= slon]))
        patch_lats = concatenate((bottom_lats[bottom_lons <= slon],left_lats,top_lats[top_lons <= slon]))
        plonx,platy = m1(patch_lons,patch_lats)
        m1_ax.plot(plonx,platy,c='k',lw=0.5,ls='-',alpha=0.7,zorder=2)

    # add ice colorbar to Fig. 1a
    if plot_fig_1a:
        original_axes = plt.gca()
        cbar_ax = inset_axes(m1_ax,width='100%',height='100%',loc=3,
                             bbox_to_anchor=(0.25,0.0,0.60,0.017),bbox_transform=m1_ax.transAxes)
                                           # left, bottom, width, height
        cbar = plt.gcf().colorbar(pcm,ticks=arange(0,101,20),format='%.0f%%',extend='neither',
                                  orientation='horizontal',cax=cbar_ax)
        cbar.ax.tick_params(labelsize=labelsize,top='on',bottom=False,labeltop='on',labelbottom=False)
        cbar.outline.set_linewidth(0.5)
        plt.sca(original_axes)

    # add inset map to Fig. 1a
    if plot_fig_1a:
        original_axes = plt.gca()
        m4_ax = inset_axes(m1_ax,width='100%',height='100%',
                           bbox_to_anchor=(0.08,0.70,0.32,0.32),bbox_transform=m1_ax.transAxes)
        m4 = Basemap(projection='spstere',boundinglat=-56,lon_0=180,resolution='i',round=True)
        circle = m4.drawmapboundary(linewidth=0.5,fill_color=ocean_color_light)
        circle.set_clip_on(False)
        m4.fillcontinents(color='0.7')
        m4.drawcoastlines(linewidth=0.25,color='k')
        m4.drawparallels(arange(-80,0,10),color='0.6',linewidth=0.25,labels=[0,0,0,0],zorder=2)
        m4.drawmeridians(arange(0,359,30),color='0.6',linewidth=0.25,labels=[0,0,0,0],zorder=2)

        top_lons,top_lats = m1(linspace(0,map_params_weddell[0],100),
                               linspace(map_params_weddell[1],map_params_weddell[1],100),inverse=True)  # left to right
        right_lons,right_lats = m1(linspace(map_params_weddell[0],map_params_weddell[0],100),
                                   linspace(map_params_weddell[1],0,100),inverse=True)   # downwards
        bottom_lons,bottom_lats = m1(linspace(map_params_weddell[0],0,100),
                                     linspace(0,0,100),inverse=True) # right to left
        left_lons,left_lats = m1(linspace(0,0,100),
                                 linspace(0,map_params_weddell[1],100),inverse=True)   # upwards
        patch_lons = concatenate((top_lons,right_lons,bottom_lons,left_lons))
        patch_lats = concatenate((top_lats,right_lats,bottom_lats,left_lats))
        plonx,platy = m4(patch_lons,patch_lats)
        patchxy = list(zip(plonx,platy))
        poly = Polygon(patchxy,linewidth=0.5,linestyle='-',edgecolor='k',facecolor='none',alpha=0.7,zorder=3)
        m4_ax.add_patch(poly)

        plt.sca(original_axes)

    # save figure
    if plot_fig_1a or plot_fig_1b or plot_fig_1c:
        plt.savefig(current_results_dir + 'figure_1.pdf',dpi=450)
        plt.close()

# Fig. 2. Storms, sea ice concentration and mixed-layer salinity at Maud Rise in 2016 and 2017.
# Extended Data Fig. 3. Evolution of sea ice concentration, air temperature and upper ocean properties at Maud Rise in
#                       2016 and 2017.
# Extended Data Fig. 4. Correspondence of sea ice loss episodes and major storms near Maud Rise.
if plot_fig_2_ED_figs_3_4:
    verbose = True
    use_ice_pickle = True
    use_hydro_pickle = True
    use_erai_pickle = True

    mr_box_small = [0,10,-67,-63]  # for reanalysis series
    erai_toi_2016 = [datetime(2016,1,1),datetime(2016,12,31)]
    erai_toi_2017 = [datetime(2017,1,1),datetime(2017,12,31)]

    polynya_datetimes = [[datetime(2016,7,27),datetime(2016,8,17)],[datetime(2017,9,3),datetime(2017,11,28)]]

    mr_obs_dist = 250         # maximum obs distance (in km) from Maud Rise center
    e_weddell_obs_dist = 500  # for comparison
    mr_center = [-65.0,3.0]   # summit of Maud Rise (65.0°S, 3.0°E)
    dpb = 21                  # days per bin for climatological envelope

    si10_crit = 20    # storm criterion (wind speed ≥ 20 m/s)
    msl_crit = 950    # storm criterion (surface pressure ≤ 950 hPa)

    polynya_sats = ['dmsp','amsr2']
    pickle_names = ['fig_2_ice','fig_2_ice_with_amsr_polynya_extent']
    if not use_ice_pickle:
        for run_idx, polynya_sat in enumerate(polynya_sats):
            sic_lon_bounds = [0,10]
            sic_lat_bounds = [-67,-63]
            polynya_lon_bounds = array([-15,20])
            polynya_lat_bounds = array([-68,-62])
            circumant_lons,circumant_lats = gt.establish_coastline(coastline_filename_prefix)
            [sea_ice_grids,sea_ice_data_avail,sea_ice_all_dates] = ldp.sea_ice_data_prep(nimbus5_dir,dmsp_v3_dir,
                                                                                         dmsp_nrt_dir,
                                                                                         amsre_dir,amsr2_dir,amsr_gridfile,
                                                                                         amsr_areafile,nsidc_ps25_grid_dir)
            sic_doy_2016_dmsp = pd.Series()
            sic_doy_2017_dmsp = pd.Series()
            sic_doy_dmsp = dict()
            sic_doy_2016_amsr = pd.Series()
            sic_doy_2017_amsr = pd.Series()
            sic_doy_amsr = dict()
            polynya_extent_doy_2016 = pd.Series()
            polynya_extent_doy_2017 = pd.Series()
            for index, date in enumerate(tt.dates_in_range((1978,11,1),tt.now())):
                if verbose: print(date)
                date_as_datetime = datetime(*date)
                doy = date_as_datetime.timetuple().tm_yday
                # SIC average
                [sic_dmsp,open_area,day_offset] = ldp.sea_ice_concentration(date,sic_lat_bounds,sic_lon_bounds,sea_ice_grids,
                                                                            sea_ice_data_avail,use_only=['dmsp'])
                [sic_amsr,open_area,day_offset] = ldp.sea_ice_concentration(date,sic_lat_bounds,sic_lon_bounds,sea_ice_grids,
                                                                            sea_ice_data_avail,use_only=['amsre','amsr2'])
                if doy not in sic_doy_dmsp: sic_doy_dmsp[doy] = []
                if doy not in sic_doy_amsr: sic_doy_amsr[doy] = []
                sic_doy_dmsp[doy].append(sic_dmsp)
                sic_doy_amsr[doy].append(sic_amsr)
                if date[0] == 2016: sic_doy_2016_dmsp = sic_doy_2016_dmsp.append(pd.Series(index=[doy],data=[sic_dmsp]))
                if date[0] == 2016: sic_doy_2016_amsr = sic_doy_2016_amsr.append(pd.Series(index=[doy],data=[sic_amsr]))
                if date[0] == 2017: sic_doy_2017_dmsp = sic_doy_2017_dmsp.append(pd.Series(index=[doy],data=[sic_dmsp]))
                if date[0] == 2017: sic_doy_2017_amsr = sic_doy_2017_amsr.append(pd.Series(index=[doy],data=[sic_amsr]))
                if date[0] == 2016 or date[0] == 2017:
                    # polynya identification
                    if verbose: print('polynya ID for ',date)
                    sat_string,polynya_string,filename_abbrev,sic_grid,sic_field, \
                    polynya_stats,polynya_grid,polynya_grid_binary,open_ocean_grid,error_code \
                        = gt.identify_polynyas_magic(polynya_sat,date,sea_ice_grids,sea_ice_data_avail,circumant_lons,
                                                     circumant_lats,open_threshold=50,
                                                     extent_threshold=0,regrid_amsr_to_25km=True)
                    # no errors in identifying polynyas
                    if error_code == 0:
                        total_polynya_extent = 0
                        for polynya_index in range(len(polynya_stats)):
                            if polynya_lat_bounds[0] <= polynya_stats[polynya_index]['centroid'][0] <= polynya_lat_bounds[1] \
                                    and polynya_lon_bounds[0] <= polynya_stats[polynya_index]['centroid'][1] <= \
                                            polynya_lon_bounds[1]:
                                total_polynya_extent += polynya_stats[polynya_index]['total_extent']
                        if date[0] == 2016: polynya_extent_doy_2016 \
                            = polynya_extent_doy_2016.append(pd.Series(index=[doy],data=[total_polynya_extent]))
                        if date[0] == 2017: polynya_extent_doy_2017 \
                            = polynya_extent_doy_2017.append(pd.Series(index=[doy],data=[total_polynya_extent]))
                        if verbose: print('>>> polynya extent: ',total_polynya_extent)
                    # fully or partially bad SIC field
                    else:
                        if date[0] == 2016: polynya_extent_doy_2016 \
                            = polynya_extent_doy_2016.append(pd.Series(index=[doy],data=[NaN]))
                        if date[0] == 2017: polynya_extent_doy_2017 \
                            = polynya_extent_doy_2017.append(pd.Series(index=[doy],data=[NaN]))

            for doy in sic_doy_dmsp.keys():
                sic_doy_dmsp[doy] = [nanmean(sic_doy_dmsp[doy]),nanstd(sic_doy_dmsp[doy]),nanmedian(sic_doy_dmsp[doy]),
                                     stats.iqr(sic_doy_dmsp[doy],rng=(25,50),nan_policy='omit'),
                                     stats.iqr(sic_doy_dmsp[doy],rng=(50,75),nan_policy='omit')]
            for doy in sic_doy_amsr.keys():
                sic_doy_amsr[doy] = [nanmean(sic_doy_amsr[doy]),nanstd(sic_doy_amsr[doy]),nanmedian(sic_doy_amsr[doy]),
                                     stats.iqr(sic_doy_amsr[doy],rng=(25,50),nan_policy='omit'),
                                     stats.iqr(sic_doy_amsr[doy],rng=(50,75),nan_policy='omit')]
            pickle.dump([sic_doy_dmsp,sic_doy_amsr,sic_doy_2016_dmsp,sic_doy_2016_amsr,sic_doy_2017_dmsp,sic_doy_2017_amsr,
                         polynya_extent_doy_2016,polynya_extent_doy_2017],
                        open(figure_pickle_dir + pickle_names[run_idx],'wb'))
    sic_doy_dmsp,sic_doy_amsr,sic_doy_2016_dmsp,sic_doy_2016_amsr,sic_doy_2017_dmsp,sic_doy_2017_amsr, \
    polynya_extent_doy_2016,polynya_extent_doy_2017 \
        = pickle.load(open(figure_pickle_dir + pickle_names[0],'rb'))
    _,_,_,_,_,_,polynya_extent_doy_2016_amsr,polynya_extent_doy_2017_amsr \
        = pickle.load(open(figure_pickle_dir + pickle_names[1],'rb'))

    sic_doys = arange(1,366+1)
    sic_doy_dmsp_mean = pd.Series(index=sic_doys,data=[sic_doy_dmsp[doy][0] for doy in sic_doys])
    sic_doy_dmsp_median \
        = pd.Series(index=sic_doys,data=[sic_doy_dmsp[doy][2] for doy in sic_doys]).rolling(window=7,center=True).mean()
    sic_doy_dmsp_iqr_25 \
        = pd.Series(index=sic_doys,data=[sic_doy_dmsp[doy][3] for doy in sic_doys]).rolling(window=7,center=True).mean()
    sic_doy_dmsp_iqr_75 \
        = pd.Series(index=sic_doys,data=[sic_doy_dmsp[doy][4] for doy in sic_doys]).rolling(window=7,center=True).mean()
    sic_doy_amsr_median \
        = pd.Series(index=sic_doys,data=[sic_doy_amsr[doy][2] for doy in sic_doys]).rolling(window=7,center=True).mean()
    sic_doy_amsr_iqr_25 \
        = pd.Series(index=sic_doys,data=[sic_doy_amsr[doy][3] for doy in sic_doys]).rolling(window=7,center=True).mean()
    sic_doy_amsr_iqr_75 \
        = pd.Series(index=sic_doys,data=[sic_doy_amsr[doy][4] for doy in sic_doys]).rolling(window=7,center=True).mean()

    # pickle SIC climatology for other purposes...
    pickle.dump(sic_doy_dmsp_mean,open(figure_pickle_dir + 'fig_2_sic_climatology_dmsp','wb'))

    if use_hydro_pickle:
        mr_obs = pickle.load(open(figure_pickle_dir + 'fig_2_mr_obs','rb'))
        e_weddell_obs = pickle.load(open(figure_pickle_dir + 'fig_2_e_weddell_obs','rb'))
    else:
        mr_obs \
            = ldp.compile_hydrographic_obs(argo_index_pickle_dir,argo_gdac_dir,wod_dir,lon_bounds=[-20,25],
                                           lat_bounds=[-75,-59],toi_bounds=[datetime(1900,1,1),datetime.today()],
                                           distance_check=mr_obs_dist,distance_center=mr_center,
                                           include_argo=True,include_wod=True,params=['ptmp','psal','sigma_theta'],
                                           compute_extras=False,max_cast_min_depth=30,min_cast_max_depth=250,
                                           reject_mld_below=250,interp_spacing=0.1,interp_depths=(0,500),calc_mld=True,
                                           calc_ml_avg=True,calc_at_depths=[20,200,250],
                                           calc_depth_avgs=[(0,200),(0,250),(200,500),(250,500)],
                                           calc_sd=[200,250,(0,250,34.8),(250,1650,34.8)],calc_tb=[200,250],
                                           pickle_dir=figure_pickle_dir,pickle_filename='fig_2_mr_obs',
                                           prof_count_dir=current_results_dir,
                                           prof_count_filename='figure_2_prof_counts_mr',verbose=verbose)
        e_weddell_obs \
            = ldp.compile_hydrographic_obs(argo_index_pickle_dir,argo_gdac_dir,wod_dir,lon_bounds=[-20,25],
                                           lat_bounds=[-75,-59],toi_bounds=[datetime(1900,1,1),datetime.today()],
                                           distance_check=[mr_obs_dist,e_weddell_obs_dist],distance_center=mr_center,
                                           include_argo=True,include_wod=True,params=['ptmp','psal','sigma_theta'],
                                           compute_extras=False,max_cast_min_depth=30,min_cast_max_depth=250,
                                           reject_mld_below=250,interp_spacing=0.1,interp_depths=(0,500),calc_mld=True,
                                           calc_ml_avg=True,calc_at_depths=[20,200,250],
                                           calc_depth_avgs=[(0,200),(0,250),(200,500),(250,500)],
                                           calc_sd=[250,(0,250,34.8)],calc_tb=None,
                                           pickle_dir=figure_pickle_dir,pickle_filename='fig_2_e_weddell_obs',
                                           prof_count_dir=current_results_dir,
                                           prof_count_filename='figure_2_prof_counts_e_weddell',verbose=verbose)

    # calculate and export statistics on compiled hydrography
    text_file = open(current_results_dir + 'figure_2_mr_hydrography_stats.txt','w')
    text_file.write('Statistics on Maud Rise hydrography compiled for MLS, MLT, SD climatologies:\n'
                    '- Number of obs: {0}\n'
                    '- Fraction of obs collected before 1970: {1:.2f}%\n'
                    '- Mean date: {2}\n'
                    '- Median date: {3}\n'
                    .format(len(mr_obs['datetimes']),
                            100 * sum(mr_obs['datetimes'] < datetime(1970,1,1)) / len(mr_obs['datetimes']),
                            datetime.fromtimestamp(mean(array([dt.timestamp() for dt in mr_obs['datetimes']]))),
                            datetime.fromtimestamp(median(array([dt.timestamp() for dt in mr_obs['datetimes']])))))
    text_file.close()
    text_file = open(current_results_dir + 'figure_2_ew_hydrography_stats.txt','w')
    text_file.write('Statistics on Eastern Weddell hydrography compiled for MLS, MLT, SD climatologies:\n'
                    '- Number of obs: {0}\n'
                    '- Fraction of obs collected before 1970: {1:.2f}%\n'
                    '- Mean date: {2}\n'
                    '- Median date: {3}\n'
                    .format(len(e_weddell_obs['datetimes']),
                            100 * sum(e_weddell_obs['datetimes'] < datetime(1970,1,1)) / len(e_weddell_obs['datetimes']),
                            datetime.fromtimestamp(mean(array([dt.timestamp() for dt in e_weddell_obs['datetimes']]))),
                            datetime.fromtimestamp(median(array([dt.timestamp() for dt in e_weddell_obs['datetimes']])))))
    text_file.close()

    # load reanalysis data and create time series of interest
    if use_erai_pickle:
        erai_series = pickle.load(open(figure_pickle_dir + 'fig_2_erai_series','rb'))
    else:
        erai_daily = ldp.load_ecmwf(era_custom_dir,'erai_daily_weddell.nc')
        erai_daily_toi_2016 = erai_daily.sel(time=slice(erai_toi_2016[0],erai_toi_2016[1]))
        erai_daily_toi_2017 = erai_daily.sel(time=slice(erai_toi_2017[0],erai_toi_2017[1]))
        erai_daily_toi_2016_2017 = erai_daily.sel(time=slice(erai_toi_2016[0],erai_toi_2017[1]))
        erai_series = dict()
        erai_series['msl_min_2016'] \
            = ldp.create_reanalysis_index(erai_daily_toi_2016,param_name='msl',avg_box=mr_box_small,
                                          min_not_mean=True)[0]
        erai_series['msl_min_2017'] \
            = ldp.create_reanalysis_index(erai_daily_toi_2017,param_name='msl',avg_box=mr_box_small,
                                          min_not_mean=True)[0]
        erai_series['msl_min_climo_2016'], erai_series['msl_min_climo_iqr_25_2016'], erai_series['msl_min_climo_iqr_75_2016'] \
            = ldp.create_reanalysis_index(erai_daily,param_name='msl',avg_box=mr_box_small,
                                          min_not_mean=True,create_climo_iqr=True,make_year=2016)
        erai_series['msl_min_climo_2017'], erai_series['msl_min_climo_iqr_25_2017'], erai_series['msl_min_climo_iqr_75_2017'] \
            = ldp.create_reanalysis_index(erai_daily,param_name='msl',avg_box=mr_box_small,
                                          min_not_mean=True,create_climo_iqr=True,make_year=2017)
        erai_series['si10_max_2016'] \
            = ldp.create_reanalysis_index(erai_daily_toi_2016,param_name='si10',avg_box=mr_box_small,
                                          max_not_mean=True)[0]
        erai_series['si10_max_2017'] \
            = ldp.create_reanalysis_index(erai_daily_toi_2017,param_name='si10',avg_box=mr_box_small,
                                          max_not_mean=True)[0]
        erai_series['si10_max_climo_2016'], erai_series['si10_max_climo_iqr_25_2016'], erai_series['si10_max_climo_iqr_75_2016'] \
            = ldp.create_reanalysis_index(erai_daily,param_name='si10',avg_box=mr_box_small,
                                          max_not_mean=True,create_climo_iqr=True,make_year=2016)
        erai_series['si10_max_climo_2017'], erai_series['si10_max_climo_iqr_25_2017'], erai_series['si10_max_climo_iqr_75_2017'] \
            = ldp.create_reanalysis_index(erai_daily,param_name='si10',avg_box=mr_box_small,
                                          max_not_mean=True,create_climo_iqr=True,make_year=2017)
        erai_series['t2m_2016'] \
            = ldp.create_reanalysis_index(erai_daily_toi_2016,param_name='t2m',avg_box=mr_box_small)[0]
        erai_series['t2m_2017'] \
            = ldp.create_reanalysis_index(erai_daily_toi_2017,param_name='t2m',avg_box=mr_box_small)[0]
        erai_series['t2m_2016_2017'] \
            = ldp.create_reanalysis_index(erai_daily_toi_2016_2017,param_name='t2m',avg_box=mr_box_small)[0]
        erai_series['t2m_climo_2016'], erai_series['t2m_climo_iqr_25_2016'], erai_series['t2m_climo_iqr_75_2016'] \
            = ldp.create_reanalysis_index(erai_daily,param_name='t2m',avg_box=mr_box_small,
                                          create_climo_iqr=True,make_year=2016)
        erai_series['t2m_climo_2017'], erai_series['t2m_climo_iqr_25_2017'], erai_series['t2m_climo_iqr_75_2017'] \
            = ldp.create_reanalysis_index(erai_daily,param_name='t2m',avg_box=mr_box_small,
                                          create_climo_iqr=True,make_year=2017)
        pickle.dump(erai_series,open(figure_pickle_dir + 'fig_2_erai_series','wb'))
        erai_daily_toi_2016.close()
        erai_daily_toi_2017.close()
        erai_daily.close()

    ##################################################################################################################
    ###################################   FIG. 2 PLOTTING ROUTINE   ##################################################
    ##################################################################################################################

    fontsize = 5
    shade_color='k'
    shade_alpha=0.15
    shade_alpha_dark=0.25
    shade_centerline_alpha=0.5
    ew_shade_color='saddlebrown'
    ew_shade_centerline_alpha=0.5
    color_accent='k'
    color_polynya='navy'

    lw=0.5
    ms=1.5
    mew=0.1   # marker edge width

    # create multiyear series that repeats a single-year climatology
    def plot_doy_series(series,ref_datetime=datetime(2013,12,31),N_years=5,smooth_N_days_betw_yrs=7,
                        just_return_series=False):
        if N_years == 1: smooth_N_days_betw_yrs = 0
        full_index = array([])
        full_data = array([])
        for y in range(N_years):
            new_index = array([timedelta(days=int(doy)) for doy in series.index]) \
                        + ref_datetime.replace(year=ref_datetime.year+y)
            single_year_mask = array([dt.year == ref_datetime.year+1+y for dt in new_index])
            full_index = append(full_index,new_index[single_year_mask])
            if smooth_N_days_betw_yrs != 0: full_data[-1*smooth_N_days_betw_yrs:] = NaN
            new_data = series.values[single_year_mask]
            if smooth_N_days_betw_yrs != 0: new_data[:smooth_N_days_betw_yrs] = NaN
            full_data = append(full_data,new_data)
        full_series = pd.Series(index=full_index,data=full_data).interpolate(method='linear').bfill().ffill()
        if just_return_series: return full_series
        return full_series.index,full_series.values

    def series_change_year(series,new_year):
        new_index = array([dt.replace(year=new_year) for dt in series.index])
        return pd.Series(index=new_index,data=series.values)

    def series_daily_interp(orig_series,new_index_toi=[datetime(2016,1,1),datetime(2018,1,1)]):
        interpolator = spin.interp1d(tt.datetime_to_datenum(orig_series.index),orig_series.values,
                                     bounds_error=False,fill_value=NaN)
        new_index = arange(*new_index_toi,timedelta(days=1))
        interp_data = interpolator(tt.datetime_to_datenum(new_index))
        return pd.Series(index=new_index,data=interp_data)

    def fig_2_axis_prep(ylabel='',xlabel_top=False,xlabel_bottom=False,add_xtick_top=False,add_xtick_bottom=False,
                        spines=[],ylabel_pos='left',ylabel_color='k',remove_yticks=[],
                        xgrid=False,ygrid=False,ylabel_offset=0.0,ylabel_pad=0.0,
                        xbounds=[datetime(2016,1,1),datetime(2017,12,31)]):
        plt.xlim(xbounds)
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        if add_xtick_top: xtick_top = True
        else:             xtick_top = xlabel_top
        if add_xtick_bottom: xtick_bottom = True
        else:                xtick_bottom = xlabel_bottom
        plt.tick_params(axis='x',which='both',bottom=xtick_bottom,top=xtick_top,
                        labeltop=xlabel_top,labelbottom=xlabel_bottom)
        if xgrid: plt.gca().grid(which='major',axis='x',linewidth=0.5,alpha=0.3)
        if ygrid: plt.gca().grid(which='major',axis='y',linewidth=0.5,alpha=0.3)
        plt.gca().tick_params(axis='both',which='major',labelsize=fontsize)
        [plt.gca().spines[side].set_linewidth(0.25) for side in plt.gca().spines.keys()]
        plt.gca().tick_params(width=0.25)
        if 'top' not in spines: plt.gca().spines['top'].set_visible(False)
        if 'bottom' not in spines: plt.gca().spines['bottom'].set_visible(False)
        for tick in remove_yticks:
            plt.gca().get_yticklabels()[tick].set_visible(False)
            plt.gca().yaxis.get_major_ticks()[tick].set_visible(False)
        if ylabel_pos == 'right':
            plt.gca().yaxis.tick_right()
            plt.gca().yaxis.set_label_position('right')
        plt.gca().patch.set_alpha(0.0)
        if ylabel_pos == 'left':  ylabel_rot =  90
        if ylabel_pos == 'right': ylabel_rot =  90  # previously -90
        plt.ylabel(ylabel,fontsize=fontsize,rotation=ylabel_rot,color=ylabel_color)
        if ylabel_pos == 'left':  plt.gca().get_yaxis().set_label_coords(-0.07-ylabel_pad,0.5+ylabel_offset)
        if ylabel_pos == 'right': plt.gca().get_yaxis().set_label_coords( 1.09+ylabel_pad,0.5+ylabel_offset)
        plt.setp(plt.gca().get_yticklabels(),color=ylabel_color)

    def fig_2_plot_sic(plot_2016=True,plot_2017=True,plot_legend=True,legend_bbox=(0.003,1.00),
                       xbounds=[datetime(2016,1,1),datetime(2017,12,31)]):
        factor = 1.7
        sic_doy_dmsp_low = sic_doy_dmsp_median-sic_doy_dmsp_iqr_25
        sic_doy_dmsp_low[sic_doy_dmsp_low < 0.0] = 0.0
        plt.fill_between(*plot_doy_series(sic_doy_dmsp_low**factor),
                         plot_doy_series(sic_doy_dmsp_median+sic_doy_dmsp_iqr_75)[1]**factor,
                         facecolor=shade_color,alpha=shade_alpha,zorder=4)
        plt.plot(*plot_doy_series(sic_doy_dmsp_median**factor),
                 c='k',linewidth=0.5,alpha=shade_centerline_alpha,zorder=5)
        # han1 = plt.fill_between([0,1],[nan,nan],facecolor=shade_color,alpha=shade_alpha,
        #                         label='Maud Rise') # dummy handle  # (±1$\sigma$)
        han2, = plt.plot([0,1],[nan,nan],c='k',lw=lw,ls='-',label='NSIDC Merged')  # dummy handle
        han3, = plt.plot([0,1],[nan,nan],c='k',lw=lw,ls=':',label='AMSR2-ASI')  # dummy handle
        if plot_2016:
            # han4, = plt.plot([0,1],[nan,nan],c=color_accent,lw=1.0,ls='-',label='2016')  # dummy handle
            plt.plot(*plot_doy_series(sic_doy_2016_dmsp**factor,N_years=1,ref_datetime=datetime(2015,12,31)),
                     c=color_accent,lw=lw,ls='-',zorder=6)
            plt.plot(*plot_doy_series(sic_doy_2016_amsr**factor,N_years=1,ref_datetime=datetime(2015,12,31)),
                     c=color_accent,lw=lw,ls=':',zorder=7)
        if plot_2017:
            # han5, = plt.plot([0,1],[nan,nan],c=color_accent,lw=1.0,ls='-',label='2017')  # dummy handle
            plt.plot(*plot_doy_series(sic_doy_2017_dmsp**factor,N_years=1,ref_datetime=datetime(2016,12,31)),
                     c=color_accent,lw=lw,ls='-',zorder=6)
            plt.plot(*plot_doy_series(sic_doy_2017_amsr**factor,N_years=1,ref_datetime=datetime(2016,12,31)),
                     c=color_accent,lw=lw,ls=':',zorder=7)
        # plt.plot([0,1],[nan,nan],c='k',lw=lw,ls='-',label=' ',visible=False)
        plt.plot([datetime(2016,1,1),datetime(2017,12,31)],[0,0],c='k',linestyle='-',linewidth=0.5,zorder=15)
        plt.ylim([-0.01,1.0])
        plt.gca().set_yticks(array([0,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])**factor)
        plt.gca().set_yticklabels(['','30','40','50','60','70','80','90','100'])
        if plot_legend:
            plt.legend(handles=[han2,han3],bbox_to_anchor=legend_bbox,handlelength=1.75,loc='upper left',
                       ncol=1,frameon=False,fontsize=fontsize - 1)  # labelspacing=1.075
        fig_2_axis_prep(ylabel='Sea ice\nconcentration (%)',spines=['top'],xbounds=xbounds,add_xtick_bottom=True)

    def find_storms(si10_series,msl_series,si10_crit=si10_crit,msl_crit=msl_crit,after=datetime(2016,5,1)):
        storm_dates_all = []
        for date in si10_series.index:
            if date < after: continue
            if si10_series[date] >= si10_crit or msl_series[date] <= msl_crit:
                storm_dates_all.append(date)
        return storm_dates_all

    # NOTE: storms identified using find_storms(), then dates are manually aggregated (if applicable) and listed below:
    storm_datetimes_2016 = [[datetime(2016,5,20,0),datetime(2016,5,26,0)],
                            [datetime(2016,6,12,0),datetime(2016,6,15,0)],
                            [datetime(2016,6,30,0),datetime(2016,7,1,0)],
                            [datetime(2016,7,9,0),datetime(2016,7,13,0)],
                            [datetime(2016,7,26,0),datetime(2016,7,28,0)],
                            [datetime(2016,8,2,0),datetime(2016,8,6,0)],
                            [datetime(2016,8,21,0),datetime(2016,8,23,0)],
                            [datetime(2016,8,28,0),datetime(2016,8,31,0)],
                            [datetime(2016,10,17,0),datetime(2016,10,19,0)],
                            [datetime(2016,10,26,0),datetime(2016,10,28,0)]]

    storm_datetimes_2017 = [[datetime(2017,7,3,0),datetime(2017,7,8,0)],
                            [datetime(2017,7,23,0),datetime(2017,7,27,0)],
                            [datetime(2017,7,30,0),datetime(2017,8,3,0)],
                            [datetime(2017,8,7,0),datetime(2017,8,8,0)],
                            [datetime(2017,8,14,0),datetime(2017,8,15,0)],
                            [datetime(2017,8,18,0),datetime(2017,8,19,0)],
                            [datetime(2017,8,30,0),datetime(2017,9,2,0)],
                            [datetime(2017,9,13,0),datetime(2017,9,19,0)],
                            [datetime(2017,11,4,0),datetime(2017,11,5,0)],
                            [datetime(2017,11,30,0),datetime(2017,12,1,0)]]

    # extract hydrographic quantities
    mr_obs_mask_5903616 = mr_obs['platforms'] == str(5903616)
    mr_obs_mask_5904471 = mr_obs['platforms'] == str(5904471)
    mr_obs_mask_5904468 = mr_obs['platforms'] == str(5904468)

    ml_avg_ptmp_median \
        = ldp.hydro_obs_to_doy_series(mr_obs,'ptmp','ml_avg',doy_median=True,days_per_bin=dpb,rm_days=dpb)
    ml_avg_ptmp_iqr_25 \
        = ldp.hydro_obs_to_doy_series(mr_obs,'ptmp','ml_avg',doy_iqr_25=True,days_per_bin=dpb,rm_days=dpb)
    ml_avg_ptmp_iqr_75 \
        = ldp.hydro_obs_to_doy_series(mr_obs,'ptmp','ml_avg',doy_iqr_75=True,days_per_bin=dpb,rm_days=dpb)
    ew_ml_avg_ptmp_median \
        = ldp.hydro_obs_to_doy_series(e_weddell_obs,'ptmp','ml_avg',doy_median=True,days_per_bin=dpb,rm_days=dpb)
    ew_ml_avg_ptmp_iqr_25 \
        = ldp.hydro_obs_to_doy_series(e_weddell_obs,'ptmp','ml_avg',doy_iqr_25=True,days_per_bin=dpb,rm_days=dpb)
    ew_ml_avg_ptmp_iqr_75 \
        = ldp.hydro_obs_to_doy_series(e_weddell_obs,'ptmp','ml_avg',doy_iqr_75=True,days_per_bin=dpb,rm_days=dpb)
    ml_avg_ptmp_5903616 = pd.Series(index=mr_obs['datetimes'][mr_obs_mask_5903616],
                                    data=mr_obs['ptmp']['ml_avg'][mr_obs_mask_5903616])
    ml_avg_ptmp_5904471 = pd.Series(index=mr_obs['datetimes'][mr_obs_mask_5904471],
                                    data=mr_obs['ptmp']['ml_avg'][mr_obs_mask_5904471])
    ml_avg_ptmp_5904468 = pd.Series(index=mr_obs['datetimes'][mr_obs_mask_5904468],
                                    data=mr_obs['ptmp']['ml_avg'][mr_obs_mask_5904468])
    avg_ml_avg_ptmp = pd.concat([series_daily_interp(ml_avg_ptmp_5903616),
                                 series_daily_interp(ml_avg_ptmp_5904471),
                                 series_daily_interp(ml_avg_ptmp_5904468)],axis=1).mean(axis=1)
    min_ml_avg_ptmp = pd.concat([series_daily_interp(ml_avg_ptmp_5903616),
                                 series_daily_interp(ml_avg_ptmp_5904471),
                                 series_daily_interp(ml_avg_ptmp_5904468)],axis=1).min(axis=1)
    max_ml_avg_ptmp = pd.concat([series_daily_interp(ml_avg_ptmp_5903616),
                                 series_daily_interp(ml_avg_ptmp_5904471),
                                 series_daily_interp(ml_avg_ptmp_5904468)],axis=1).max(axis=1)
    ml_avg_psal_median \
        = ldp.hydro_obs_to_doy_series(mr_obs,'psal','ml_avg',doy_median=True,days_per_bin=dpb,rm_days=dpb)
    ml_avg_psal_iqr_25 \
        = ldp.hydro_obs_to_doy_series(mr_obs,'psal','ml_avg',doy_iqr_25=True,days_per_bin=dpb,rm_days=dpb)
    ml_avg_psal_iqr_75 \
        = ldp.hydro_obs_to_doy_series(mr_obs,'psal','ml_avg',doy_iqr_75=True,days_per_bin=dpb,rm_days=dpb)
    ew_ml_avg_psal_median \
        = ldp.hydro_obs_to_doy_series(e_weddell_obs,'psal','ml_avg',doy_median=True,days_per_bin=dpb,rm_days=dpb)
    ew_ml_avg_psal_iqr_25 \
        = ldp.hydro_obs_to_doy_series(e_weddell_obs,'psal','ml_avg',doy_iqr_25=True,days_per_bin=dpb,rm_days=dpb)
    ew_ml_avg_psal_iqr_75 \
        = ldp.hydro_obs_to_doy_series(e_weddell_obs,'psal','ml_avg',doy_iqr_75=True,days_per_bin=dpb,rm_days=dpb)
    ml_avg_psal_5903616 = pd.Series(index=mr_obs['datetimes'][mr_obs_mask_5903616],
                                    data=mr_obs['psal']['ml_avg'][mr_obs_mask_5903616])
    ml_avg_psal_5904471 = pd.Series(index=mr_obs['datetimes'][mr_obs_mask_5904471],
                                    data=mr_obs['psal']['ml_avg'][mr_obs_mask_5904471])
    ml_avg_psal_5904468 = pd.Series(index=mr_obs['datetimes'][mr_obs_mask_5904468],
                                    data=mr_obs['psal']['ml_avg'][mr_obs_mask_5904468])
    avg_ml_avg_psal = pd.concat([series_daily_interp(ml_avg_psal_5903616),
                                 series_daily_interp(ml_avg_psal_5904471),
                                 series_daily_interp(ml_avg_psal_5904468)],axis=1).mean(axis=1)
    min_ml_avg_psal = pd.concat([series_daily_interp(ml_avg_psal_5903616),
                                 series_daily_interp(ml_avg_psal_5904471),
                                 series_daily_interp(ml_avg_psal_5904468)],axis=1).min(axis=1)
    max_ml_avg_psal = pd.concat([series_daily_interp(ml_avg_psal_5903616),
                                 series_daily_interp(ml_avg_psal_5904471),
                                 series_daily_interp(ml_avg_psal_5904468)],axis=1).max(axis=1)
    ldens_5903616 = pd.Series(index=mr_obs['datetimes'][mr_obs_mask_5903616],
                              data=mr_obs['sigma_theta'][250][mr_obs_mask_5903616])
    ldens_5904471 = pd.Series(index=mr_obs['datetimes'][mr_obs_mask_5904471],
                              data=mr_obs['sigma_theta'][250][mr_obs_mask_5904471])
    ldens_5904468 = pd.Series(index=mr_obs['datetimes'][mr_obs_mask_5904468],
                              data=mr_obs['sigma_theta'][250][mr_obs_mask_5904468])
    avg_ldens = pd.concat([series_daily_interp(ldens_5903616),
                           series_daily_interp(ldens_5904471),
                           series_daily_interp(ldens_5904468)],axis=1).mean(axis=1)
    mr_mld_median = ldp.hydro_obs_to_doy_series(mr_obs,None,'mlds',doy_median=True,days_per_bin=dpb,rm_days=dpb)
    mr_mld_iqr_25 = ldp.hydro_obs_to_doy_series(mr_obs,None,'mlds',doy_iqr_25=True,days_per_bin=dpb,rm_days=dpb)
    mr_mld_iqr_75 = ldp.hydro_obs_to_doy_series(mr_obs,None,'mlds',doy_iqr_75=True,days_per_bin=dpb,rm_days=dpb)
    ew_mld_median = ldp.hydro_obs_to_doy_series(e_weddell_obs,None,'mlds',doy_median=True,days_per_bin=dpb,rm_days=dpb)
    ew_mld_iqr_25 = ldp.hydro_obs_to_doy_series(e_weddell_obs,None,'mlds',doy_iqr_25=True,days_per_bin=dpb,rm_days=dpb)
    ew_mld_iqr_75 = ldp.hydro_obs_to_doy_series(e_weddell_obs,None,'mlds',doy_iqr_75=True,days_per_bin=dpb,rm_days=dpb)
    mld_5903616 = pd.Series(index=mr_obs['datetimes'][mr_obs_mask_5903616],
                            data=mr_obs['mlds'][mr_obs_mask_5903616])
    mld_5904471 = pd.Series(index=mr_obs['datetimes'][mr_obs_mask_5904471],
                            data=mr_obs['mlds'][mr_obs_mask_5904471])
    mld_5904468 = pd.Series(index=mr_obs['datetimes'][mr_obs_mask_5904468],
                            data=mr_obs['mlds'][mr_obs_mask_5904468])
    avg_mld = pd.concat([series_daily_interp(mld_5903616),
                         series_daily_interp(mld_5904471),
                         series_daily_interp(mld_5904468)],axis=1).mean(axis=1)
    min_mld = pd.concat([series_daily_interp(mld_5903616),
                         series_daily_interp(mld_5904471),
                         series_daily_interp(mld_5904468)],axis=1).min(axis=1)
    max_mld = pd.concat([series_daily_interp(mld_5903616),
                         series_daily_interp(mld_5904471),
                         series_daily_interp(mld_5904468)],axis=1).max(axis=1)

    # calculate ML salinity needed to reduce ML-average-sigma_theta to its value at 250 m (averaged betw. 2 floats)
    ml_psal_for_destab = []
    for dt_idx in range(len(avg_ml_avg_psal.index)):
        destab_psal = NaN
        for test_psal in arange(avg_ml_avg_psal[dt_idx],38.0,0.001):
            ml_asal = gsw.SA_from_SP(test_psal,0.0,0.0,-65.0)  # lon = 0°E, lat = 65°S (approx.)
            ml_ctmp = gsw.CT_from_pt(ml_asal,avg_ml_avg_ptmp[dt_idx])
            if gsw.sigma0(ml_asal,ml_ctmp) >= avg_ldens[dt_idx]:
                destab_psal = test_psal
                break
        ml_psal_for_destab.append(destab_psal)
    ml_psal_for_destab = pd.Series(data=ml_psal_for_destab,index=avg_ml_avg_psal.index)

    fig = plt.figure(figsize=(6,5))

    fig.add_axes([0.15,0.8,0.7,0.15]) # [x0, y0, width, height] for lower left point (from bottom left of figure)
    fig_2_plot_sic(plot_2017=True)

    for storm_datetime in storm_datetimes_2016 + storm_datetimes_2017:
        plt.gca().axvspan(xmin=storm_datetime[0]-timedelta(days=1),xmax=storm_datetime[1],ymin=1,ymax=1.09,
                          facecolor='k',alpha=1.0,zorder=1,clip_on=False)
    plt.text(0.16,1.01,'Winter storms:',horizontalalignment='right',verticalalignment='bottom',
             transform=plt.gca().transAxes,color='k',fontsize=fontsize-1)

    right_axis = plt.gca().twinx()
    plt.sca(right_axis)
    polynya_extent_doy_2016[polynya_extent_doy_2016 > 150000] = 150000
    polynya_extent_doy_2017[polynya_extent_doy_2017 > 150000] = 150000
    polynya_extent_doy_2016_amsr[polynya_extent_doy_2016_amsr > 150000] = 150000
    polynya_extent_doy_2017_amsr[polynya_extent_doy_2017_amsr > 150000] = 150000
    polynya_doys_2016 = pd.Index(polynya_datetimes[0]).dayofyear
    polynya_doys_2017 = pd.Index(polynya_datetimes[1]).dayofyear
    polynya_extent_doy_2016 \
        = polynya_extent_doy_2016.truncate(before=polynya_doys_2016[0],after=polynya_doys_2016[1])
    polynya_extent_doy_2017 \
        = polynya_extent_doy_2017.truncate(before=polynya_doys_2017[0],after=polynya_doys_2017[1]-1)
    polynya_extent_doy_2016_amsr \
        = polynya_extent_doy_2016_amsr.truncate(before=polynya_doys_2016[0],after=polynya_doys_2016[1])
    polynya_extent_doy_2017_amsr \
        = polynya_extent_doy_2017_amsr.truncate(before=polynya_doys_2017[0],after=polynya_doys_2017[1]-1)
    plt.plot(*plot_doy_series(polynya_extent_doy_2016,N_years=1,ref_datetime=datetime(2015,12,31)),
             c=color_polynya,alpha=0.75,linewidth=lw,zorder=1)
    plt.plot(*plot_doy_series(polynya_extent_doy_2017,N_years=1,ref_datetime=datetime(2016,12,31)),
             c=color_polynya,alpha=0.75,linewidth=lw,zorder=1)
    plt.plot(*plot_doy_series(polynya_extent_doy_2016_amsr,N_years=1,ref_datetime=datetime(2015,12,31)),
             ls=':',c=color_polynya,alpha=0.75,linewidth=lw,zorder=1)
    plt.plot(*plot_doy_series(polynya_extent_doy_2017_amsr,N_years=1,ref_datetime=datetime(2016,12,31)),
             ls=':',c=color_polynya,alpha=0.75,linewidth=lw,zorder=1)
    plt.ylim([-2000,175000])
    plt.gca().set_yticks([0,25000,50000,75000,100000,125000,150000])
    plt.gca().set_yticklabels([0,'','50','','100','','≥150'])
    fig_2_axis_prep('Polynya extent\n' + r'(10$^3$ km$^2$)',spines=['top'],ylabel_pos='right',
                    ylabel_color=color_polynya,ylabel_offset=-0.05,ylabel_pad=-0.01)

    fig.add_axes([0.15,0.60,0.7,0.20])
    plt.plot(*plot_doy_series(ew_ml_avg_psal_median-ew_ml_avg_psal_iqr_25),
             ls=':',c=ew_shade_color,linewidth=lw,alpha=ew_shade_centerline_alpha,zorder=1)
    h1, = plt.plot(*plot_doy_series(ew_ml_avg_psal_median),
                   ls='--',c=ew_shade_color,linewidth=lw,alpha=ew_shade_centerline_alpha,zorder=1,
                   label='Eastern Weddell')
    plt.plot(*plot_doy_series(ew_ml_avg_psal_median+ew_ml_avg_psal_iqr_75),
             ls=':',c=ew_shade_color,linewidth=lw,alpha=ew_shade_centerline_alpha,zorder=1)
    h2 = plt.fill_between(*plot_doy_series(ml_avg_psal_median-ml_avg_psal_iqr_25),
                          plot_doy_series(ml_avg_psal_median+ml_avg_psal_iqr_75)[1],
                          facecolor=shade_color,alpha=shade_alpha,zorder=2,
                          label='Maud Rise')
    plt.plot(*plot_doy_series(ml_avg_psal_median),
             c=shade_color,linewidth=lw,alpha=shade_centerline_alpha,zorder=2)
    plt.plot(max_ml_avg_psal,c=color_accent,linewidth=lw,zorder=3)
    h3, = plt.plot(ml_psal_for_destab,c='maroon',linestyle='--',linewidth=lw,zorder=3,
                   label='Overturning limit') # previously c='maroon'
    h4a, = plt.plot(datetime(2016,2,27),34.287,markeredgewidth=0,marker='*',ms=4,markerfacecolor='maroon',
                    ls='None',zorder=4,label=r'R/V $\it{Fuji}$, 1974-02-27')
    h4b, = plt.plot(datetime(2017,2,27),34.287,markeredgewidth=0,marker='*',ms=4,markerfacecolor='maroon',
                    ls='None',zorder=4)
    plt.ylim([33.60,34.90])
    plt.gca().set_yticks([33.9,34.1,34.3,34.5,34.7])
    fig_2_axis_prep('Mixed-layer salinity\n(psu)',ylabel_pos='left',ylabel_offset=0.00,
                    spines=['bottom'],xlabel_bottom=True)
    leg = plt.legend(handles=[h3,h2,h1,h4a],bbox_to_anchor=(0.003,-0.02),loc='lower left',ncol=4,frameon=False,
                     fontsize=fontsize-1,handlelength=1.75,handletextpad=0.6)
    leg.texts[3].set_position((0,-1.0))

    # print statistics for July MLS
    for day in range(1,32):
        mls_elev = max_ml_avg_psal.loc[datetime(2016,7,day)] \
                   - plot_doy_series(ml_avg_psal_median,N_years=1,ref_datetime=datetime(2015,12,31),
                                     just_return_series=True).loc[datetime(2016,7,day)]
        print('Max MLS elevation on July {0}, 2016: {1:.2f} psu'.format(day,mls_elev))

    mls_elev = avg_ml_avg_psal - plot_doy_series(ml_avg_psal_median,N_years=1,ref_datetime=datetime(2015,12,31),
                                                 just_return_series=True)
    print('Average and standard deviation of average MLS elevation from January-May 2016: {0:.2f} ± {1:.2f} psu'
          .format(mls_elev[datetime(2016,1,1):datetime(2016,5,31)].mean(),
                  mls_elev[datetime(2016,1,1):datetime(2016,5,31)].std()))
    print('Average and standard deviation of average MLS elevation from January-June 2016: {0:.2f} ± {1:.2f} psu'
          .format(mls_elev[datetime(2016,1,1):datetime(2016,6,30)].mean(),
                  mls_elev[datetime(2016,1,1):datetime(2016,6,30)].std()))

    mls_elev = avg_ml_avg_psal - plot_doy_series(ml_avg_psal_median,N_years=1,ref_datetime=datetime(2016,12,31),
                                                 just_return_series=True)
    print('Average and standard deviation of average MLS elevation from January-May 2017: {0:.2f} ± {1:.2f} psu'
          .format(mls_elev[datetime(2017,1,1):datetime(2017,5,31)].mean(),
                  mls_elev[datetime(2017,1,1):datetime(2017,5,31)].std()))
    print('Average and standard deviation of average MLS elevation from January-June 2017: {0:.2f} ± {1:.2f} psu'
          .format(mls_elev[datetime(2017,1,1):datetime(2017,6,30)].mean(),
                  mls_elev[datetime(2017,1,1):datetime(2017,6,30)].std()))

    for dt in pd.date_range(start=datetime(2016,7,1),end=datetime(2016,12,1)):
        salt_to_destab = ml_psal_for_destab.loc[dt] - \
                         plot_doy_series(ml_avg_psal_median,N_years=1,ref_datetime=datetime(2015,12,31),
                                         just_return_series=True).loc[dt]
        print('Change in MLS from climatology needed to destabilize on {0}: {1:.2f} psu'.format(dt,salt_to_destab))

    for dt in pd.date_range(start=datetime(2016,7,1),end=datetime(2016,8,1)):
        salt_to_destab = ml_psal_for_destab.loc[dt] - max_ml_avg_psal.loc[dt]
        print('Change in MLS from observed needed to destabilize on {0}: {1:.2f} psu'.format(dt,salt_to_destab))

    mr_obs_mos = array([dt.month for dt in mr_obs['datetimes']])
    dts7 = mr_obs['datetimes'][mr_obs_mos == 7]
    mls7 = mr_obs['psal']['ml_avg'][mr_obs_mos == 7]
    sort_idxs = mls7.argsort()
    sort_idxs = flip(sort_idxs)
    print('Two highest MLS measured in July near Maud Rise (r < 250 km) were on:\n'
          '- {0}, {1} psu\n'
          '- {2}, {3} psu\n'
          'NOTE: this is from {4} July MLS observations featuring {5} years'
          ''.format(dts7[sort_idxs][0],mls7[sort_idxs][0],dts7[sort_idxs][1],mls7[sort_idxs][1],
                    len(mls7),len(unique([dt.year for dt in dts7]))))

    ew_obs_mos = array([dt.month for dt in e_weddell_obs['datetimes']])
    dts7 = e_weddell_obs['datetimes'][ew_obs_mos == 7]
    mls7 = e_weddell_obs['psal']['ml_avg'][ew_obs_mos == 7]
    sort_idxs = mls7.argsort()
    sort_idxs = flip(sort_idxs)
    print('Two highest MLS measured in July in Eastern Weddell (250 km < r < 500 km) were on:\n'
          '- {0}, {1} psu\n'
          '- {2}, {3} psu\n'
          'NOTE: this is from {4} July MLS observations featuring {5} years'
          ''.format(dts7[sort_idxs][0],mls7[sort_idxs][0],dts7[sort_idxs][1],mls7[sort_idxs][1],
                    len(mls7),len(unique([dt.year for dt in dts7]))))

    for dt in pd.date_range(start=datetime(2017,1,1),end=datetime(2017,4,1)):
        mls_elev = max_ml_avg_psal.loc[dt] \
                   - plot_doy_series(ml_avg_psal_median,N_years=1,ref_datetime=datetime(2016,12,31),
                                     just_return_series=True).loc[dt]
        print('Max MLS elevation on {0}: {1:.2f} psu'.format(dt,mls_elev))

    # upwelling calculation: overview
    print('For upwelling calculation... average climatological MLS from Jan-May near Maud Rise: {0:.2f} psu'
          .format(plot_doy_series(ml_avg_psal_median,N_years=1,ref_datetime=datetime(2015,12,31),
                                  just_return_series=True).loc['2016-01-01':'2016-05-31'].mean()))
    argo_gdac_index = pickle.load(open(argo_index_pickle_dir + 'argo_gdac_index.pickle','rb'))
    argo_soccom_index = pickle.load(open(argo_index_pickle_dir + 'argo_soccom_index.pickle','rb'))

    # upwelling calculation: Jan-May 2015
    baseline_curltau = -2.06e-7  # N m-3
    this_curltau = -2.90e-7      # N m-3
    num_days = 150               # from Jan-May 2015
    ekman_addtl = (num_days*24*60*60) * (this_curltau - baseline_curltau) / (1027.8 * gt.coriolis(-65.0))
    print('For upwelling calculation... additional upwelling in Jan-May 2015: {0:.2f} m'.format(ekman_addtl))
    wmoids = [5903616,5904468,5904471]
    toi_span = [20141231000000,20150601000000]
    float_data = []
    for f_idx, wmoid in enumerate(wmoids):
        this_float_meta = ldp.argo_gdac_float_meta(argo_gdac_index['local_prof_index'],wmoid)
        toi_mask = logical_and(this_float_meta['prof_datetimes'] >= toi_span[0],
                               this_float_meta['prof_datetimes'] <= toi_span[1])
        float_data.append(ldp.argo_float_data(wmoid,argo_gdac_dir,argo_gdac_index,argo_soccom_index,
                                              prof_nums=array(this_float_meta['prof_nums'])[toi_mask],
                                              compute_extras=False))
    mld_all = []
    mls_all = []
    sub_ml_sals = []
    for f_idx, wmoid in enumerate(wmoids):
        for p_idx in range(len(float_data[f_idx]['profiles'])):
            mld = gt.mld(float_data[f_idx]['profiles'][p_idx])
            mld_all.append(mld)
            mls_all.append(gt.vert_prof_eval(float_data[f_idx]['profiles'][p_idx],'psal',(0,mld),extrap='nearest'))
            sub_ml_sals.append(gt.vert_prof_eval(float_data[f_idx]['profiles'][p_idx],'psal',(mld,mld+ekman_addtl),extrap='nearest'))
    delta_S_ml = ((ekman_addtl*mean(sub_ml_sals) + (mean(mls_all)*(mean(mld_all)-ekman_addtl))) /
                  mean(mld_all)) - mean(mls_all)
    # error propagation due solely to float salinity accuracy
    psal_std = 0.01
    mean_mls_error = sqrt(len(mls_all) * (psal_std**2)) / len(mls_all)
    mean_sub_ml_sal_error = sqrt(len(sub_ml_sals) * (psal_std**2)) / len(sub_ml_sals)
    delta_S_ml_error_numerator1 = ekman_addtl * mean_sub_ml_sal_error
    delta_S_ml_error_numerator2 = (mean(mld_all)-ekman_addtl) * mean_mls_error
    delta_S_ml_error_numerator = sqrt(delta_S_ml_error_numerator1**2 + delta_S_ml_error_numerator2**2)
    delta_S_ml_error = sqrt((delta_S_ml_error_numerator/mean(mld_all))**2 + mean_mls_error**2)
    print('TEST - For upwelling calculation... propagated uncertainty in delta_S_ml in Jan-May 2015: {0:.3f} psu'
          .format(delta_S_ml_error))
    print('For upwelling calculation... average MLD in Jan-May 2015: {0:.2f} m'.format(mean(mld_all)))
    print('For upwelling calculation... average MLS in Jan-May 2015: {0:.2f} psu'.format(mean(mls_all)))
    print('For upwelling calculation... average MLD to MLD+(addtl Ekman) psal in Jan-May 2015: {0:.2f} psu'.format(mean(sub_ml_sals)))
    print('Final upwelling calculation... delta_S_ml in Jan-May 2015: {0:.3f} psu'.format(delta_S_ml))

    # upwelling calculation: Jan-May 2016
    this_curltau = -3.08e-7      # N m-3
    num_days = 151               # from Jan-May 2016
    ekman_addtl = (num_days*24*60*60) * (this_curltau - baseline_curltau) / (1027.8 * gt.coriolis(-65.0))
    print('For upwelling calculation... additional upwelling in Jan-May 2016: {0:.2f} m'.format(ekman_addtl))
    wmoids = [5903616,5904468,5904471]
    toi_span = [20151231000000,20160601000000]
    float_data = []
    for f_idx, wmoid in enumerate(wmoids):
        this_float_meta = ldp.argo_gdac_float_meta(argo_gdac_index['local_prof_index'],wmoid)
        toi_mask = logical_and(this_float_meta['prof_datetimes'] >= toi_span[0],
                               this_float_meta['prof_datetimes'] <= toi_span[1])
        float_data.append(ldp.argo_float_data(wmoid,argo_gdac_dir,argo_gdac_index,argo_soccom_index,
                                              prof_nums=array(this_float_meta['prof_nums'])[toi_mask],
                                              compute_extras=False))
    mld_all = []
    mls_all = []
    sub_ml_sals = []
    for f_idx, wmoid in enumerate(wmoids):
        for p_idx in range(len(float_data[f_idx]['profiles'])):
            mld = gt.mld(float_data[f_idx]['profiles'][p_idx])
            mld_all.append(mld)
            mls_all.append(gt.vert_prof_eval(float_data[f_idx]['profiles'][p_idx],'psal',(0,mld),extrap='nearest'))
            sub_ml_sals.append(gt.vert_prof_eval(float_data[f_idx]['profiles'][p_idx],'psal',(mld,mld+ekman_addtl),extrap='nearest'))
    delta_S_ml = ((ekman_addtl*mean(sub_ml_sals) + (mean(mls_all)*(mean(mld_all)-ekman_addtl))) /
                  mean(mld_all)) - mean(mls_all)
    # error propagation due solely to float salinity accuracy
    psal_std = 0.01
    mean_mls_error = sqrt(len(mls_all) * (psal_std**2)) / len(mls_all)
    mean_sub_ml_sal_error = sqrt(len(sub_ml_sals) * (psal_std**2)) / len(sub_ml_sals)
    delta_S_ml_error_numerator1 = ekman_addtl * mean_sub_ml_sal_error
    delta_S_ml_error_numerator2 = (mean(mld_all)-ekman_addtl) * mean_mls_error
    delta_S_ml_error_numerator = sqrt(delta_S_ml_error_numerator1**2 + delta_S_ml_error_numerator2**2)
    delta_S_ml_error = sqrt((delta_S_ml_error_numerator/mean(mld_all))**2 + mean_mls_error**2)
    print('TEST - For upwelling calculation... propagated uncertainty in delta_S_ml in Jan-May 2016: {0:.3f} psu'
          .format(delta_S_ml_error))
    print('For upwelling calculation... average MLD in Jan-May 2016: {0:.2f} m'.format(mean(mld_all)))
    print('For upwelling calculation... average MLS in Jan-May 2016: {0:.2f} psu'.format(mean(mls_all)))
    print('For upwelling calculation... average MLD to MLD+(addtl Ekman) psal in Jan-May 2016: {0:.2f} psu'.format(mean(sub_ml_sals)))
    print('Final upwelling calculation... delta_S_ml in Jan-May 2016: {0:.3f} psu'.format(delta_S_ml))
    # end printing statistics

    yspan = 0.35
    this_axis_height = 0.20
    for polynya_datetime in polynya_datetimes:
        plt.gca().axvspan(xmin=polynya_datetime[0]-timedelta(days=1),xmax=polynya_datetime[1],
                          ymin=0,ymax=yspan / this_axis_height,
                          facecolor='steelblue',alpha=0.15,zorder=1,clip_on=False)

    fig.canvas.draw()
    xticklabels = [xtl.get_text() for xtl in plt.gca().get_xticklabels()]
    jan_idxs = where(array(xticklabels) == 'Jan')[0]
    xticklabels[jan_idxs[0]] = 'Jan\n2016'  # note hard-coding here...
    xticklabels[jan_idxs[1]] = 'Jan\n2017'
    plt.gca().set_xticklabels(xticklabels)

    plt.savefig(current_results_dir + 'figure_2.pdf')
    plt.close()

    ##################################################################################################################
    ###################################   ED FIG. 4 PLOTTING ROUTINE   ###############################################
    ##################################################################################################################

    fig = plt.figure(figsize=(7,7))

    # 2016

    fig.add_axes([0.15,0.85,0.7,0.10]) # [x0, y0, width, height] for lower left point (from bottom left of figure)
    fig_2_plot_sic(plot_2016=True,plot_2017=False,xbounds=[datetime(2016,1,1),datetime(2016,12,31)])

    for storm_idx,storm_datetime in enumerate(storm_datetimes_2016):
        plt.text(storm_datetime[0] + timedelta(days=1.5),1.05,str(storm_idx + 1),
                 horizontalalignment='center',fontsize=fontsize)

    fig.add_axes([0.15,0.70,0.7,0.15])
    dsicdt_2016_dmsp = plot_doy_series(100*sic_doy_2016_dmsp.diff(),N_years=1,ref_datetime=datetime(2015,12,31),
                                       just_return_series=True).rolling(window=3).mean()
    dsicdt_2016_amsr = plot_doy_series(100*sic_doy_2016_amsr.diff(),N_years=1,ref_datetime=datetime(2015,12,31),
                                       just_return_series=True).rolling(window=3).mean()
    plt.fill_between(dsicdt_2016_dmsp.index,dsicdt_2016_dmsp.values,0,where=dsicdt_2016_dmsp.values <= 0,
                     interpolate=True,facecolor='darkgoldenrod',alpha=0.7,zorder=1)
    plt.plot([datetime(2016,1,1),datetime(2016,12,31)],[0,0],c='k',ls='--',lw=lw,zorder=2)
    plt.plot(dsicdt_2016_dmsp,c=color_accent,lw=lw,ls='-',zorder=3)
    plt.plot(dsicdt_2016_amsr,c=color_accent,lw=lw,ls=':',zorder=4)
    fig_2_axis_prep('d(SIC)/d$\it{t}$\n(% day$^{-1}$)',ylabel_pos='right',ylabel_offset=0.0,ylabel_pad=-0.03,
                    xbounds=[datetime(2016,1,1),datetime(2016,12,31)],spines=['bottom'])

    fig.add_axes([0.15,0.63,0.7,0.07])
    msl_series_daily_2016 = erai_series['msl_min_2016'].resample('D').min()
    msl_series_daily_climo \
        = erai_series['msl_min_climo_2016'].resample('D').mean().rolling(window=7,center=True,min_periods=1).mean()
    plt.plot(msl_series_daily_climo,c=shade_color,linewidth=lw,alpha=shade_centerline_alpha,zorder=2)
    plt.plot(msl_series_daily_2016,c='k',linewidth=lw,zorder=3)
    plt.plot([datetime(2016,1,1),datetime(2016,12,31)],[msl_crit,msl_crit],c='k',ls='--',lw=lw,zorder=1)
    plt.fill_between(msl_series_daily_2016.index,msl_series_daily_2016.values,msl_crit,
                     where=msl_series_daily_2016.values <= msl_crit,interpolate=True,
                     facecolor='darkgoldenrod',alpha=0.7,zorder=1)
    plt.ylim([925,1005])
    plt.gca().set_yticks([940,960,980,1000])
    plt.gca().set_yticklabels(['940','960','980','1,000'])
    fig_2_axis_prep('Minimum\nsea-level pressure\n' + r'(hPa)',ylabel_pos='left',ylabel_offset=0.0,
                    xbounds=[datetime(2016,1,1),datetime(2016,12,31)])

    fig.add_axes([0.15,0.55,0.7,0.08])
    si10_series_daily_2016 = erai_series['si10_max_2016'].resample('D').max()
    si10_series_daily_climo \
        = erai_series['si10_max_climo_2016'].resample('D').mean().rolling(window=7,center=True,min_periods=1).mean()
    plt.plot(si10_series_daily_climo,c=shade_color,linewidth=lw,alpha=shade_centerline_alpha,zorder=2)
    plt.plot(si10_series_daily_2016,c='k',linewidth=lw,zorder=3)
    plt.plot([datetime(2016,1,1),datetime(2016,12,31)],[si10_crit,si10_crit],c='k',ls='--',lw=lw,zorder=1)
    plt.fill_between(si10_series_daily_2016.index,si10_series_daily_2016.values,si10_crit,
                     where=si10_series_daily_2016.values >= si10_crit,interpolate=True,
                     facecolor='darkgoldenrod',alpha=0.7,zorder=1)
    plt.ylim([-5,28])
    plt.gca().set_yticks([5,10,15,20,25])
    plt.gca().set_yticklabels(['5','','15','','25'])
    fig_2_axis_prep('Maximum\n10-m wind speed\n' + r'(m s$^{-1}$)',ylabel_pos='right',ylabel_pad=-0.02,ylabel_offset=0.1,
                    xbounds=[datetime(2016,1,1),datetime(2016,12,31)])

    print('2016 storms found: {0}'.format(find_storms(si10_series_daily_2016,msl_series_daily_2016)))
    storm_alpha = 0.2
    storm_color = 'darkgoldenrod'
    yspan = 0.40
    this_axis_height = 0.08
    for storm_datetime in storm_datetimes_2016:
        plt.gca().axvspan(xmin=storm_datetime[0],xmax=storm_datetime[1],ymin=0,ymax=yspan / this_axis_height,
                          facecolor=storm_color,alpha=storm_alpha,zorder=1,clip_on=False)

    # 2017

    fig.add_axes([0.15,0.45,0.7,0.10]) # [x0, y0, width, height] for lower left point (from bottom left of figure)
    fig_2_plot_sic(plot_2016=False,plot_2017=True,plot_legend=False,xbounds=[datetime(2017,1,1),datetime(2017,12,31)])

    for storm_idx_2017,storm_datetime in enumerate(storm_datetimes_2017):   # start numbering from last 2016 storm
        storm_label = str(storm_idx + storm_idx_2017 + 2)
        if storm_label == '13': storm_label = '13 '
        if storm_label == '14': storm_label = '14  '
        if storm_label == '15': storm_label = '15  '
        plt.text(storm_datetime[0] + timedelta(days=1.5),1.05,storm_label,
                 horizontalalignment='center',fontsize=fontsize)

    fig.add_axes([0.15,0.30,0.7,0.15])
    dsicdt_2017_dmsp = plot_doy_series(100*sic_doy_2017_dmsp.diff(),N_years=1,ref_datetime=datetime(2016,12,31),
                                       just_return_series=True).rolling(window=3).mean()
    dsicdt_2017_amsr = plot_doy_series(100*sic_doy_2017_amsr.diff(),N_years=1,ref_datetime=datetime(2016,12,31),
                                       just_return_series=True).rolling(window=3).mean()
    plt.fill_between(dsicdt_2017_dmsp.index,dsicdt_2017_dmsp.values,0,where=dsicdt_2017_dmsp.values <= 0,
                     interpolate=True,facecolor='darkgoldenrod',alpha=0.7,zorder=1)
    plt.plot([datetime(2017,1,1),datetime(2017,12,31)],[0,0],c='k',ls='--',lw=lw,zorder=2)
    plt.plot(dsicdt_2017_dmsp,c=color_accent,lw=lw,ls='-',zorder=3)
    plt.plot(dsicdt_2017_amsr,c=color_accent,lw=lw,ls=':',zorder=4)
    fig_2_axis_prep('d(SIC)/d$\it{t}$\n(% day$^{-1}$)',ylabel_pos='right',ylabel_offset=-0.1,ylabel_pad=-0.03,
                    xbounds=[datetime(2017,1,1),datetime(2017,12,31)],spines=['bottom'])

    fig.add_axes([0.15,0.23,0.7,0.07])
    msl_series_daily_2017 = erai_series['msl_min_2017'].resample('D').min()
    msl_series_daily_climo \
        = erai_series['msl_min_climo_2017'].resample('D').mean().rolling(window=7,center=True,min_periods=1).mean()
    plt.plot(msl_series_daily_climo,c=shade_color,linewidth=lw,alpha=shade_centerline_alpha,zorder=2)
    plt.plot(msl_series_daily_2017,c='k',linewidth=lw,zorder=3)
    plt.plot([datetime(2017,1,1),datetime(2017,12,31)],[msl_crit,msl_crit],c='k',ls='--',lw=lw,zorder=1)
    plt.fill_between(msl_series_daily_2017.index,msl_series_daily_2017.values,msl_crit,
                     where=msl_series_daily_2017.values <= msl_crit,interpolate=True,
                     facecolor='darkgoldenrod',alpha=0.7,zorder=1)
    plt.ylim([925,1010])
    plt.gca().set_yticks([940,960,980,1000])
    plt.gca().set_yticklabels(['940','960','980','1,000'])
    fig_2_axis_prep('Minimum\nsea-level pressure\n' + r'(hPa)',ylabel_pos='left',ylabel_offset=0.0,
                    xbounds=[datetime(2017,1,1),datetime(2017,12,31)])

    fig.add_axes([0.15,0.16,0.7,0.07])
    si10_series_daily_2017 = erai_series['si10_max_2017'].resample('D').max()
    si10_series_daily_climo \
        = erai_series['si10_max_climo_2017'].resample('D').mean().rolling(window=7,center=True,min_periods=1).mean()
    plt.plot(si10_series_daily_climo,c=shade_color,linewidth=lw,alpha=shade_centerline_alpha,zorder=2)
    plt.plot(si10_series_daily_2017,c='k',linewidth=lw,zorder=3)
    plt.plot([datetime(2017,1,1),datetime(2017,12,31)],[si10_crit,si10_crit],c='k',ls='--',lw=lw,zorder=1)
    plt.fill_between(si10_series_daily_2017.index,si10_series_daily_2017.values,si10_crit,
                     where=si10_series_daily_2017.values >= si10_crit,interpolate=True,
                     facecolor='darkgoldenrod',alpha=0.7,zorder=1)
    plt.gca().set_yticks([5,10,15,20,25])
    plt.gca().set_yticklabels(['5','','15','','25'])
    fig_2_axis_prep('Maximum\n10-m wind speed\n' + r'(m s$^{-1}$)',ylabel_pos='right',ylabel_pad=-0.02,
                    xbounds=[datetime(2017,1,1),datetime(2017,12,31)],
                    spines=['bottom'],xlabel_bottom=True)

    print('2017 storms found: {0}'.format(find_storms(si10_series_daily_2017,msl_series_daily_2017)))
    yspan = 0.39
    this_axis_height = 0.07
    for storm_datetime in storm_datetimes_2017:
        plt.gca().axvspan(xmin=storm_datetime[0]-timedelta(days=1),xmax=storm_datetime[1],
                          ymin=0,ymax=yspan / this_axis_height,
                          facecolor=storm_color,alpha=storm_alpha,zorder=1,clip_on=False)

    # print storm statistics:
    print('Max wind speed on 2016-07-26: {0:.2f} m/s\n'
          'Max wind speed on 2016-07-27: {1:.2f} m/s\n'
          'Min surface pressure on 2016-07-26: {2:.2f} hPa\n'
          'Min surface pressure on 2016-07-27: {3:.2f} hPa\n'
          'Max wind speed on 2016-08-02: {4:.2f} m/s\n'
          'Max wind speed on 2016-08-04: {5:.2f} m/s\n'
          ''.format(si10_series_daily_2016.loc['2016-07-26'],si10_series_daily_2016.loc['2016-07-27'],
                    msl_series_daily_2016.loc['2016-07-26'],msl_series_daily_2016.loc['2016-07-27'],
                    si10_series_daily_2016.loc['2016-08-02'],si10_series_daily_2016.loc['2016-08-04']))

    plt.savefig(current_results_dir + 'ED_figure_4.pdf')
    plt.close()

    ##################################################################################################################
    ###################################   ED FIG. 3 PLOTTING ROUTINE   ###############################################
    ##################################################################################################################

    fig = plt.figure(figsize=(6,5))

    fig.add_axes([0.15,0.8,0.7,0.15])  # [x0, y0, width, height] for lower left point (from bottom left of figure)
    fig_2_plot_sic(plot_2017=True)

    for storm_datetime in storm_datetimes_2016 + storm_datetimes_2017:
        plt.gca().axvspan(xmin=storm_datetime[0] - timedelta(days=1),xmax=storm_datetime[1],ymin=1,ymax=1.09,
                          facecolor='k',alpha=1.0,zorder=1,clip_on=False)
    plt.text(0.16,1.01,'Winter storms:',horizontalalignment='right',verticalalignment='bottom',
             transform=plt.gca().transAxes,color='k',fontsize=fontsize - 1)

    right_axis = plt.gca().twinx()
    plt.sca(right_axis)
    polynya_extent_doy_2016[polynya_extent_doy_2016 > 150000] = 150000
    polynya_extent_doy_2017[polynya_extent_doy_2017 > 150000] = 150000
    polynya_extent_doy_2016_amsr[polynya_extent_doy_2016_amsr > 150000] = 150000
    polynya_extent_doy_2017_amsr[polynya_extent_doy_2017_amsr > 150000] = 150000
    polynya_doys_2016 = pd.Index(polynya_datetimes[0]).dayofyear
    polynya_doys_2017 = pd.Index(polynya_datetimes[1]).dayofyear
    polynya_extent_doy_2016 \
        = polynya_extent_doy_2016.truncate(before=polynya_doys_2016[0],after=polynya_doys_2016[1])
    polynya_extent_doy_2017 \
        = polynya_extent_doy_2017.truncate(before=polynya_doys_2017[0],after=polynya_doys_2017[1]-1)
    polynya_extent_doy_2016_amsr \
        = polynya_extent_doy_2016_amsr.truncate(before=polynya_doys_2016[0],after=polynya_doys_2016[1])
    polynya_extent_doy_2017_amsr \
        = polynya_extent_doy_2017_amsr.truncate(before=polynya_doys_2017[0],after=polynya_doys_2017[1]-1)
    plt.plot(*plot_doy_series(polynya_extent_doy_2016,N_years=1,ref_datetime=datetime(2015,12,31)),
             c=color_polynya,alpha=0.75,linewidth=lw,zorder=1)
    plt.plot(*plot_doy_series(polynya_extent_doy_2017,N_years=1,ref_datetime=datetime(2016,12,31)),
             c=color_polynya,alpha=0.75,linewidth=lw,zorder=1)
    plt.plot(*plot_doy_series(polynya_extent_doy_2016_amsr,N_years=1,ref_datetime=datetime(2015,12,31)),
             ls=':',c=color_polynya,alpha=0.75,linewidth=lw,zorder=1)
    plt.plot(*plot_doy_series(polynya_extent_doy_2017_amsr,N_years=1,ref_datetime=datetime(2016,12,31)),
             ls=':',c=color_polynya,alpha=0.75,linewidth=lw,zorder=1)
    plt.ylim([-2000,175000])
    plt.gca().set_yticks([0,25000,50000,75000,100000,125000,150000])
    plt.gca().set_yticklabels([0,'','50','','100','','≥150'])
    fig_2_axis_prep('Polynya extent\n' + r'(10$^3$ km$^2$)',spines=['top'],ylabel_pos='right',
                    ylabel_color=color_polynya,ylabel_offset=-0.05,ylabel_pad=-0.01)

    fig.add_axes([0.15,0.68,0.7,0.12])
    series_daily = erai_series['t2m_2016_2017']
    series_daily_climo \
        = pd.concat([erai_series['t2m_climo_2016'].resample('D').mean(),
                     erai_series['t2m_climo_2017'].resample('D').mean()]).rolling(window=7,center=True,min_periods=1).mean()
    series_daily_climo_iqr_25 \
        = pd.concat([erai_series['t2m_climo_iqr_25_2016'].resample('D').mean(),
                     erai_series['t2m_climo_iqr_25_2017'].resample('D').mean()]).rolling(window=7,center=True,min_periods=1).mean()
    series_daily_climo_iqr_75 \
        = pd.concat([erai_series['t2m_climo_iqr_75_2016'].resample('D').mean(),
                     erai_series['t2m_climo_iqr_75_2017'].resample('D').mean()]).rolling(window=7,center=True,min_periods=1).mean()
    plt.fill_between(series_daily_climo.index,
                     series_daily_climo.values - series_daily_climo_iqr_25.values,
                     series_daily_climo.values + series_daily_climo_iqr_75.values,
                     facecolor=shade_color,alpha=shade_alpha,zorder=1)
    plt.plot(series_daily_climo,c=shade_color,linewidth=lw,alpha=shade_centerline_alpha,zorder=2)
    plt.plot(series_daily,c=color_accent,linewidth=lw - 0.15,zorder=3)
    plt.gca().set_yticks([-20,-10,0])
    plt.ylim([-29,7])
    fig_2_axis_prep('2-m\nair temperature\n(°C)',ylabel_pos='left',ylabel_offset=0.0)

    fig.add_axes([0.15,0.57,0.7,0.12])
    plt.plot(*plot_doy_series(ew_mld_median - ew_mld_iqr_25),
             ls=':',c=ew_shade_color,linewidth=lw,alpha=ew_shade_centerline_alpha,zorder=1)
    h2, = plt.plot(*plot_doy_series(ew_mld_median),
                   ls='--',c=ew_shade_color,linewidth=lw,alpha=ew_shade_centerline_alpha,zorder=1)
    plt.plot(*plot_doy_series(ew_mld_median + ew_mld_iqr_75),
             ls=':',c=ew_shade_color,linewidth=lw,alpha=ew_shade_centerline_alpha,zorder=1)
    h1 = plt.fill_between(*plot_doy_series(mr_mld_median - mr_mld_iqr_25),
                          plot_doy_series(mr_mld_median + mr_mld_iqr_75)[1],
                          facecolor=shade_color,alpha=shade_alpha,zorder=2)
    plt.plot(*plot_doy_series(mr_mld_median),c=shade_color,linewidth=lw,alpha=shade_centerline_alpha,zorder=2)
    plt.plot(avg_mld,c=color_accent,linewidth=lw,zorder=3)
    plt.gca().invert_yaxis()
    plt.gca().set_yticks([150,100,50,0])
    plt.ylim([200,0])
    fig_2_axis_prep('Mixed-layer depth\n(m)',ylabel_pos='right',ylabel_offset=0.1,ylabel_pad=-0.02)

    fig.add_axes([0.15,0.50,0.7,0.12])
    plt.plot(*plot_doy_series(ew_ml_avg_ptmp_median - ew_ml_avg_ptmp_iqr_25),
             ls=':',c=ew_shade_color,linewidth=lw,alpha=ew_shade_centerline_alpha,zorder=1)
    h2, = plt.plot(*plot_doy_series(ew_ml_avg_ptmp_median),
                   ls='--',c=ew_shade_color,linewidth=lw,alpha=ew_shade_centerline_alpha,zorder=1,
                   label='Eastern Weddell')
    plt.plot(*plot_doy_series(ew_ml_avg_ptmp_median + ew_ml_avg_ptmp_iqr_75),
             ls=':',c=ew_shade_color,linewidth=lw,alpha=ew_shade_centerline_alpha,zorder=1)
    h1 = plt.fill_between(*plot_doy_series(ml_avg_ptmp_median - ml_avg_ptmp_iqr_25),
                          plot_doy_series(ml_avg_ptmp_median + ml_avg_ptmp_iqr_75)[1],
                          facecolor=shade_color,alpha=shade_alpha,zorder=2,
                          label='Maud Rise')
    plt.plot(*plot_doy_series(ml_avg_ptmp_median),
             c=shade_color,linewidth=lw,alpha=shade_centerline_alpha,zorder=2)
    plt.plot(avg_ml_avg_ptmp,c=color_accent,linewidth=lw,zorder=3)
    plt.gca().set_yticks([-2,-1,0,1,2])
    plt.ylim([-2.1,3])
    fig_2_axis_prep('Mixed-layer\ntemperature\n(°C)',ylabel_pos='left',ylabel_offset=-0.1,ylabel_pad=-0.02)
    plt.legend(handles=[h1,h2],bbox_to_anchor=(0.003,-1.78),loc='lower left',ncol=2,frameon=False,
               fontsize=fontsize-1,handlelength=1.75,handletextpad=0.6)

    # print statistics on MLT during August melt event
    mlt_elev = (ml_avg_ptmp_5904471.loc['2016-08-31'] \
                - plot_doy_series(ml_avg_ptmp_median,N_years=1,ref_datetime=datetime(2015,12,31),
                                 just_return_series=True).loc[datetime(2016,8,31)]).values[0]
    print('MLT elevation on August 31, 2016 from 5904471: {0:.2f} °C'.format(mlt_elev))

    fig.add_axes([0.15,0.29,0.7,0.22])
    sd_depth = 250  # keep as int
    mr_sd_median = ldp.hydro_obs_to_doy_series(mr_obs,'sd',sd_depth,doy_median=True,days_per_bin=dpb,rm_days=dpb)
    mr_sd_iqr_25 = ldp.hydro_obs_to_doy_series(mr_obs,'sd',sd_depth,doy_iqr_25=True,days_per_bin=dpb,rm_days=dpb)
    mr_sd_iqr_75 = ldp.hydro_obs_to_doy_series(mr_obs,'sd',sd_depth,doy_iqr_75=True,days_per_bin=dpb,rm_days=dpb)
    ew_sd_median = ldp.hydro_obs_to_doy_series(e_weddell_obs,'sd',sd_depth,doy_median=True,days_per_bin=dpb,rm_days=dpb)
    ew_sd_iqr_25 = ldp.hydro_obs_to_doy_series(e_weddell_obs,'sd',sd_depth,doy_iqr_25=True,days_per_bin=dpb,rm_days=dpb)
    ew_sd_iqr_75 = ldp.hydro_obs_to_doy_series(e_weddell_obs,'sd',sd_depth,doy_iqr_75=True,days_per_bin=dpb,rm_days=dpb)
    sd_5903616 = pd.Series(index=mr_obs['datetimes'][mr_obs_mask_5903616],
                           data=mr_obs['sd'][sd_depth][mr_obs_mask_5903616])
    sd_5904471 = pd.Series(index=mr_obs['datetimes'][mr_obs_mask_5904471],
                           data=mr_obs['sd'][sd_depth][mr_obs_mask_5904471])
    sd_5904468 = pd.Series(index=mr_obs['datetimes'][mr_obs_mask_5904468],
                           data=mr_obs['sd'][sd_depth][mr_obs_mask_5904468])
    min_sd = pd.concat([series_daily_interp(sd_5903616),
                        series_daily_interp(sd_5904471),
                        series_daily_interp(sd_5904468)],axis=1).min(axis=1)
    max_sd = pd.concat([series_daily_interp(sd_5903616),
                        series_daily_interp(sd_5904471),
                        series_daily_interp(sd_5904468)],axis=1).max(axis=1)
    plt.fill_between(*plot_doy_series(mr_sd_median - mr_sd_iqr_25),plot_doy_series(mr_sd_median + mr_sd_iqr_75)[1],
                     facecolor=shade_color,alpha=shade_alpha,zorder=2)
    plt.plot(*plot_doy_series(mr_sd_median),c=shade_color,linewidth=lw,alpha=shade_centerline_alpha,zorder=2)
    plt.plot(min_sd,c=color_accent,linewidth=lw,zorder=3)
    plt.gca().invert_yaxis()
    plt.gca().set_yticks([1.75,1.5,1.25,1.0,0.75,0.5])
    plt.ylim([2.15,0.25])

    freeze_2016 = [datetime(2016,5,27),datetime(2016,7,11)]
    base = min_sd[freeze_2016[0]]
    height = min_sd[freeze_2016[0]]-min_sd[freeze_2016[1]]
    plt.plot([freeze_2016[0],freeze_2016[0]],[base,base-height],c='0.4',lw=0.5,ls=':') # prev. 'darkred'
    plt.plot(freeze_2016,[base-height,base-height],c='0.4',lw=0.5,ls=':')
    plt.text(freeze_2016[0],base-0.5*height,horizontalalignment='right',verticalalignment='bottom',
             s='{0:.2f} m '.format(height),fontsize=fontsize-1,color='0.4')

    melt_2016 = [datetime(2016,9,29),datetime(2016,12,23)]
    base = min_sd[melt_2016[1]]
    height = min_sd[melt_2016[1]]-min_sd[melt_2016[0]]
    plt.plot([melt_2016[1],melt_2016[1]],[base,base-height],c='0.4',lw=0.5,ls=':')
    plt.plot(melt_2016,[base-height,base-height],c='0.4',lw=0.5,ls=':')
    plt.text(melt_2016[1],base-0.65*height,horizontalalignment='left',verticalalignment='bottom',
             s=' {0:.2f} m'.format(height),fontsize=fontsize-1,color='0.4')

    melt_climo = [datetime(2016,9,16),datetime(2017,1,11)]
    mr_sd_median_series = pd.Series(index=plot_doy_series(mr_sd_median)[0],data=plot_doy_series(mr_sd_median)[1])
    base = mr_sd_median_series[melt_climo[1]]
    height = mr_sd_median_series[melt_climo[1]]-mr_sd_median_series[melt_climo[0]]
    plt.plot([melt_climo[0],melt_climo[0]],[base,base-height],c='0.4',lw=0.5,ls=':')
    plt.plot(melt_climo,[base,base],c='0.4',lw=0.5,ls=':')
    plt.text(melt_climo[0],base-0.35*height,horizontalalignment='left',verticalalignment='top',
             s=' {0:.2f} m'.format(height),fontsize=fontsize-1,color='0.4')

    diff_2017 = [melt_2016[1],melt_climo[1]]
    base = mr_sd_median_series[diff_2017[1]]
    height = mr_sd_median_series[diff_2017[1]] - min_sd[diff_2017[0]]
    plt.plot([diff_2017[1],diff_2017[1]],[base,base-height],c='0.4',lw=0.75,ls=':')
    plt.plot(diff_2017,[base - height,base - height],c='0.4',lw=0.75,ls=':')
    plt.text(diff_2017[1],base - 0.5 * height,horizontalalignment='left',verticalalignment='center',
             s=' {0:.2f} m'.format(height),fontsize=fontsize - 1,color='0.4',weight='bold')

    freeze_2017 = [datetime(2017,5,23),datetime(2017,6,24)]
    base = min_sd[freeze_2017[0]]
    height = min_sd[freeze_2017[0]]-min_sd[freeze_2017[1]]
    plt.plot([freeze_2017[0],freeze_2017[0]],[base,base-height],c='0.4',lw=0.5,ls=':')
    plt.plot(freeze_2017,[base-height,base-height],c='0.4',lw=0.5,ls=':')
    plt.text(freeze_2017[0],base-0.4*height,horizontalalignment='right',verticalalignment='bottom',
             s='{0:.2f} m '.format(height),fontsize=fontsize-1,color='0.4')

    melt_2017 = [freeze_2017[1],datetime(2017,9,5)]
    base = min_sd[melt_2017[1]]
    height = min_sd[melt_2017[1]]-min_sd[melt_2017[0]]
    plt.plot([melt_2017[1],melt_2017[1]],[base,base-height],c='0.4',lw=0.5,ls=':')
    plt.plot(melt_2017,[base-height,base-height],c='0.4',lw=0.5,ls=':')
    plt.text(melt_2017[1],base-0.9*height,horizontalalignment='left',verticalalignment='bottom',
             s=' {0:.2f} m'.format(height),fontsize=fontsize-1,color='0.4')

    ideal_2 = [melt_2016[0],melt_2016[1],freeze_2017[0],freeze_2017[1],melt_2017[1]]
    plt.plot(ideal_2,min_sd[ideal_2],c='0.4',lw=0.5,ls='-')

    fig_2_axis_prep('0-{0} m\nfreshwater anomaly\n(m sea ice equivalent)'.format(sd_depth),
                    ylabel_pos='right',ylabel_offset=0.05,ylabel_pad=0.00,
                    spines=['bottom'],xlabel_bottom=True)

    yspan = 0.66
    this_axis_height = 0.22
    for polynya_datetime in polynya_datetimes:
        plt.gca().axvspan(xmin=polynya_datetime[0]-timedelta(days=1),xmax=polynya_datetime[1],
                          ymin=0,ymax=yspan / this_axis_height,
                          facecolor='steelblue',alpha=0.15,zorder=1,clip_on=False)

    fig.canvas.draw()
    xticklabels = [xtl.get_text() for xtl in plt.gca().get_xticklabels()]
    jan_idxs = where(array(xticklabels) == 'Jan')[0]
    xticklabels[jan_idxs[0]] = 'Jan\n2016'  # note hard-coding here...
    xticklabels[jan_idxs[1]] = 'Jan\n2017'
    plt.gca().set_xticklabels(xticklabels)

    plt.savefig(current_results_dir + 'ED_figure_3.pdf')
    plt.close()

# Fig. 3. Local meteorology and heat loss during the 2016 polynya.
if plot_fig_3:
    # requires sea ice pickles created above in <<plot_fig_2_ED_figs_3_4>>
    sic_doy_dmsp,sic_doy_amsr,sic_doy_2016_dmsp,sic_doy_2016_amsr,sic_doy_2017_dmsp,sic_doy_2017_amsr, \
    polynya_extent_doy_2016,polynya_extent_doy_2017 \
        = pickle.load(open(figure_pickle_dir + 'fig_2_ice_with_amsr_polynya_extent','rb'))

    use_gwkm_pickle = True
    use_erai_pickle = True

    # load COARE 2.0 turbulent heat flux time series (not generated within these scripts, but available upon request)
    if use_gwkm_pickle:
        era_gwkm = pickle.load(open(figure_pickle_dir + 'fig_3_era_gwkm','rb'))
    else:
        era_sources = ['erai']
        era_vars = ['icec','thfx_ow']
        era_gwkm = dict()
        for era_source in era_sources:
            for era_var in era_vars:
                data_frame = pd.read_csv(era_processed_gwk_moore_dir + '{0}_csv/{1}_{0}_ec_y2016.csv'
                                                                       ''.format(era_source,era_var),header=None)
                data_frame.index = [datetime(2016,data_frame.loc[row,1],data_frame.loc[row,2])
                                    + timedelta(hours=int(data_frame.loc[row,3]))
                                    for row in range(len(data_frame))]
                data_frame = data_frame.loc[:,4:]
                data_frame = data_frame.rename(columns={4:'mean',5:'mean_climo_mean',6:'mean_climo_std',
                                                        7:'max',8:'max_climo_mean',9:'max_climo_std',
                                                        10:'min',11:'min_climo_mean',12:'min_climo_std'})
                era_gwkm['{0}_{1}'.format(era_source,era_var)] = data_frame
        pickle.dump(era_gwkm,open(figure_pickle_dir + 'fig_3_era_gwkm','wb'))

    # total ocean-atmosphere heat flux that would be experienced by a region of open water
    erai_thfx_2016 = 100 * era_gwkm['erai_thfx_ow']['mean'] / (100 - era_gwkm['erai_icec']['mean'])

    # calculate average heat flux during 2016 polynya:
    avg_thfx_2016 = erai_thfx_2016.loc[datetime(2016,7,27):datetime(2016,8,16,23)].mean()
    print('Average ocean-atmosphere heat flux during 2016 polynya = {0:.2f} W m-2'.format(avg_thfx_2016))

    # output heat flux
    pickle.dump(erai_thfx_2016,open(figure_pickle_dir + 'fig_3_erai_thfx','wb'))

    # load 2016 reanalysis data and create time series of interest
    toi_2016 = [datetime(2016,7,22),datetime(2016,8,20)]
    mr_box_small = [0,10,-67,-63]
    if use_erai_pickle:
        [erai_t2m,erai_msl_min,erai_si10_max,erai_si10] = pickle.load(open(figure_pickle_dir + 'fig_3_erai_daily','rb'))
    else:
        erai_daily = ldp.load_ecmwf(era_custom_dir,'erai_daily_weddell.nc',datetime_range=toi_2016)
        erai_t2m = ldp.create_reanalysis_index(erai_daily,param_name='t2m',avg_box=mr_box_small,min_not_mean=False)[0]
        erai_msl_min = ldp.create_reanalysis_index(erai_daily,param_name='msl',avg_box=mr_box_small,min_not_mean=True)[0]
        erai_si10_max = ldp.create_reanalysis_index(erai_daily,param_name='si10',avg_box=mr_box_small,max_not_mean=True)[0]
        erai_si10 = ldp.create_reanalysis_index(erai_daily,param_name='si10',avg_box=mr_box_small,min_not_mean=False)[0]
        pickle.dump([erai_t2m,erai_msl_min,erai_si10_max,erai_si10],open(figure_pickle_dir + 'fig_3_erai_daily','wb'))
    polynya_ws10m = erai_si10.loc[datetime(2016,7,27):datetime(2016,8,17)]
    polynya_ws10m_accum = cumsum(polynya_ws10m - mean(polynya_ws10m))
    erai_thfx_2016_polynya = erai_thfx_2016.loc[datetime(2016,7,27):datetime(2016,8,17)]

    def plot_doy_series(series,ref_datetime=datetime(2015,12,31),shift_12_hours=False):
        if shift_12_hours: hour_shift = 12
        else:              hour_shift = 0
        new_index = array([timedelta(days=int(doy),hours=hour_shift) for doy in series.index]) + ref_datetime
        return new_index,series.values

    def fig_2_axis_prep(toi=None,ylabel='',spines=[],xlabel_top=False,xlabel_bottom=False,ylabel_pos='left',
                        ylabel_color='k',ylabel_offset=0.0,ylabel_extra_pad=0.0,remove_yticks=[],
                        ygrid=False,xgrid=False):
        plt.xlim(toi)
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=3))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %-d'))
        plt.tick_params(axis='x',which='both',bottom=xlabel_bottom,top=xlabel_top,
                        labeltop=xlabel_top,labelbottom=xlabel_bottom)
        plt.xticks(rotation=45)
        if xgrid: plt.gca().grid(which='major',axis='x',linewidth=0.5,alpha=0.3)
        if ygrid: plt.gca().grid(which='major',axis='y',linewidth=0.5,alpha=0.3)
        plt.gca().tick_params(axis='both',which='major',labelsize=fontsize)
        [plt.gca().spines[side].set_linewidth(0.5) for side in plt.gca().spines.keys()]
        plt.gca().tick_params(width=0.5)
        if 'top' not in spines: plt.gca().spines['top'].set_visible(False)
        if 'bottom' not in spines: plt.gca().spines['bottom'].set_visible(False)
        for tick in remove_yticks:
            plt.gca().get_yticklabels()[tick].set_visible(False)
            plt.gca().yaxis.get_major_ticks()[tick].set_visible(False)
        if ylabel_pos == 'right':
            plt.gca().yaxis.tick_right()
            plt.gca().yaxis.set_label_position('right')
        plt.gca().patch.set_alpha(0.0)
        if ylabel_pos == 'left':  ylabel_rot = 90
        if ylabel_pos == 'right': ylabel_rot = -90
        plt.ylabel(ylabel,fontsize=fontsize,rotation=ylabel_rot,color=ylabel_color)
        if ylabel_pos == 'left':  plt.gca().get_yaxis().set_label_coords(-0.09 - ylabel_extra_pad,0.5 + ylabel_offset)
        if ylabel_pos == 'right': plt.gca().get_yaxis().set_label_coords(1.12 + ylabel_extra_pad,0.5 - ylabel_offset)
        plt.setp(plt.gca().get_yticklabels(),color=ylabel_color)

    def find_storms(si10_series,msl_series,si10_crit=20,msl_crit=950,after=datetime(2016,7,22)):
        storm_times_all = []
        for time in si10_series.index:
            if time < after: continue
            if si10_series[time] >= si10_crit or msl_series[time] <= msl_crit:
                storm_times_all.append(time)
        return storm_times_all


    # plotting routine

    fig = plt.figure(figsize=(3.5,3.0))
    fontsize = 5

    fig.add_axes([0.2,0.75-((5/65)*0.20),0.6,(1+(5/65))*0.20])
    plt.fill_between(polynya_ws10m_accum.index,polynya_ws10m_accum.values,0,facecolor='maroon',alpha=0.1,zorder=1)
    plt.plot(polynya_ws10m_accum,c='maroon',linewidth=0.75,zorder=2)
    plt.plot([datetime(2016,7,27),datetime(2016,7,27)],[0,polynya_ws10m_accum.loc[datetime(2016,7,27)]],
             c='maroon',ls=':',linewidth=0.5,zorder=1)
    plt.ylim([-5,65])
    plt.gca().set_yticks([0.3,20.3,40.3,60.3])
    plt.gca().set_yticklabels(['0','20','40','60'])
    fig_2_axis_prep(toi=toi_2016,ylabel='Cumulative wind speed\n' + r'anomaly (m s$^{-1}$)',ylabel_color='maroon',
                    ylabel_pos='right',ylabel_extra_pad=0.075)

    fig.add_axes([0.2,0.75,0.6,0.20])
    plt.fill_between(*plot_doy_series(polynya_extent_doy_2016,shift_12_hours=True),0,
                     facecolor='k',alpha=0.1,zorder=3)
    plt.plot(*plot_doy_series(polynya_extent_doy_2016,shift_12_hours=True),c='k',linewidth=0.75,zorder=4)
    plt.plot([datetime(2016,1,1),datetime(2016,12,31)],[0,0],c='k',linestyle=':',linewidth=0.75,zorder=4)
    plt.ylim([0,40000])
    plt.gca().set_yticks([100,10100,20100,30100,40100])
    plt.gca().set_yticklabels(['','10','20','30','40'])
    fig_2_axis_prep(toi=toi_2016,ylabel='Polynya extent\n' + r'(10$^3$ km$^2$)',ylabel_color='k',
                    ylabel_offset=0.1,ylabel_extra_pad=0.025,
                    spines=['top'],ylabel_pos='left')

    storm_datetimes_2016 = find_storms(erai_si10_max,erai_msl_min)
    for storm_datetime in storm_datetimes_2016:
        plt.gca().axvspan(xmin=storm_datetime,xmax=storm_datetime+timedelta(hours=12),ymin=1,ymax=1.12,
                          facecolor='k',alpha=1.0,zorder=1,clip_on=False)
    plt.text(0.01,1.01,'Storms:',horizontalalignment='left',verticalalignment='bottom',
             transform=plt.gca().transAxes,color='k',fontsize=fontsize)

    fig.add_axes([0.2,0.60,0.6,0.15])
    plt.plot(erai_si10,c='maroon',linewidth=0.75,zorder=1)
    plt.plot([datetime(2016,7,27),datetime(2016,8,17)],[mean(polynya_ws10m),mean(polynya_ws10m)],
             c='maroon',linestyle='--',linewidth=0.5,zorder=2,label='Mean during polynya')
    plt.ylim([1,25])
    plt.gca().set_yticks([5,10,15,20])
    fig_2_axis_prep(toi=toi_2016,ylabel='10-m\nwind speed\n' + r'(m s$^{-1}$)',ylabel_color='maroon',
                    ylabel_pos='left',ylabel_extra_pad=0.025)
    plt.legend(loc='upper right',frameon=False,fontsize=fontsize)

    fig.add_axes([0.2,0.46,0.6,0.15])
    plt.plot(erai_thfx_2016_polynya,c='k',linewidth=0.75,zorder=2)
    plt.plot([datetime(2016,7,27),datetime(2016,8,17)],[mean(erai_thfx_2016_polynya),mean(erai_thfx_2016_polynya)],
             c='k',linestyle='--',linewidth=0.5,zorder=1,label='Mean during polynya')
    plt.plot([datetime(2016,7,27),datetime(2016,7,27)],[0,erai_thfx_2016.loc[datetime(2016,7,27)]],
             c='k',ls=':',linewidth=0.5,zorder=1)
    plt.plot([datetime(2016,8,17),datetime(2016,8,17)],[0,erai_thfx_2016.loc[datetime(2016,8,17)]],
             c='k',ls=':',linewidth=0.5,zorder=1)
    plt.ylim([0,800])
    plt.gca().set_yticks([200,400,600])
    fig_2_axis_prep(toi=toi_2016,ylabel='Heat flux\n' + r'(W m$^{-2}$)',
                    ylabel_extra_pad=0.075,
                    ylabel_pos='right',spines=['bottom'],xlabel_bottom=True)

    # print maximum heat flux value:
    print('Value of ERAI THFX on {0}: {1:.2f} W/m2'
          .format('2016-08-02 12:00:00',erai_thfx_2016.loc['2016-08-02 12:00:00']))

    # lagged correlation analysis of cumulative wind speed and polynya extent
    polynya_extent_2016_amsr = polynya_extent_doy_2016.copy()
    polynya_extent_2016_amsr.index \
        = array([timedelta(days=int(doy),hours=12) for doy in polynya_extent_2016_amsr.index]) + datetime(2015,12,31)
    polynya_extent_lags = arange(0,4,0.25)  # in days (i.e. 0.25 = 6 hours)
    ws10m_accum_polynya_extent_corr \
        = [polynya_ws10m_accum.corr(polynya_extent_2016_amsr.shift(freq=timedelta(days=lag)),method='pearson')
           for lag in polynya_extent_lags]
    print_str = 'Cumulative sum of area mean 10 m wind speed anomaly from time mean\n\n' \
                'Mean 10 m wind speed from 2016-7-27 to 2016-8-16: {0} m/s\n\n' \
                'Pearson correlation coefficient computed with lagged polynya extent:\n' \
                'Polynya extent lags (hours): {1}\n' \
                'Correlation coefficients: {2}\n' \
                ''.format(mean(polynya_ws10m),24 * polynya_extent_lags,ws10m_accum_polynya_extent_corr)

    text_file = open(current_results_dir + 'figure_3_accum_wind_speed_polynya_extent.txt','w')
    text_file.write(print_str)
    text_file.close()

    plt.savefig(current_results_dir + 'figure_3.pdf')
    plt.close()

# Fig. 4. Hydrographic observations from Maud Rise from 2011–2018.
# Extended Data Fig. 6. Full set of profiling float hydrographic observations from Maud Rise from 2011–2018.
# Extended Data Fig. 7. Heat loss during the 2016 polynya estimated from hydrographic observations.
if plot_fig_4_ED_figs_6_7:
    plot_by_depth_composite = True
    plot_by_depth = True

    [sea_ice_grids,sea_ice_data_avail,sea_ice_all_dates] = ldp.sea_ice_data_prep(nimbus5_dir,dmsp_v3_dir,dmsp_nrt_dir,
                                                                                 amsre_dir,amsr2_dir,amsr_gridfile,
                                                                                 amsr_areafile,nsidc_ps25_grid_dir)

    argo_gdac_index = pickle.load(open(argo_index_pickle_dir + 'argo_gdac_index.pickle','rb'))
    argo_soccom_index = pickle.load(open(argo_index_pickle_dir + 'argo_soccom_index.pickle','rb'))

    polynya_dates = [20160727000000,20160816000000,20170901000000,20171201000000]
    polynya_dates = [tt.convert_tuple_to_datetime(tt.convert_14_to_tuple(pd)) for pd in polynya_dates]

    wmoids = [5903616,5904468,5904471]
    toi_span = [[20111218000000,20160603000000],  # 1627 days
                [20150118000000,20180509000000],  # 1205 days, previously 1070 days
                [20141220000000,20180623000000]]  # 1279 days, previously 1269 days
    traj_plot_params = [(1.2 * 500000,220000,-65.4,1),(1.2 * 570000,500000,-65.5,3),(1.2 * 590000,500000,-65.5,2)]
    params = ['ptmp','psal','Oxygen','Nsquared','destab']
    condensed_cbar_labels = ['Potential temperature\n(°C)','Salinity\n(psu)','Oxygen\n' + r'(µmol kg$^{-1}$)',
                             'Buoyancy frequency\nsquared ' + r'(10$^{-7}$ s$^{-2}$)',
                             'Convection resistance\n' + r'(m$^2$ s$^{-2}$)']
    width_ratios = [1627,1205,1279 * (12.0 / 9.6)]
    height_ratios = [1650 + 175,1650,1650,1650,1650]
    xlabel_years_only = [True,True,True]
    show_ice_bars = True
    all_trajs = []
    float_data = []
    for f_idx,wmoid in enumerate(wmoids):
        this_float_meta = ldp.argo_gdac_float_meta(argo_gdac_index['local_prof_index'],wmoid)
        toi_mask = logical_and(this_float_meta['prof_datetimes'] >= toi_span[f_idx][0],
                               this_float_meta['prof_datetimes'] <= toi_span[f_idx][1])
        float_meta = [[wmoid,this_float_meta['prof_lons'][toi_mask],this_float_meta['prof_lats'][toi_mask],
                       this_float_meta['prof_position_flags'][toi_mask],
                       this_float_meta['prof_datetimes'][toi_mask]]]
        all_trajs.append(float_meta)
        float_data.append(ldp.argo_float_data(wmoid,argo_gdac_dir,argo_gdac_index,argo_soccom_index,
                                              prof_nums=array(this_float_meta['prof_nums'])[toi_mask],
                                              compute_extras=True,smooth_N2_PV=True,smooth_N2_PV_window=50.0))

    # calculation: estimate heat loss between pre-2016-polynya and May 31, 2017 (before ice growth)
    # NOTE: this loads the 'fig_3_erai_thfx' pickle from the Fig. 3 section
    erai_thfx_2016 = pickle.load(open(figure_pickle_dir + 'fig_3_erai_thfx','rb'))
    avg_thfx_2016 = -1 * erai_thfx_2016.loc[datetime(2016,7,27):datetime(2016,8,16,23)].mean()

    rho_w = 1027.8  # kg/m3
    c_p = 3850  # J kg-1 °C-1
    dz = 0.1  # m
    dt = 21 * 24 * 60 * 60  # 21 days to seconds
    z_range = [200,1650]  # m
    polynya_start_on = 20160727000000
    pre_polynya_idx = [(all_trajs[1][0][4] < polynya_start_on).argmin() - 1,
                       (all_trajs[2][0][4] < polynya_start_on).argmin() - 1]  # 5904468 and 5904471, respectively
    end_period_on = 20170101000000  # before 2017 ice growth
    post_polynya_idx = [(all_trajs[1][0][4] < end_period_on).argmin() - 1,
                        (all_trajs[2][0][4] < end_period_on).argmin() - 1]  # up to and including these indices
    heat_loss_estimates = []  # one entry per profile; both floats included here
    post_polynya_dates = []  # for testing/reality check
    for f_idx in [0,1]:
        for p_idx in range(pre_polynya_idx[f_idx],post_polynya_idx[f_idx] + 1):
            z_vec,ptmp_vec = gt.vert_prof_even_spacing(float_data[f_idx + 1]['profiles'][p_idx],'ptmp',
                                                       z_coor='depth',spacing=dz,interp_method='linear',
                                                       extrap='nearest',top=z_range[0],bottom=z_range[1])
            heat_content = rho_w * c_p * sum(ptmp_vec) * dz
            if p_idx == pre_polynya_idx[f_idx]:
                initial_heat_content = heat_content
            else:
                heat_loss_estimates.append((heat_content - initial_heat_content) / dt)
                post_polynya_dates.append(all_trajs[f_idx + 1][0][4][p_idx])
    heat_loss_estimates = array(heat_loss_estimates)
    days_since_polynya_open \
        = array([td.days for td in array([tt.convert_14_to_datetime(dti) for dti in post_polynya_dates])
                 - tt.convert_14_to_datetime(polynya_start_on)])
    print('Statistics for heat loss between pre-2016 polynya profile and January 1, 2017, '
          'between {0} and {1} m:\n'
          '- mean heat loss: {2:.2f} W m-2\n'
          '- median heat loss: {3:.2f} W m-2\n'
          '- {4:.2f}% of heat loss values were between 0 and the average open-water polynya heat flux of {5:.2f} W m-2\n'
          '- N = {6} estimates from 2 floats'
          .format(*z_range,mean(heat_loss_estimates),median(heat_loss_estimates),
                  100 * len(heat_loss_estimates[logical_and(heat_loss_estimates < 0,
                                                            heat_loss_estimates > avg_thfx_2016)]) / len(
                      heat_loss_estimates),
                  avg_thfx_2016,len(heat_loss_estimates)))

    plt.figure(figsize=(6.5,4))
    plt.hist(heat_loss_estimates,bins=arange(-450,100,50),color='k',rwidth=0.95)
    current_ylim = plt.ylim()
    plt.scatter(heat_loss_estimates,tile(-0.75,len(heat_loss_estimates)),s=25,c='k',marker='d',linewidths=0,alpha=0.8)
    new_ylim = [-1.5,current_ylim[1]]
    plt.ylim(new_ylim)
    plt.plot([0,0],new_ylim,c='k',ls='-',lw=1)
    plt.plot([avg_thfx_2016,avg_thfx_2016],new_ylim,c='r',ls='--',lw=3)
    plt.plot([-375,75],[0,0],c='k',ls='-',lw=1)
    plt.xlim([-375,75])
    plt.ylabel('Profile count')
    plt.xlabel('Heat flux estimate (W m$^{-2}$)')
    plt.tight_layout()
    plt.savefig(current_results_dir + 'ED_figure_7.pdf')
    plt.close()

    # full sections by depth
    if plot_by_depth:
        pt.section_compiler(wmoids,data_dir,current_results_dir,'ED_figure_6',float_data,params,figsize=(8.5,7.5),
                            depth_lim=(0,1650),mld=True,plot_title=False,sea_ice_grids=sea_ice_grids,
                            sea_ice_data_avail=sea_ice_data_avail,add_date_bars=polynya_dates,
                            condensed_cbar_labels=condensed_cbar_labels,width_ratios=width_ratios,
                            height_ratios=height_ratios,all_trajs=None,traj_plot_params=traj_plot_params,
                            show_ice_bars=show_ice_bars,density_coor=False,force_label_size=6,
                            explicit_yticks=[0,500,1000,1500],years_only=xlabel_years_only)

    # condensed sections
    if plot_by_depth_composite:
        wmoids = [5903616,5904468,5904471]
        toi_span = [[0,20150205000000],[20150201000000,20170210000000],[20170210000000,20180623000000]]
        params = ['ptmp','psal','Oxygen','Nsquared']
        condensed_cbar_labels = ['Potential temperature\n(°C)','Salinity\n(psu)','Oxygen\n(µmol/kg)',
                                 'Buoyancy frequency\nsquared ' + r'(10$^{-7}$ s$^{-2}$)']
        width_ratios = [1.64,1.07,0.84]
        height_ratios = [1650 + 175,1650,1650,1650]
        xlabel_years_only = [True,True,True]
        show_ice_bars = True
        all_trajs = []
        float_data = []
        for f_idx,wmoid in enumerate(wmoids):
            this_float_meta = ldp.argo_gdac_float_meta(argo_gdac_index['local_prof_index'],wmoid)
            toi_mask = logical_and(this_float_meta['prof_datetimes'] >= toi_span[f_idx][0],
                                   this_float_meta['prof_datetimes'] <= toi_span[f_idx][1])
            float_meta = [[wmoid,this_float_meta['prof_lons'][toi_mask],this_float_meta['prof_lats'][toi_mask],
                           this_float_meta['prof_position_flags'][toi_mask],
                           this_float_meta['prof_datetimes'][toi_mask]]]
            all_trajs.append(float_meta)
            float_data.append(ldp.argo_float_data(wmoid,argo_gdac_dir,argo_gdac_index,argo_soccom_index,
                                                  prof_nums=array(this_float_meta['prof_nums'])[toi_mask],
                                                  compute_extras=True,smooth_N2_PV=True,smooth_N2_PV_window=50.0))

        pt.section_compiler(wmoids,data_dir,current_results_dir,'figure_4',float_data,params,figsize=(6.5,5.25),
                            depth_lim=(0,1650),mld=True,plot_title=False,sea_ice_grids=sea_ice_grids,
                            sea_ice_data_avail=sea_ice_data_avail,add_date_bars=polynya_dates,
                            condensed_cbar_labels=condensed_cbar_labels,width_ratios=width_ratios,
                            height_ratios=height_ratios,all_trajs=None,
                            traj_plot_params=traj_plot_params,show_ice_bars=show_ice_bars,density_coor=False,
                            force_label_size=6,explicit_yticks=[0,500,1000,1500],years_only=xlabel_years_only,
                            w_pad=0.1)

# Fig. 5. Relationships between past polynyas near Maud Rise and climate forcing from 1972–2018.
# Extended Data Fig. 9. Additional relationships between past polynyas near Maud Rise, climate forcing, and
#                       sub-pycnocline temperatures.
# Extended Data Table 1. Correlations and trends for climate indices and sub-pycnocline temperature records.
if plot_fig_5_ED_fig_9_ED_table_1:
    use_sic_polynya_pickle = True
    use_mr_precip_pickle = True
    use_weddell_storm_pickle = True
    use_hydro_obs_pickle = True
    use_erai_sanae_pickle = True
    verbose = True

    rm_mo = 24          # number of months to include in rolling means
    rm_min = 12         # minimum number of months for rolling mean to be calculated
    rm_cent = True      # if True, calculate centered rolling mean; if False, calculate right-edge (accumulated) version

    corr_rm_mo = 12     # 12 months to mitigate effects of seasonality
    corr_rm_min = 6
    corr_lag_range = arange(-36,36,1)  # in months

    open_thresholds = [40,50,60]    # SIC thresholds for polynya identification
    extent_threshold = 0            # minimum polynya extent in km^2 to include in cycle plots and pickles
    sic_lon_bounds = array([0,10])
    sic_lat_bounds = array([-67,-63])
    polynya_lon_bounds = array([-20,15])
    polynya_lat_bounds = array([-68,-62])
    # note: this script will fail if searching for polynyas in a box crossing the opposite meridian (-180)
    daily_extent_threshold = 1000     # in km^2; note this simply requires two 25x25 km pixels above the SIC threshold
    use_fixed_winter_period = False   # if False, winter period = one week after first day of 90% SIC
                                      #                        to one week before last day of 90% SIC
                                      # if True, use DOYs below
                                      # note 1: if SIC criteria not available for a year, defaults to fixed period below
                                      # note 2: years listed in <<fixed_winter_period_years>> default to fixed period
    fixed_winter_period = (tt.convert_date_to_365((1900,7,1)),tt.convert_date_to_365((1900,10,31))) # simply Jul-Oct
    lower_sic_threshold = 0.2  # the freeze/melt period start/end threshold; not used in paper
    upper_sic_threshold = 0.9  # the winter period SIC threshold
    fixed_winter_period_years = [1973,1974,1975,1976,2017]
    missing_years = [1972,1977,1978,1987]  # years known to have incomplete/missing SIC and polynya records

    # note: boxes are [lon_W,lon_E,lat_S,lat_N]
    weddell_low_search_box = [-60,60,-70,-60] # searches for mean monthly low location; final box mean ± 1-sigma lat/lon
    weddell_curlt_box = [-60,45,-80,-60]      # Weddell gyre, for wind stress curl; excludes continent (land cells)
    mr_curlt_box      = [  0,10,-67,-63]      # Maud Rise,    for wind stress curl; excludes continent (land cells)
    mr_storm_box      = [-15,20,-68,-62]      # for storm indices, precip records
    storm_si10_thresh = 20.0                  # storm max wind speed criterion (m/s)
    storm_slp_thresh  = 950.0                 # storm min sea-level pressure criterion (hPa)
    mr_storm_count_winter_months = [5,10]     # winter month range (inclusive) for storm indices

    mr_center = [-65.0,3.0]            # summit of Maud Rise (65.0°S, 3.0°E)
    e_weddell_hydro_radius = 500       # search radius (km) from Maud Rise
    mr_obs_box = [[-20,25],[-75,-59]]  # be generous; just a first pass to whittle down data (prev. [[-15,20],[-68,-62]])
    mr_obs_toi = [datetime(1970,1,1),datetime.today()]
    mr_obs_toi_recent = [datetime(2002,7,1),datetime(2017,12,31)]
    waghc_anomaly_depths_deep = (218,258,302,350,402,458,518,582,650,722,798,878,962,1050)
    waghc_anomaly_depth_instructions = [258,(250,1000,waghc_anomaly_depths_deep)]
    ew_depths = [15,150,250,258,waghc_anomaly_depths_deep]
    ew_tmp_depth = 258
    ew_tmp_key = '258_anomaly'
    ew_tmp_depth_range = (250,1000)
    ew_tmp_depth_range_key = '250-1000_anomaly'
    mr_depth_check = [30,None]  # rejection criteria for casts that start too deep or end too shallow (e.g. [20,200])

    # hard-coded vertical bars for polynya years (polynya-days > 0)
    polynya_years = [1973,1974,1975,1976,1993,1994,1995,1996,1999,2000,2001,2005,2016,2017]
    polynya_year_alpha = 0.075

    # vertical bars for polynyas from 2003-2017
    polynya_alpha = 0.25
    polynya_color = 'maroon'
    polynya_datetimes = [[datetime(2005,6,25),datetime(2005,7,18)], # widened from 2005,7,1 to 2005,7,13 for aesthetics
                         [datetime(2016,7,27),datetime(2016,8,17)],
                         [datetime(2017,8,31),datetime(2017,12,1)]]

    # load Antarctic coastline
    circumant_lons,circumant_lats = gt.establish_coastline(coastline_filename_prefix)

    # load sea ice concentration metadata
    [sea_ice_grids,sea_ice_data_avail,sea_ice_all_dates] \
        = ldp.sea_ice_data_prep(nimbus5_dir,dmsp_v3_dir,dmsp_nrt_dir,amsre_dir,amsr2_dir,amsr_gridfile,
                                amsr_areafile,nsidc_ps25_grid_dir)

    def detrend(series,slope,intercept,just_return_trend=False):
        new_data = intercept + (slope*mdates.date2num(series.index.date))
        trend_series = pd.Series(data=new_data,index=series.index)
        if just_return_trend: return trend_series
        else:                 return series - trend_series

    if not use_sic_polynya_pickle:
        # create SIC record
        sic_record = {}
        sic_days_of_year = {}
        for index,date in enumerate(tt.dates_in_range(sea_ice_all_dates[0],tt.now())):
            [conc,open_area,day_offset] = ldp.sea_ice_concentration(date,sic_lat_bounds,sic_lon_bounds,sea_ice_grids,
                                                                    sea_ice_data_avail,use_goddard_over_amsr=True)
            if date[0] not in sic_record: sic_record[date[0]] = []
            if date[0] not in sic_days_of_year: sic_days_of_year[date[0]] = []
            sic_record[date[0]].append(conc)
            sic_days_of_year[date[0]].append(tt.convert_date_to_365(date))
            if verbose: print('creating SIC record for ' + str(date))
        pickle.dump(sic_record,open(figure_pickle_dir + 'fig_5_sic_record','wb'))
        pickle.dump(sic_days_of_year,open(figure_pickle_dir + 'fig_5_sic_days_of_year','wb'))

        # create polynya record for each SIC threshold
        for open_threshold in open_thresholds:
            polynya_record = {} # dict with years as keys to lists for polynya for each day with good data (<= 366)
            # e.g. polynya_record[year] = list with length len(polynya_days_of_year)
            polynya_days_of_year = {} # as above, except with days of year instead of polynya areas
            for index,date in enumerate(sea_ice_all_dates):
                # hierarchy of sensors
                if sea_ice_data_avail['nimbus5'][date][1]:
                    sat_abbrev = 'nimbus5'
                elif sea_ice_data_avail['dmsp'][date][1]:
                    sat_abbrev = 'dmsp'
                elif sea_ice_data_avail['amsre'][date][1]:
                    sat_abbrev = 'amsre'
                elif sea_ice_data_avail['amsr2'][date][1]:
                    sat_abbrev = 'amsr2'
                else:
                    if date[0] not in polynya_record: polynya_record[date[0]] = []
                    if date[0] not in polynya_days_of_year: polynya_days_of_year[date[0]] = []
                    polynya_record[date[0]].append(NaN)
                    polynya_days_of_year[date[0]].append(tt.convert_date_to_365(date))
                    continue

                # identify polynyas
                sat_string,polynya_string,filename_abbrev,sic_grid,sic_field, \
                polynya_stats,polynya_grid,polynya_grid_binary,open_ocean_grid,error_code \
                    = gt.identify_polynyas_magic(sat_abbrev,date,sea_ice_grids,sea_ice_data_avail,circumant_lons,
                                                 circumant_lats,open_threshold=open_threshold,
                                                 extent_threshold=extent_threshold,regrid_amsr_to_25km=True)

                # no errors in identifying polynyas
                if error_code == 0:
                    total_polynya_extent = 0
                    for polynya_index in range(len(polynya_stats)):
                        if polynya_lat_bounds[0] <= polynya_stats[polynya_index]['centroid'][0] <= polynya_lat_bounds[1] \
                                and polynya_lon_bounds[0] <= polynya_stats[polynya_index]['centroid'][1] <= \
                                        polynya_lon_bounds[1]:
                            total_polynya_extent += polynya_stats[polynya_index]['total_extent']
                    # save polynya record for all sectors and all thresholds on THIS date
                    if date[0] not in polynya_record: polynya_record[date[0]] = []
                    if date[0] not in polynya_days_of_year: polynya_days_of_year[date[0]] = []
                    polynya_record[date[0]].append(total_polynya_extent)
                    polynya_days_of_year[date[0]].append(tt.convert_date_to_365(date))
                    if verbose: print('polynya record for {0}% on {1}: {2}'
                                      .format(open_threshold,str(date),str(total_polynya_extent)))

                # fully or partially bad SIC field
                else:
                    if date[0] not in polynya_record: polynya_record[date[0]] = []
                    if date[0] not in polynya_days_of_year: polynya_days_of_year[date[0]] = []
                    polynya_record[date[0]].append(NaN)
                    polynya_days_of_year[date[0]].append(tt.convert_date_to_365(date))
                    continue

            pickle.dump(polynya_record,open(figure_pickle_dir + 'fig_5_polynya_record_{0}'.format(open_threshold),'wb'))
            pickle.dump(polynya_days_of_year,
                        open(figure_pickle_dir + 'fig_5_polynya_days_of_year_{0}'.format(open_threshold),'wb'))
    else:
        sic_record = pickle.load(open(figure_pickle_dir + 'fig_5_sic_record','rb'))
        sic_days_of_year = pickle.load(open(figure_pickle_dir + 'fig_5_sic_days_of_year','rb'))

    # polynya statistics calculation
    polynya_max_all_thresh = []
    polynya_percent_all_thresh = []
    polynya_days_all_thresh = []
    for open_threshold in open_thresholds:
        # load daily record of polynyas
        polynya_record = pickle.load(open(figure_pickle_dir+'fig_5_polynya_record_{0}'.format(open_threshold),'rb'))
        polynya_days_of_year = pickle.load(open(figure_pickle_dir+'fig_5_polynya_days_of_year_{0}'.format(open_threshold),'rb'))

        # calculate sea ice statistics using daily SIC time series from plot_sea_ice_polynya_cycle()
        # note for metrics: defaults to NaN if not achieved in a given year
        freeze_start_doy_by_year = {}  # first day of, e.g., ≥10% SIC in each year
        freeze_end_doy_by_year = {}    # first day of, e.g., ≥90% SIC in each year
        melt_start_doy_by_year = {}    # last day of, e.g., ≥90% SIC in each year
        melt_end_doy_by_year = {}      # last day of, e.g., ≥10% SIC in each year (or Day 366 if still ≥10% by Day 366)
        winter_length_by_year = {}
        freeze_length_by_year = {}
        melt_length_by_year = {}
        freeze_approx_doy_range = [20,217]
        for year in list(sic_record.keys()):
            freeze_started = False
            winter_started = False
            winter_ended = False
            melt_ended = False
            for doy_idx, doy in enumerate(sic_days_of_year[year]):
                if not freeze_started and freeze_approx_doy_range[0] < doy < freeze_approx_doy_range[1] \
                        and sic_record[year][doy_idx] >= lower_sic_threshold:
                    freeze_started = True
                    freeze_start_doy_by_year[year] = doy
                if freeze_started and not winter_started and sic_record[year][doy_idx] >= upper_sic_threshold:
                    winter_started = True
                    freeze_end_doy_by_year[year] = doy
                if winter_started and not winter_ended and sic_record[year][doy_idx] <= upper_sic_threshold:
                    winter_ended = True
                    melt_start_doy_by_year[year] = doy - 1
                if winter_started and winter_ended and sic_record[year][doy_idx] >= upper_sic_threshold:
                    winter_ended = False
                if winter_started and winter_ended and not melt_ended and sic_record[year][doy_idx] <= lower_sic_threshold:
                    melt_ended = True
                    melt_end_doy_by_year[year] = doy - 1
                if winter_started and winter_ended and melt_ended and sic_record[year][doy_idx] >= lower_sic_threshold:
                    melt_ended = False
            if year in freeze_end_doy_by_year and year in melt_start_doy_by_year:
                winter_length_by_year[year] = melt_start_doy_by_year[year] - freeze_end_doy_by_year[year]
            if year in freeze_start_doy_by_year and year in freeze_end_doy_by_year:
                freeze_length_by_year[year] = freeze_end_doy_by_year[year] - freeze_start_doy_by_year[year]
            if year in melt_start_doy_by_year and year in melt_end_doy_by_year:
                melt_length_by_year[year] = melt_end_doy_by_year[year] - melt_start_doy_by_year[year]

        # convert sea ice statistics by year to Pandas Series
        freeze_start = pd.Series(freeze_start_doy_by_year)
        freeze_end = pd.Series(freeze_end_doy_by_year)
        melt_start = pd.Series(melt_start_doy_by_year)
        melt_end = pd.Series(melt_end_doy_by_year)
        winter_length = pd.Series(winter_length_by_year)
        freeze_length = pd.Series(freeze_length_by_year)
        melt_length = pd.Series(melt_length_by_year)

        # calculate polynya statistics
        avg_daily_polynya_extent_by_year = {}
        avg_daily_polynya_extent_when_over_threshold_by_year = {}
        max_polynya_extent_by_year = {}
        fraction_days_above_polynya_threshold_by_year = {}
        days_above_polynya_threshold_by_year = {}
        for year in list(polynya_record.keys()):
            if not use_fixed_winter_period:
                if year in freeze_end_doy_by_year: winter_period_begin = freeze_end_doy_by_year[year] + 7
                else:                              winter_period_begin = fixed_winter_period[0]
                if year in melt_start_doy_by_year: winter_period_end = melt_start_doy_by_year[year] - 7
                else:                              winter_period_end = fixed_winter_period[1]
                winter_period = [winter_period_begin,winter_period_end]
                if year in fixed_winter_period_years:
                    winter_period = fixed_winter_period
            else:
                winter_period = fixed_winter_period
            total_polynya_extent = 0
            days_above_polynya_threshold_counter = 0
            total_polynya_extent_on_days_above_threshold = 0
            max_daily_polynya_extent = 0
            day_counter = 0
            for doy_index, doy in enumerate(polynya_days_of_year[year]):
                if winter_period[0] <= doy <= winter_period[1]:
                    if not isnan(polynya_record[year][doy_index]):
                        total_polynya_extent += polynya_record[year][doy_index]
                        if polynya_record[year][doy_index] >= daily_extent_threshold:
                            days_above_polynya_threshold_counter += 1
                            total_polynya_extent_on_days_above_threshold += polynya_record[year][doy_index]
                        day_counter += 1
                        if polynya_record[year][doy_index] > max_daily_polynya_extent:
                            max_daily_polynya_extent = polynya_record[year][doy_index]
            if day_counter > 0:
                avg_daily_polynya_extent_by_year[year] = total_polynya_extent/day_counter
                if days_above_polynya_threshold_counter == 0:
                    avg_daily_polynya_extent_when_over_threshold_by_year[year] = 0
                    max_polynya_extent_by_year[year] = 0
                else:
                    avg_daily_polynya_extent_when_over_threshold_by_year[year] \
                        = total_polynya_extent_on_days_above_threshold/days_above_polynya_threshold_counter
                    max_polynya_extent_by_year[year] = max_daily_polynya_extent
                fraction_days_above_polynya_threshold_by_year[year] = days_above_polynya_threshold_counter/day_counter
                days_above_polynya_threshold_by_year[year] = days_above_polynya_threshold_counter
            else:
                avg_daily_polynya_extent_when_over_threshold_by_year[year] = NaN
                max_polynya_extent_by_year[year] = NaN
                fraction_days_above_polynya_threshold_by_year[year] = NaN
                days_above_polynya_threshold_by_year[year] = NaN

        # convert polynya time series into Pandas Series
        polynya_max = pd.Series(max_polynya_extent_by_year).dropna()
        polynya_percent = pd.Series(fraction_days_above_polynya_threshold_by_year).dropna()
        polynya_days = pd.Series(days_above_polynya_threshold_by_year).dropna()
        polynya_max_all_thresh.append(polynya_max)
        polynya_percent_all_thresh.append(polynya_percent)
        polynya_days_all_thresh.append(polynya_days)

    # load climate indices
    sam_index = ldp.load_sam_index(climate_indices_dir)

    # convert indices from Pandas DataFrame to Series
    sam_index_series = ldp.climate_index_DataFrame_to_Series(sam_index,dropna=True)

    # Antarctic met station records
    # NOTE: for quality control of sparse or suspiciously anomalous values,
    #       first two years of Neumayer (1981, 1982) and SANAE AWS (1997, 1998) records are dropped
    reader_novo = ldp.load_reader_station(reader_dir,'Novolazarevskaya.All.msl_pressure.txt')
    reader_neumayer = ldp.load_reader_station(reader_dir,'Neumayer.All.msl_pressure.txt').loc['1983':]
    reader_syowa = ldp.load_reader_station(reader_dir,'Syowa.All.msl_pressure.txt')
    isd_sanae_saf = ldp.load_isd_station(isd_dir,890010)['mslp'].groupby(pd.Grouper(freq='MS')).mean()
    isd_sanae_aws = ldp.load_isd_station(isd_dir,890040)['mslp'].groupby(pd.Grouper(freq='MS')).mean().loc['1999':]
    isd_neumayer = ldp.load_isd_station(isd_dir,890020)['mslp'].groupby(pd.Grouper(freq='MS')).mean()
    isd_maitri = ldp.load_isd_station(isd_dir,895140)['mslp'].groupby(pd.Grouper(freq='MS')).mean()

    # load reanalysis fields
    erai_monthly_mean = ldp.load_ecmwf(era_custom_dir,'erai_monthly_mean_weddell.nc')
    erai_monthly_mean_forecast = ldp.load_ecmwf(era_custom_dir,'erai_monthly_mean_weddell_forecast.nc')

    # comparison of ERA-Interim with Queen Maud Land station record
    sanae_aws_loc = (-2.8,-71.7)
    isd_sanae_aws_subdaily = ldp.load_isd_station(isd_dir,890040)['mslp']
    if use_erai_sanae_pickle:
        erai_sanae_aws_subdaily = pickle.load(open(figure_pickle_dir + 'fig_5_erai_sanae','rb'))
    else:
        erai_daily = ldp.load_ecmwf(era_custom_dir,'erai_daily_weddell.nc')
        erai_sanae_aws_subdaily \
            = erai_daily['msl'].sel(lons=sanae_aws_loc[0],lats=sanae_aws_loc[1],method='nearest').to_series()
        pickle.dump(erai_sanae_aws_subdaily,open(figure_pickle_dir + 'fig_5_erai_sanae','wb'))
    text_file = open(current_results_dir + 'figure_5_erai_sanae_mslp_comparison.txt','w')
    text_file.write('Comparison of subdaily ERA-Interim and Queen Maud Land station pressure record:\n'
                    'Correlation coefficient: {0:.2f}\n'
                    'Mean absolute deviation: {1:.2f} hPa\n'
                    'Mean deviation (bias) [ERAI minus SANAE-AWS]: {2:.2f} hPa'
                    .format(erai_sanae_aws_subdaily.corr(isd_sanae_aws_subdaily),
                            nanmean(abs(erai_sanae_aws_subdaily - isd_sanae_aws_subdaily)),
                            nanmean(erai_sanae_aws_subdaily - isd_sanae_aws_subdaily)))
    text_file.close()

    # determine Weddell Low location and create index
    erai_wl_index, erai_wl_index_filtered, _, wl_avg_box, wl_center \
        = ldp.create_reanalysis_index(erai_monthly_mean,param_name='msl',
                                      rm_window=rm_mo,rm_min=rm_min,rm_center=rm_cent,
                                      calc_box_here=True,search_box=weddell_low_search_box)
    text_file = open(current_results_dir + 'figure_5_weddell_low_position.txt','w')
    text_file.write('Weddell Low identified in ERA-Interim\n'
                    'with mean position at: {0}\n'
                    'resulting in averaging box of: {1}'.format(wl_center,wl_avg_box))
    text_file.close()

    # create precipitation record
    if use_mr_precip_pickle:
        erai_mr_precip = pickle.load(open(figure_pickle_dir + 'ED_fig_9_erai_mr_precip','rb'))
    else:
        erai_daily_forecast = ldp.load_ecmwf(era_custom_dir,'erai_daily_weddell_forecast.nc')
        erai_mr_precip_subdaily \
            = ldp.create_reanalysis_index(erai_daily_forecast,param_name='tp',avg_box=mr_storm_box)[0]
        erai_mr_precip = (erai_mr_precip_subdaily * 6 * 60 * 60).groupby(pd.Grouper(freq='MS')).sum()
        pickle.dump(erai_mr_precip,open(figure_pickle_dir + 'ED_fig_9_erai_mr_precip','wb'))

    # create storm record
    if use_weddell_storm_pickle:
        erai_mr_min_pres = pickle.load(open(figure_pickle_dir + 'fig_5_erai_mr_min_pres','rb'))
        erai_mr_max_si10 = pickle.load(open(figure_pickle_dir + 'fig_5_erai_mr_max_si10','rb'))
    else:
        erai_daily = ldp.load_ecmwf(era_custom_dir,'erai_daily_weddell.nc')
        erai_mr_min_pres \
            = ldp.create_reanalysis_index(erai_daily,param_name='msl',avg_box=mr_storm_box,min_not_mean=True)[0]
        pickle.dump(erai_mr_min_pres,open(figure_pickle_dir + 'fig_5_erai_mr_min_pres','wb'))
        erai_mr_max_si10 \
            = ldp.create_reanalysis_index(erai_daily,param_name='si10',avg_box=mr_storm_box,max_not_mean=True)[0]
        pickle.dump(erai_mr_max_si10,open(figure_pickle_dir + 'fig_5_erai_mr_max_si10','wb'))

    storm_rm_mo = int((rm_mo / 12) * (1 + mr_storm_count_winter_months[1] - mr_storm_count_winter_months[0]))
    storm_rm_min = int((rm_min / 12) * (1 + mr_storm_count_winter_months[1] - mr_storm_count_winter_months[0]))
    storm_corr_rm_mo = int((corr_rm_mo / 12) * (1 + mr_storm_count_winter_months[1] - mr_storm_count_winter_months[0]))
    storm_corr_rm_min = int((corr_rm_min / 12) * (1 + mr_storm_count_winter_months[1] - mr_storm_count_winter_months[0]))

    print('Percent of 6-hourly ERAI with min pressure < threshold: {0:.2f}'
          .format(100*(erai_mr_min_pres < 950).sum()/erai_mr_min_pres.count()))
    print('Percent of 6-hourly ERAI with max wind speed > threshold: {0:.2f}'
          .format(100*(erai_mr_max_si10 > 20).sum()/erai_mr_max_si10.count()))

    erai_mr_daily_min_pressure = erai_mr_min_pres.groupby(pd.Grouper(freq='D')).min()
    erai_mr_daily_max_si10 = erai_mr_max_si10.groupby(pd.Grouper(freq='D')).max()
    erai_mr_daily_storm_mask = logical_or(erai_mr_daily_min_pressure < storm_slp_thresh,
                                          erai_mr_daily_max_si10 > storm_si10_thresh)
    erai_mr_storm_count_by_month \
        = erai_mr_daily_min_pressure[erai_mr_daily_storm_mask].groupby(pd.Grouper(freq='MS')).count().loc[:'2018']
    erai_mr_storm_count_by_month[logical_or(erai_mr_storm_count_by_month.index.month
                                            < mr_storm_count_winter_months[0],
                                            erai_mr_storm_count_by_month.index.month
                                            > mr_storm_count_winter_months[1])] = NaN
    erai_mr_storm_count_filtered \
        = erai_mr_storm_count_by_month.dropna().rolling(window=storm_rm_mo,min_periods=storm_rm_min,center=rm_cent).mean()
    erai_mr_storm_count_winter = erai_mr_storm_count_by_month.dropna().resample('AS').mean()

    # create wind stress curl records
    erai_weddell_curlt \
        = ldp.create_reanalysis_index(erai_monthly_mean_forecast,param_name='curlt',avg_box=weddell_curlt_box,
                                      mask_land=erai_mask)[0]
    erai_mr_curlt \
        = ldp.create_reanalysis_index(erai_monthly_mean_forecast,param_name='curlt',avg_box=mr_curlt_box,
                                      mask_land=erai_mask)[0]

    # close reanalysis datasets
    erai_monthly_mean.close()
    erai_monthly_mean_forecast.close()
    

    # create Weddell hydrography record from WOD and Argo observations
    mr_params_to_calc = ['psal','temp','ptmp','sigma_theta']
    mr_params_for_climo_anomalies = ['temp']
    if not use_hydro_obs_pickle:
        bypass_compile_obs = False
        if not bypass_compile_obs:
            ### PART 1: eastern Weddell (500 km radius from MR)

            # run to compile all historical obs (necessary to map obs from previous decades)
            ldp.compile_hydrographic_obs(argo_index_pickle_dir,argo_gdac_dir,wod_dir,
                                         lon_bounds=mr_obs_box[0],lat_bounds=mr_obs_box[1],toi_bounds=mr_obs_toi,
                                         distance_check=e_weddell_hydro_radius,distance_center=mr_center,
                                         include_argo=True,include_wod=True,params=mr_params_to_calc,compute_extras=False,
                                         max_cast_min_depth=mr_depth_check[0],min_cast_max_depth=mr_depth_check[1],
                                         reject_mld_below=None,interp_spacing=0.1,interp_depths=None,
                                         calc_mld=True,calc_ml_avg=True,
                                         calc_at_depths=ew_depths,calc_depth_avgs=mr_depth_averages,
                                         pickle_dir=figure_pickle_dir,pickle_filename='ED_fig_9_e_weddell_obs',
                                         prof_count_dir=current_results_dir,
                                         prof_count_filename='ED_figure_9_prof_counts_e_weddell',verbose=verbose)

            # run simply to get stats for recent obs
            ldp.compile_hydrographic_obs(argo_index_pickle_dir,argo_gdac_dir,wod_dir,
                                         lon_bounds=mr_obs_box[0],lat_bounds=mr_obs_box[1],
                                         toi_bounds=mr_obs_toi_recent,
                                         distance_check=e_weddell_hydro_radius,distance_center=mr_center,
                                         include_argo=True,include_wod=True,params=mr_params_to_calc,compute_extras=False,
                                         max_cast_min_depth=mr_depth_check[0],min_cast_max_depth=mr_depth_check[1],
                                         reject_mld_below=None,interp_spacing=0.1,interp_depths=None,
                                         calc_mld=True,calc_ml_avg=True,
                                         calc_at_depths=ew_depths,calc_depth_avgs=mr_depth_averages,
                                         pickle_dir=figure_pickle_dir,pickle_filename='ED_fig_9_e_weddell_obs_recent_only',
                                         prof_count_dir=current_results_dir,
                                         prof_count_filename='ED_figure_9_prof_counts_e_weddell_recent',verbose=verbose)

        ### PART 2: calculate anomalies, group observations, calculate statistics of interest
        e_weddell_hydro_record \
            = ldp.hydro_obs_to_monthly_series(figure_pickle_dir,'ED_fig_9_e_weddell_obs_recent_only','ED_fig_9_e_weddell_hydro_record',
                                              mr_params_to_calc,mr_params_for_climo_anomalies,
                                              waghc_anomaly_depth_instructions,waghc_dir,
                                              climo='waghc17',N_min=5,verbose=verbose)

    else:
        e_weddell_hydro_record = pickle.load(open(figure_pickle_dir + 'ED_fig_9_e_weddell_hydro_record','rb'))


    ##################################################################################################################
    ############################################   PLOTTING PREPARATION   ############################################
    ##################################################################################################################

    # polynya extent and days
    polynya_days_all = []
    polynya_max_all = []
    for thresh_idx, open_threshold in enumerate(open_thresholds):
        polynya_days = polynya_days_all_thresh[thresh_idx].copy()
        polynya_max = polynya_max_all_thresh[thresh_idx].copy()
        polynya_days.index = [datetime(year=y,month=7,day=1) for y in list(polynya_days.index)] # center bars
        polynya_max.index = [datetime(year=y,month=7,day=1) for y in list(polynya_max.index)]
        polynya_days = polynya_days.drop(datetime(1987,7,1)) # because of large data gap in early winter 1987
        polynya_max = polynya_max.drop(datetime(1987,7,1))
        polynya_days.loc['1979':'1986'] *= 2    # double polynya day counts for SMMR period of every-other-day data
        polynya_days_all.append(polynya_days)
        polynya_max_all.append(polynya_max)

    # SAM index
    sam_mean = sam_index_series.loc['1972':'2018'].mean()
    sam_index_series_filtered = sam_index_series.rolling(window=rm_mo,min_periods=rm_min,center=rm_cent).mean()
    [sam_slope,sam_intercept,_,sam_p] \
        = stats.linregress(mdates.date2num(sam_index_series.dropna().loc['1972':'2018'].index.date),
                           sam_index_series.dropna().loc['1972':'2018'].values)[0:4]
    sam_index_trend = detrend(sam_index_series_filtered,sam_slope,sam_intercept,just_return_trend=True)

    # Weddell Low index
    erai_wl_index_no_nans = erai_wl_index[~isnan(erai_wl_index.values)]
    erai_wl_index_filtered_no_nans = erai_wl_index_filtered[~isnan(erai_wl_index_filtered.values)]
    wl_index_mean = erai_wl_index.loc[:'2018'].mean()
    [erai_wl_slope,erai_wl_intercept,_,erai_wl_p] \
        = stats.linregress(mdates.date2num(erai_wl_index_no_nans.loc[:'2018'].index.date),
                           erai_wl_index_no_nans.loc[:'2018'].values)[0:4]
    erai_trend_line = detrend(erai_wl_index_filtered_no_nans,erai_wl_slope,erai_wl_intercept,just_return_trend=True)

    # Novo index
    novo = reader_novo.loc['1972':]
    novo_filtered = novo.rolling(window=rm_mo,min_periods=rm_min,center=rm_cent).mean()
    novo_no_nans = novo[~isnan(novo.values)]
    novo_filtered_no_nans = novo_filtered[~isnan(novo_filtered.values)]
    novo_mean = novo.loc[:'2018'].mean()
    [novo_slope,novo_intercept,_,novo_p] \
        = stats.linregress(mdates.date2num(novo_no_nans.loc[:'2018'].index.date),
                           novo_no_nans.loc[:'2018'].values)[0:4]
    novo_trend_line = detrend(novo_filtered_no_nans,novo_slope,novo_intercept,just_return_trend=True)

    # Weddell gyre wind stress curl
    curlt_mean = erai_weddell_curlt.loc[:'2018'].mean()
    erai_weddell_curlt_filtered = erai_weddell_curlt.rolling(window=rm_mo,min_periods=rm_min,center=rm_cent).mean()
    curlt_no_nans = erai_weddell_curlt[~isnan(erai_weddell_curlt.values)]
    curlt_filtered_no_nans = erai_weddell_curlt_filtered[~isnan(erai_weddell_curlt_filtered.values)]
    [curlt_slope,curlt_intercept,_,curlt_p] \
        = stats.linregress(mdates.date2num(curlt_no_nans.loc[:'2018'].index.date),
                           curlt_no_nans.loc[:'2018'].values)[0:4]
    curlt_trend_line = detrend(curlt_filtered_no_nans,curlt_slope,curlt_intercept,just_return_trend=True)

    # Maud Rise wind stress curl
    mr_curlt_mean = erai_mr_curlt.loc[:'2018'].mean()
    erai_mr_curlt_filtered = erai_mr_curlt.rolling(window=rm_mo,min_periods=rm_min,center=rm_cent).mean()
    mr_curlt_no_nans = erai_mr_curlt[~isnan(erai_mr_curlt.values)]
    mr_curlt_filtered_no_nans = erai_mr_curlt_filtered[~isnan(erai_mr_curlt_filtered.values)]
    [mr_curlt_slope,mr_curlt_intercept,_,mr_curlt_p] \
        = stats.linregress(mdates.date2num(mr_curlt_no_nans.loc[:'2018'].index.date),
                           mr_curlt_no_nans.loc[:'2018'].values)[0:4]
    mr_curlt_trend_line = detrend(mr_curlt_filtered_no_nans,mr_curlt_slope,mr_curlt_intercept,just_return_trend=True)
    
    # print wind stress curl values, to be manually copied above for back-of-the-envelope Ekman salt flux calculation
    mr_curlt_mean_2015_start = erai_mr_curlt.loc['2015-01-01':'2015-05-01'].mean()
    mr_curlt_mean_2016_start = erai_mr_curlt.loc['2016-01-01':'2016-05-01'].mean()
    print('Average Maud Rise wind stress curl through 2014: {0:.2f} N/m^3'.format(erai_mr_curlt.loc[:'2014'].mean()))
    print('Average Maud Rise wind stress curl in Jan-May 2015: {0:.2f} N/m^3'.format(mr_curlt_mean_2015_start))
    print('Average Maud Rise wind stress curl in Jan-May 2016: {0:.2f} N/m^3'.format(mr_curlt_mean_2016_start))

    # storm index
    erai_mr_storm_count_no_nans = erai_mr_storm_count_by_month[~isnan(erai_mr_storm_count_by_month.values)]
    erai_mr_storm_count_filtered_no_nans = erai_mr_storm_count_filtered[~isnan(erai_mr_storm_count_filtered.values)]
    erai_mr_storm_count_mean = erai_mr_storm_count_by_month.loc[:'2018'].mean()

    [erai_mr_storm_count_slope,erai_mr_storm_count_intercept,_,erai_mr_storm_count_p] \
        = stats.linregress(mdates.date2num(erai_mr_storm_count_no_nans.loc[:'2018'].index.date),
                           erai_mr_storm_count_no_nans.loc[:'2018'].values)[0:4]
    erai_mr_storm_count_trend = detrend(erai_mr_storm_count_filtered_no_nans,erai_mr_storm_count_slope,
                                        erai_mr_storm_count_intercept,just_return_trend=True)

    # Maud Rise precipitation
    precip_mean = erai_mr_precip.loc[:'2018'].mean()
    erai_mr_precip_filtered = erai_mr_precip.rolling(window=rm_mo,min_periods=rm_min,center=rm_cent).mean()
    precip_no_nans = erai_mr_precip[~isnan(erai_mr_precip.values)]
    precip_filtered_no_nans = erai_mr_precip_filtered[~isnan(erai_mr_precip_filtered.values)]
    [precip_slope,precip_intercept,_,precip_p] \
        = stats.linregress(mdates.date2num(precip_no_nans.loc[:'2018'].index.date),
                           precip_no_nans.loc[:'2018'].values)[0:4]
    precip_trend_line = detrend(precip_filtered_no_nans,precip_slope,precip_intercept,just_return_trend=True)

    # Eastern Weddell hydrography
    hyd_toi = mr_obs_toi_recent

    ew_tmp_anomalies = e_weddell_hydro_record['temp'][ew_tmp_key]['halfyear_median'].loc[hyd_toi[0]:hyd_toi[1]]
    ew_tmp_anomalies_lower = e_weddell_hydro_record['temp'][ew_tmp_key]['halfyear_quartile_25'].loc[hyd_toi[0]:hyd_toi[1]]
    ew_tmp_anomalies_upper = e_weddell_hydro_record['temp'][ew_tmp_key]['halfyear_quartile_75'].loc[hyd_toi[0]:hyd_toi[1]]

    ew_tmp_depth_range_anomalies \
        = e_weddell_hydro_record['temp'][ew_tmp_depth_range_key]['halfyear_median'].loc[hyd_toi[0]:hyd_toi[1]]
    ew_tmp_depth_range_anomalies_lower \
        = e_weddell_hydro_record['temp'][ew_tmp_depth_range_key]['halfyear_quartile_25'].loc[hyd_toi[0]:hyd_toi[1]]
    ew_tmp_depth_range_anomalies_upper \
        = e_weddell_hydro_record['temp'][ew_tmp_depth_range_key]['halfyear_quartile_75'].loc[hyd_toi[0]:hyd_toi[1]]

    # for violin plots
    ew_tmp_anomalies_mask = e_weddell_hydro_record['temp'][ew_tmp_key]['halfyear_n_obs'] >= 11  # only periods with n_obs >= 11
    ew_tmp_anomalies_mask = ew_tmp_anomalies_mask[ew_tmp_anomalies_mask].index
    ew_tmp_anomalies_grouped \
        = e_weddell_hydro_record['temp'][ew_tmp_key]['halfyear_grouped'][ew_tmp_anomalies_mask].loc[hyd_toi[0]:hyd_toi[1]]
    ew_tmp_anomalies_n_obs \
        = e_weddell_hydro_record['temp'][ew_tmp_key]['halfyear_n_obs'][ew_tmp_anomalies_mask].loc[hyd_toi[0]:hyd_toi[1]]
    ew_tmp_anomalies_mask_small_sample = logical_and(e_weddell_hydro_record['temp'][ew_tmp_key]['halfyear_n_obs'] >= 5,
                                                     e_weddell_hydro_record['temp'][ew_tmp_key]['halfyear_n_obs'] <= 10)
    ew_tmp_anomalies_mask_small_sample = ew_tmp_anomalies_mask_small_sample[ew_tmp_anomalies_mask_small_sample].index
    ew_tmp_anomalies_grouped_small_sample \
        = e_weddell_hydro_record['temp'][ew_tmp_key]['halfyear_grouped'][ew_tmp_anomalies_mask_small_sample].loc[hyd_toi[0]:hyd_toi[1]]

    ew_tmp_depth_range_anomalies_mask = e_weddell_hydro_record['temp'][ew_tmp_depth_range_key]['halfyear_n_obs'] >= 11  # only periods with n_obs >= 11
    ew_tmp_depth_range_anomalies_mask = ew_tmp_depth_range_anomalies_mask[ew_tmp_depth_range_anomalies_mask].index
    ew_tmp_depth_range_anomalies_grouped \
        = e_weddell_hydro_record['temp'][ew_tmp_depth_range_key]['halfyear_grouped'][ew_tmp_depth_range_anomalies_mask].loc[hyd_toi[0]:hyd_toi[1]]
    ew_tmp_depth_range_anomalies_n_obs \
        = e_weddell_hydro_record['temp'][ew_tmp_depth_range_key]['halfyear_n_obs'][ew_tmp_depth_range_anomalies_mask].loc[hyd_toi[0]:hyd_toi[1]]
    ew_tmp_depth_range_anomalies_mask_small_sample \
        = logical_and(e_weddell_hydro_record['temp'][ew_tmp_depth_range_key]['halfyear_n_obs'] >= 5,
                      e_weddell_hydro_record['temp'][ew_tmp_depth_range_key]['halfyear_n_obs'] <= 10)
    ew_tmp_depth_range_anomalies_mask_small_sample \
        = ew_tmp_depth_range_anomalies_mask_small_sample[ew_tmp_depth_range_anomalies_mask_small_sample].index
    ew_tmp_depth_range_anomalies_grouped_small_sample \
        = e_weddell_hydro_record['temp'][ew_tmp_depth_range_key]['halfyear_grouped'][ew_tmp_depth_range_anomalies_mask_small_sample].loc[hyd_toi[0]:hyd_toi[1]]

    [ew_tmp_anomalies_slope_end,ew_tmp_anomalies_intercept_end,_,ew_tmp_anomalies_p_end] \
        = stats.linregress(mdates.date2num(ew_tmp_anomalies.loc['2008-01-01':'2016-06-30'].index.date),
                           ew_tmp_anomalies.loc['2008-01-01':'2016-06-30'].values)[0:4]
    [ew_tmp_anomalies_slope,ew_tmp_anomalies_intercept,_,ew_tmp_anomalies_p] \
        = stats.linregress(mdates.date2num(ew_tmp_anomalies.loc['2002-07-01':'2016-06-30'].index.date),
                           ew_tmp_anomalies.loc['2002-07-01':'2016-06-30'].values)[0:4]

    [ew_tmp_depth_range_anomalies_slope_end,ew_tmp_depth_range_anomalies_intercept_end,_,ew_tmp_depth_range_anomalies_p_end] \
        = stats.linregress(mdates.date2num(ew_tmp_depth_range_anomalies.loc['2008-01-01':'2016-06-30'].index.date),
                           ew_tmp_depth_range_anomalies.loc['2008-01-01':'2016-06-30'].values)[0:4]
    [ew_tmp_depth_range_anomalies_slope,ew_tmp_depth_range_anomalies_intercept,_,ew_tmp_depth_range_anomalies_p] \
        = stats.linregress(mdates.date2num(ew_tmp_depth_range_anomalies.loc['2002-07-01':'2016-06-30'].dropna().index.date),
                           ew_tmp_depth_range_anomalies.loc['2002-07-01':'2016-06-30'].dropna().values)[0:4]



    ##################################################################################################################
    ############################################   FIG. 5 PLOTTING ROUTINE   #########################################
    ##################################################################################################################

    fig = plt.figure(figsize=(7,7.5))
    ED_fig = plt.figure(figsize=(7,8.25))
    fontsize = 6
    bar_edgewidth = 0.5
    xlims = [datetime(1972,1,1),datetime(2018,12,31)]
    xticks = [datetime(y,1,1) for y in arange(1975,2018,5)]
    inset_xlims = [datetime(2002,6,1),datetime(2017,12,31)]

    def fig_5_axis_prep(ylabel='',spines=[],xlabel_top=False,xlabel_bottom=False,xtick_top=False,xtick_bottom=False,
                        ylabel_pos='left',ylabel_color='k',remove_yticks=[],xgrid=False,ygrid=False,ylabel_offset=0.0,
                        set_ax_xlims=xlims,set_ax_xticks=xticks,ylabel_extra_pad=0):
        plt.xlim(set_ax_xlims)
        if set_ax_xticks is not None: plt.xticks(set_ax_xticks)
        plt.gca().xaxis.set_minor_locator(mdates.YearLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.tick_params(axis='x',which='both',top=xtick_top,bottom=xtick_bottom,
                        labeltop=xlabel_top,labelbottom=xlabel_bottom)
        if xgrid and ygrid: plt.gca().grid(linewidth=0.5,alpha=0.3)
        if xgrid: plt.gca().grid(which='minor',axis='x',linewidth=0.25,alpha=0.3)
        if ygrid: plt.gca().grid(which='major',axis='y',linewidth=0.25,alpha=0.3)
        plt.gca().tick_params(axis='both',which='major',labelsize=fontsize)
        [plt.gca().spines[side].set_linewidth(0.5) for side in plt.gca().spines.keys()]
        plt.gca().tick_params(width=0.5)
        if 'top' not in spines: plt.gca().spines['top'].set_visible(False)
        if 'bottom' not in spines: plt.gca().spines['bottom'].set_visible(False)
        for tick in remove_yticks:
            plt.gca().get_yticklabels()[tick].set_visible(False)
            plt.gca().yaxis.get_major_ticks()[tick].set_visible(False)
        if ylabel_pos == 'right':
            plt.gca().yaxis.tick_right()
            plt.gca().yaxis.set_label_position('right')
        plt.gca().patch.set_alpha(0.0)
        if ylabel_pos == 'left':  ylabel_rot = 90
        if ylabel_pos == 'right': ylabel_rot = 90
        plt.ylabel(ylabel,fontsize=fontsize,rotation=ylabel_rot,color=ylabel_color)
        if ylabel_pos == 'left':  plt.gca().get_yaxis().set_label_coords(-0.09 - ylabel_extra_pad,0.5 + ylabel_offset)
        if ylabel_pos == 'right': plt.gca().get_yaxis().set_label_coords(1.09 + ylabel_extra_pad,0.5 + ylabel_offset)
        plt.setp(plt.gca().get_yticklabels(),color=ylabel_color)

    # maximum annual polynya extent and polynya days
    for curr_fig in [fig,ED_fig]:
        plt.figure(curr_fig.number) # set active figure
        curr_fig.add_axes([0.15,0.835,0.70,0.13]) # [x0, y0, width, height] for lower left point (from bottom left of figure)

        # maximum annual polynya extent
        back_bars = plt.bar(polynya_max_all[2].index,polynya_max_all[2].values,width=300,align='center',
                           color='k',edgecolor='k',linewidth=bar_edgewidth,zorder=2)
        top_bars = plt.bar(polynya_max_all[1].index,polynya_max_all[1].values,width=200,align='center',
                           color='0.5',edgecolor='0.5',linewidth=bar_edgewidth,zorder=3)
        front_bars = plt.bar(polynya_max_all[0].index,polynya_max_all[0].values,width=100,align='center',
                            color='0.9',edgecolor='0.9',linewidth=bar_edgewidth,zorder=4)
        plt.ylim(0,100000)
        plt.gca().set_yticks([0,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000])
        plt.gca().set_yticklabels([0,'','20','','40','','60','','80','','≥100'])
        fig_5_axis_prep('Maximum polynya\n' + r'extent (10$^3$ km$^2$)',ylabel_pos='left',ylabel_extra_pad=-0.03,
                        spines=['top','bottom'],xlabel_bottom=True,xtick_bottom=True)

        # missing data
        for year in missing_years:
            plt.plot(datetime(year,7,1),-4000,ls='None',lw=0,
                     marker='*',markersize=4,markerfacecolor='k',markeredgewidth=0,clip_on=False)

        # legend for both polynya max and polynya days
        diamond_color = 'maroon'
        plt.plot([0,0],[NaN,NaN],c='0.9',ls='-',lw=3,label='40% SIC threshold')
        plt.plot([0,0],[NaN,NaN],c='0.5',ls='-',lw=3,label='50% SIC threshold')
        plt.plot([0,0],[NaN,NaN],c='k',ls='-',lw=3,label='60% SIC threshold',
                 marker='D',ms=3,markerfacecolor=diamond_color,markeredgewidth=0)
        plt.plot([0,0],[NaN,NaN],ls='None',lw=0,label='Insufficient winter data',
                 marker='*',ms=4,markerfacecolor='k',markeredgewidth=0)
        plt.legend(ncol=4,scatterpoints=1,handlelength=1.5,loc='lower left',bbox_to_anchor=(0.01,1.0),
                   fontsize=fontsize,frameon=False)

        # polynya days
        right_axis = plt.gca().twinx()
        plt.sca(right_axis)
        polynya_days_all[2].loc['1974'] = 60
        polynya_days_all[2].loc['1975'] = 60
        polynya_days_all[2].loc['1976'] = 60
        plt.scatter(polynya_days_all[2].index[polynya_days_all[2].values > 0],
                    polynya_days_all[2].values[polynya_days_all[2].values > 0],
                    c=diamond_color,s=5,marker='D',zorder=5)
        plt.gca().set_yticks([0,5,10,15,20,25,30,35,40,45,50,55,60])
        plt.gca().set_yticklabels(['0','','10','','20','','30','','40','','50','','≥60'])
        plt.ylim(0,60)
        fig_5_axis_prep('Polynya days',ylabel_pos='right',ylabel_color=diamond_color,ylabel_extra_pad=-0.02,
                        spines=['top','bottom'])

    plt.figure(fig.number) # set active figure

    # SAM
    con_ax_top = fig.add_axes([0.15,0.70,0.70,0.10])
    plt.plot(sam_index_series_filtered.index,sam_index_series_filtered.values,color='k',linestyle='-',
             linewidth=0.75,zorder=3)
    plt.plot([*xlims],[sam_mean,sam_mean],color='k',linestyle='--',linewidth=0.5,zorder=3)
    plt.plot(sam_index_trend.index,sam_index_trend.values,color='k',linestyle=':',linewidth=0.5,zorder=3)
    mpl.rcParams['hatch.linewidth'] = 0.1
    plt.fill_between(sam_index_series_filtered.index,sam_index_series_filtered.values,sam_index_trend,
                     where=sam_index_series_filtered > sam_index_trend,interpolate=True,
                     facecolor='steelblue',alpha=0.5,zorder=2)
    plt.fill_between(sam_index_series_filtered.index,sam_index_series_filtered.values,sam_mean,
                     where=sam_index_series_filtered > sam_mean,interpolate=True,
                     facecolor='none',linewidth=0,hatch='\\\\\\\\\\\\\\',zorder=2)
    plt.ylim([-1.2,1.8])
    plt.yticks([-1.0,0.0,1.0])
    fig_5_axis_prep('SAM index',ylabel_pos='left',ylabel_offset=-0.1,ylabel_extra_pad=-0.03,
                    spines=['top'],xtick_top=True)

    # Weddell Low index and Novo station record
    fig.add_axes([0.15,0.56,0.70,0.16])
    station_names = ['Neumayer (8°W)','SANAE AWS (3°W)','SANAE (2°W)',
                     'Maitri (12°E)','Novolazarevskaya (12°E)','Syowa (40°E)']     # these will be used below
    stations = [reader_neumayer,isd_sanae_aws,isd_sanae_saf,isd_maitri,reader_novo,reader_syowa]
    station_colors = ['#17becf','#bcbd22','#7f7f7f','#d62728','#8c564b','#cc650b']
    selected_stations = [4]
    for s in selected_stations:
        plt.plot(stations[s].rolling(window=rm_mo,min_periods=rm_min,center=rm_cent).mean(),
                 color=station_colors[s],linestyle='-',linewidth=0.75,zorder=4)
    ### Novo
    plt.plot(novo_trend_line.index,novo_trend_line.values,color=station_colors[4],linestyle=':',linewidth=0.5,zorder=3)
    plt.fill_between(novo_filtered_no_nans.index,novo_filtered_no_nans.values,novo_trend_line,
                     where=novo_filtered_no_nans.values < novo_trend_line,
                     interpolate=True,facecolor=station_colors[4],alpha=0.5,zorder=2)
    ### Weddell Low
    plt.plot(erai_wl_index_filtered.index,erai_wl_index_filtered.values,color='midnightblue',linestyle='-',
             linewidth=0.75,zorder=5)
    plt.plot([*xlims],[wl_index_mean,wl_index_mean],color='midnightblue',linestyle='--',linewidth=0.5,zorder=3)
    plt.plot(erai_trend_line.index,erai_trend_line.values,color='k',linestyle=':',linewidth=0.5,zorder=3)
    plt.fill_between(erai_wl_index_filtered_no_nans.index,erai_wl_index_filtered_no_nans.values,erai_trend_line,
                     where=erai_wl_index_filtered_no_nans.values < erai_trend_line,
                     interpolate=True,facecolor='steelblue',alpha=0.5,zorder=2)
    plt.fill_between(erai_wl_index_filtered_no_nans.index,erai_wl_index_filtered_no_nans.values,wl_index_mean,
                     where=erai_wl_index_filtered_no_nans.values < wl_index_mean,
                     interpolate=True,facecolor='none',linewidth=0,hatch='\\\\\\\\\\\\\\',zorder=2)
    plt.yticks([980,982,984,986,988])
    plt.gca().invert_yaxis()
    fig_5_axis_prep('Mean sea-level pressure\n(hPa)',ylabel_pos='right',
                    ylabel_offset=0.05,ylabel_extra_pad=-0.01)

    # winter storm-days count
    fig.add_axes([0.15,0.505,0.70,0.07])
    plt.plot([*xlims],[erai_mr_storm_count_mean,erai_mr_storm_count_mean],
             color='midnightblue',linestyle='--',linewidth=0.5,zorder=5)
    plt.plot(erai_mr_storm_count_trend.index,erai_mr_storm_count_trend.values,
             color='k',linestyle=':',linewidth=0.5,zorder=3)
    for year in range(1970,xlims[1].year+1):
        if str(year) in erai_mr_storm_count_winter.index:
            plt.scatter(datetime(year,6,16),erai_mr_storm_count_winter.loc[str(year)],c='midnightblue',
                        s=2,zorder=2,alpha=1.0)
    plt.plot(erai_mr_storm_count_winter.loc[:'2018'].index + timedelta(days=6*365/12),
             erai_mr_storm_count_winter.loc[:'2018'].values,
             color='midnightblue',linestyle='-',linewidth=0.75,alpha=0.9,zorder=1)
    plt.yticks([4,7,10])
    fig_5_axis_prep('Winter storm-days\n' + r'(month$^{-1}$)',ylabel_pos='left',
                    ylabel_extra_pad=-0.03,ylabel_offset=-0.1)

    # Weddell gyre and Maud Rise wind stress curl
    con_ax_mid = fig.add_axes([0.15,0.42,0.70,0.10])
    plt.plot(erai_mr_curlt_filtered,color='midnightblue',linestyle='--',linewidth=0.75,zorder=5)
    plt.plot([*xlims],[mr_curlt_mean,mr_curlt_mean],color='midnightblue',linestyle='--',linewidth=0.5,zorder=3)
    plt.plot(mr_curlt_trend_line.index,mr_curlt_trend_line.values,color='k',linestyle=':',linewidth=0.5,zorder=3)
    plt.fill_between(mr_curlt_filtered_no_nans.index,mr_curlt_filtered_no_nans.values,mr_curlt_trend_line,
                     where=mr_curlt_filtered_no_nans.values < mr_curlt_trend_line,
                     interpolate=True,facecolor='steelblue',alpha=0.5,zorder=2)
    plt.fill_between(mr_curlt_filtered_no_nans.index,mr_curlt_filtered_no_nans.values,mr_curlt_mean,
                     where=mr_curlt_filtered_no_nans.values < mr_curlt_mean,
                     interpolate=True,facecolor='none',linewidth=0,hatch='\\\\\\\\\\\\\\',zorder=2)
    plt.plot(erai_weddell_curlt_filtered,color='midnightblue',linestyle=(0,(1,1)),linewidth=0.75,zorder=5)
    plt.plot([*xlims],[curlt_mean,curlt_mean],color='midnightblue',linestyle='--',linewidth=0.5,zorder=3)
    plt.plot(curlt_trend_line.index,curlt_trend_line.values,color='k',linestyle=':',linewidth=0.5,zorder=3)
    plt.fill_between(curlt_filtered_no_nans.index,curlt_filtered_no_nans.values,curlt_trend_line,
                     where=curlt_filtered_no_nans.values < curlt_trend_line,
                     interpolate=True,facecolor='steelblue',alpha=0.5-0.2,zorder=2)
    plt.fill_between(curlt_filtered_no_nans.index,curlt_filtered_no_nans.values,curlt_mean,
                     where=curlt_filtered_no_nans.values < curlt_mean,
                     interpolate=True,facecolor='none',linewidth=0,hatch='\\\\\\\\\\\\\\',zorder=2)
    plt.yticks([-2.8,-2.4,-2.0,-1.6])
    plt.gca().invert_yaxis()
    fig_5_axis_prep('Wind stress curl\n({0})'.format(r'10$^{-7}$ N m$^{-3}$'),
                    ylabel_pos='right',ylabel_offset=0.0,ylabel_extra_pad=-0.015,
                    spines=['bottom'],xtick_bottom=True)

    # polynya year vertical bars
    yspan = 0.38
    this_axis_height = 0.10
    for polynya_year in polynya_years:
        plt.gca().axvspan(xmin=datetime(polynya_year,1,1),xmax=datetime(polynya_year,12,31),
                          ymin=0,ymax=yspan / this_axis_height,
                          facecolor=polynya_color,alpha=polynya_year_alpha,zorder=1,clip_on=False)

    # various dummy handles for legend
    plt.plot([NaN,NaN],[0,1],color='midnightblue',linestyle='-',linewidth=0.75,label='Weddell Low')
    for s in selected_stations:
        plt.plot([NaN,NaN],[0,1],color=station_colors[s],linestyle='-',linewidth=0.75,label='Novolazarevskaya Station')
    plt.plot([NaN,NaN],[0,1],'o',color='midnightblue',linestyle='-',linewidth=0.75,markersize=1.25,label='Eastern Weddell')
    plt.plot([NaN,NaN],[0,1],color='midnightblue',linestyle='--',linewidth=0.75,label='Maud Rise')
    plt.plot([NaN,NaN],[0,1],color='midnightblue',linestyle=(0,(1,1)),linewidth=0.75,label='Weddell gyre')
    leg = plt.legend(bbox_to_anchor=(0.0,-0.06),loc='upper left',ncol=5,handlelength=1.7,frameon=False,fontsize=fontsize)

    plt.savefig(current_results_dir + 'figure_5.pdf')
    plt.close(fig)

    ##################################################################################################################
    #########################################   ED FIG. 9 PLOTTING ROUTINE   #########################################
    ##################################################################################################################

    plt.figure(ED_fig.number) # set active figure

    # all relevant Antarctic station records
    con_ax_top = ED_fig.add_axes([0.15,0.69,0.70,0.11])
    for s in range(len(stations)):
        plt.plot(stations[s].rolling(window=rm_mo,min_periods=rm_min,center=rm_cent).mean(),
                 color=station_colors[s],linestyle='-',linewidth=0.75,zorder=2)
    plt.yticks([982,984,986,988,990])
    plt.gca().invert_yaxis()
    fig_5_axis_prep('Queen Maud Land\nmean sea-level pressure\n(hPa)',ylabel_pos='right',
                    spines=['top'],xtick_top=True)

    # E. Weddell precipitation
    con_ax_mid = ED_fig.add_axes([0.15,0.61,0.70,0.08])
    plt.plot(erai_mr_precip_filtered,color='midnightblue',linestyle='-',linewidth=0.75,zorder=5)
    plt.plot([*xlims],[precip_mean,precip_mean],color='midnightblue',linestyle='--',linewidth=0.5,zorder=3)
    plt.plot(precip_trend_line.index,precip_trend_line.values,color='k',linestyle=':',linewidth=0.5,zorder=3)
    plt.fill_between(precip_filtered_no_nans.index,precip_filtered_no_nans.values,precip_mean,
                     where=precip_filtered_no_nans.values < precip_mean,
                     interpolate=True,facecolor='steelblue',alpha=0.5,zorder=2)
    plt.ylim([0.037,0.051])
    plt.yticks([0.041,0.045,0.049])
    plt.gca().set_yticklabels(['4.1','4.5','4.9'])
    fig_5_axis_prep('Eastern Weddell\nprecipitation (cm month$^{-1}$)',ylabel_pos='left',ylabel_extra_pad=-0.02,
                    spines=['bottom'],xtick_bottom=True)

    # polynya year vertical bars
    yspan = 0.19
    this_axis_height = 0.08
    for polynya_year in polynya_years:
        plt.gca().axvspan(xmin=datetime(polynya_year,1,1),xmax=datetime(polynya_year,12,31),
                          ymin=0,ymax=yspan / this_axis_height,
                          facecolor=polynya_color,alpha=polynya_year_alpha,zorder=1,clip_on=False)

    # inset delimiter
    con0a = ConnectionPatch(xyA=((inset_xlims[0] - xlims[0]) / (xlims[1] - xlims[0]),1),
                            xyB=((inset_xlims[0] - xlims[0]) / (xlims[1] - xlims[0]),0),
                            coordsA='axes fraction',coordsB='axes fraction',axesA=con_ax_top,axesB=con_ax_mid,
                            color='k',linewidth=0.5,linestyle=':',alpha=0.5)
    con_ax_mid.add_artist(con0a)
    con0b = ConnectionPatch(xyA=((inset_xlims[1] - xlims[0]) / (xlims[1] - xlims[0]),1),
                            xyB=((inset_xlims[1] - xlims[0]) / (xlims[1] - xlims[0]),0),
                            coordsA='axes fraction',coordsB='axes fraction',axesA=con_ax_top,axesB=con_ax_mid,
                            color='k',linewidth=0.5,linestyle=':',alpha=0.5)
    con_ax_mid.add_artist(con0b)

    # E. Weddell 250-m temp anomaly record
    con_ax_bottom = ED_fig.add_axes([0.16,0.45,0.665,0.12])
    con1 = ConnectionPatch(xyA=((inset_xlims[0] - xlims[0]) / (xlims[1] - xlims[0]),0),xyB=(0,1),
                           coordsA='axes fraction',coordsB='axes fraction',axesA=con_ax_mid,axesB=con_ax_bottom,
                           color='k',linewidth=0.5,linestyle=':')
    con_ax_bottom.add_artist(con1)
    con2 = ConnectionPatch(xyA=((inset_xlims[1] - xlims[0]) / (xlims[1] - xlims[0]),0),xyB=(1,1),
                           coordsA='axes fraction',coordsB='axes fraction',axesA=con_ax_mid,axesB=con_ax_bottom,
                           color='k',linewidth=0.5,linestyle=':')
    con_ax_bottom.add_artist(con2)
    plt.plot([*xlims],[0,0],color='k',linestyle='--',linewidth=0.5,zorder=2)
    violins = plt.violinplot(ew_tmp_anomalies_grouped.values,mdates.date2num(ew_tmp_anomalies_grouped.index.to_pydatetime()),
                             showextrema=False,widths=150*(0.2+(ew_tmp_anomalies_n_obs.values/ew_tmp_anomalies_n_obs.max())))
    for vl in violins['bodies']:
        vl.set_facecolor('steelblue'); vl.set_edgecolor('black'); vl.set_linewidth(0.25)
        vl.set_alpha(0.5); vl.set_zorder(3)
    for idx_dt in ew_tmp_anomalies_grouped_small_sample.index:
        plt.plot(tile(mdates.date2num(idx_dt.to_pydatetime()),len(ew_tmp_anomalies_grouped_small_sample[idx_dt])),
                 ew_tmp_anomalies_grouped_small_sample[idx_dt],
                 ls='',ms=2,markerfacecolor='steelblue',markeredgecolor='none',marker='o',alpha=0.5)
    plt.errorbar(ew_tmp_anomalies.index,ew_tmp_anomalies.values,
                 yerr=[ew_tmp_anomalies.values-ew_tmp_anomalies_lower.values,
                       ew_tmp_anomalies_upper.values-ew_tmp_anomalies.values],
                 fmt='none',ecolor='k',elinewidth=0.5,capsize=2.0,capthick=0.5,zorder=4)
    plt.plot(ew_tmp_anomalies.index,ew_tmp_anomalies.values,color='k',linestyle='-',linewidth=0.75,
             marker='o',markersize=2,zorder=5)
    plt.yticks([-0.3,-0.15,0.0,0.15,0.3])
    plt.ylim([-0.6,0.5])
    fig_5_axis_prep('Eastern Weddell\ntemperature anomaly (°C)\nat {0} m'.format(ew_tmp_depth),
                    ylabel_extra_pad=0.02,ylabel_offset=0.0,
                    ylabel_pos='right',set_ax_xlims=inset_xlims,set_ax_xticks=None,
                    spines=['top'],xtick_top=True)

    # E. Weddell 250-1000 m temp anomaly record
    ylim_above = plt.ylim()
    this_ylim = [-0.4,0.5]
    this_axis_height = 0.12*(this_ylim[1]-this_ylim[0])/(ylim_above[1]-ylim_above[0])
    ED_fig.add_axes([0.16,0.37,0.665,this_axis_height])
    plt.plot([*xlims],[0,0],color='k',linestyle='--',linewidth=0.5,zorder=2)
    violins = plt.violinplot(ew_tmp_depth_range_anomalies_grouped.values,
                             mdates.date2num(ew_tmp_depth_range_anomalies_grouped.index.to_pydatetime()),
                             showextrema=False,widths=150*(0.2+(ew_tmp_depth_range_anomalies_n_obs.values/
                                                                ew_tmp_depth_range_anomalies_n_obs.max())))
    for vl in violins['bodies']:
        vl.set_facecolor('steelblue'); vl.set_edgecolor('black'); vl.set_linewidth(0.25)
        vl.set_alpha(0.5); vl.set_zorder(3)
    for idx_dt in ew_tmp_depth_range_anomalies_grouped_small_sample.index:
        plt.plot(tile(mdates.date2num(idx_dt.to_pydatetime()),
                      len(ew_tmp_depth_range_anomalies_grouped_small_sample[idx_dt])),
                 ew_tmp_depth_range_anomalies_grouped_small_sample[idx_dt],
                 ls='',ms=2,markerfacecolor='steelblue',markeredgecolor='none',marker='o',alpha=0.5)
    plt.errorbar(ew_tmp_depth_range_anomalies.index,ew_tmp_depth_range_anomalies.values,
                 yerr=[ew_tmp_depth_range_anomalies.values-ew_tmp_depth_range_anomalies_lower.values,
                       ew_tmp_depth_range_anomalies_upper.values-ew_tmp_depth_range_anomalies.values],
                 fmt='none',ecolor='k',elinewidth=0.5,capsize=2.0,capthick=0.5,zorder=4)
    plt.plot(ew_tmp_depth_range_anomalies.index,ew_tmp_depth_range_anomalies.values,color='k',linestyle='-',
             linewidth=0.75,marker='o',markersize=2,zorder=5)
    plt.yticks([-0.3,-0.15,0.0,0.15,0.3])
    plt.ylim([-0.4,0.5])
    fig_5_axis_prep('Eastern Weddell\ntemperature anomaly (°C)\nfor {0}-{1} m'.format(*ew_tmp_depth_range),
                    ylabel_offset=-0.05,ylabel_pos='left',set_ax_xlims=inset_xlims,set_ax_xticks=None,
                    spines=['bottom'],xlabel_bottom=True,xtick_bottom=True)

    yspan = 0.45+0.12-0.37
    this_axis_height = this_axis_height
    for polynya_datetime in polynya_datetimes:
        plt.gca().axvspan(xmin=polynya_datetime[0],xmax=polynya_datetime[1],ymin=0,ymax=yspan / this_axis_height,
                          facecolor=polynya_color,alpha=polynya_alpha,zorder=1,clip_on=False)

    # various dummy handles for legend
    for s in range(len(stations)):
        plt.plot([NaN,NaN],[0,1],color=station_colors[s],linestyle='-',linewidth=0.75,label=station_names[s])
    leg = plt.legend(bbox_to_anchor=(-0.0,-0.3),loc='upper left',ncol=3,frameon=False,fontsize=fontsize)

    plt.savefig(current_results_dir + 'ED_figure_9.pdf')
    plt.close(ED_fig)


    ##################################################################################################################
    ######################################   ED TABLE 1 PLOTTING ROUTINE   ###########################################
    ##################################################################################################################

    process_counter = {'pc':1}
    def process(original_series,allow_detrend=True,m=None,b=None,p=None,p_crit=0.05,diff_rm_mo=None,diff_rm_min=None,
                pc=process_counter):
        # args: m=slope,b=intercept,p=p_value

        # subset series and drop NaNs
        original_series = original_series.copy().loc['1972':].dropna()

        # shift monthly series to start on first of each month
        day_start = original_series.index[0].day
        if day_start != 1:
            original_series = original_series.shift(freq=timedelta(days=-1*(day_start-1)))

        # detrend series if requested using detrend() function above
        if allow_detrend:
            if p < p_crit:
                trend = detrend(original_series,m,b,just_return_trend=True)
                original_series = original_series - trend
                print('Detrending series #{0} with p={1:.02f}'.format(pc['pc'],p))

        # create filtered version using uniform parameters, unless specified otherwise
        if diff_rm_mo is None:
            filtered_series = original_series.rolling(window=corr_rm_mo,min_periods=corr_rm_min,center=True).mean()
        else:
            filtered_series = original_series.rolling(window=diff_rm_mo,min_periods=diff_rm_min,center=True).mean()

        pc['pc'] += 1

        return filtered_series

    def corr_lag(series1,series2):
        series_2_lags = corr_lag_range  # in months
        max_corr = 0.0
        max_corr_lag = NaN
        for lag in series_2_lags:
            correlation = series1.corr(series2.shift(lag),method='pearson')
            if abs(correlation) > abs(max_corr):
                max_corr = correlation
                max_corr_lag = lag
        return max_corr, max_corr_lag

    # process series to be correlated (after calculating running mean)
    all_series = [process(sam_index_series,allow_detrend=True,m=sam_slope,b=sam_intercept,p=sam_p),
                  process(erai_wl_index,allow_detrend=True,m=erai_wl_slope,b=erai_wl_intercept,p=erai_wl_p),
                  process(novo,allow_detrend=True,m=novo_slope,b=novo_intercept,p=novo_p),
                  process(erai_mr_curlt,allow_detrend=True,m=mr_curlt_slope,b=mr_curlt_intercept,p=mr_curlt_p),
                  process(erai_weddell_curlt,allow_detrend=True,m=curlt_slope,b=curlt_intercept,p=curlt_p),
                  process(erai_mr_storm_count_by_month,allow_detrend=True,p=erai_mr_storm_count_p,
                          m=erai_mr_storm_count_slope,b=erai_mr_storm_count_intercept,
                          diff_rm_mo=storm_corr_rm_mo,diff_rm_min=storm_corr_rm_min),
                  process(erai_mr_precip,allow_detrend=True,m=precip_slope,b=precip_intercept,p=precip_p)]

    # note: different rolling mean indicated by asterisk: *
    #       detrended indicated by dagger: \textsuperscript{\dag}
    all_series_names = [r'Southern Annular Mode (SAM) index\textsuperscript{\dag}',
                        r'Weddell Low SLP',
                        r'Novolazarevskaya SLP\textsuperscript{\dag}',
                        r'Maud Rise wind stress curl',
                        r'Weddell gyre wind stress curl\textsuperscript{\dag}',
                        r'E. Weddell winter storm days \newline per month*\textsuperscript{\dag}',
                        r'E. Weddell precipitation']

    trend_str = dict()
    trend_str['1'] = '& 1972-2018 \par & {0:.2f} \par & {1:.2f} \par '.format(sam_slope * 365.24 * 10,sam_p)
    trend_str['2'] = '& 1979-2018 \par & {0:.2f} hPa \par & {1:.2f} \par '.format(erai_wl_slope * 365.24 * 10,erai_wl_p)
    trend_str['3'] = '& 1972-2018 \par & {0:.2f} hPa \par & {1:.2f} \par '.format(novo_slope * 365.24 * 10,novo_p)
    trend_str['4'] = '& 1979-2018 \par & {0:.2f}{1} & {2:.2f} \par ' \
                     ''.format(mr_curlt_slope * 365.24 * 10,r'$\cdot$10$^{\textrm{-7}}$ N m$^{\textrm{-3}}$',mr_curlt_p)
    trend_str['5'] = '& 1979-2018 \par & {0:.2f}{1} & {2:.2f} \par ' \
                     ''.format(curlt_slope * 365.24 * 10,r'$\cdot$10$^{\textrm{-7}}$ N m$^{\textrm{-3}}$',curlt_p)
    trend_str['6'] = '& 1979-2018 \par & {0:.2f} \par & {1:.2f} \par '.format(erai_mr_storm_count_slope * 365.24 * 10,
                                                                         erai_mr_storm_count_p)
    trend_str['7'] = '& 1979-2018 \par & {0:.2f} cm month{1} & {2:.2f} \par ' \
                     ''.format(precip_slope * 365.24 * 1000,r'$^{\textrm{-1}}$',precip_p)

    # create correlation matrix of series
    # note: series1 is in header column; series2 is in header row
    table = r'\documentclass[10pt]{article} \usepackage[T1]{fontenc} \usepackage[margin=1.0in]{geometry} ' \
            r'\pagenumbering{gobble} \usepackage{multirow} \usepackage{gensymb} \usepackage{array} ' \
            r'\usepackage{amsmath} \usepackage{makecell} ' \
            r'\usepackage{helvet} \renewcommand{\familydefault}{\sfdefault} \begin{document} ' \
            r'\begin{table} \scriptsize \renewcommand{\arraystretch}{1.9}' \
            r' \begin{tabular}{>{\centering\arraybackslash}m{1.9cm} | ' \
            + r'>{\centering\arraybackslash}m{0.7cm} '*len(all_series) \
            + r' | >{\centering\arraybackslash}m{1.4cm} >{\centering\arraybackslash}m{1.3cm} ' \
              r'>{\centering\arraybackslash}m{0.7cm}} \hline ' \
              r'& 1 & 2 & 3 & 4 & 5 & 6 & 7 & Period & Trend (decade$^{-1}$) & \emph{p} \\ \hline '
    for s1 in range(len(all_series)):
        table = table + r'{0} \par {1} & '.format(s1+1,all_series_names[s1])
        for s2 in range(len(all_series)):
            if s2 >= s1:
                max_corr, max_corr_lag = corr_lag(all_series[s1],all_series[s2])
                table = table + r'{0:.2f} \par ({1}) '.format(max_corr,max_corr_lag)
            if s2+1 < len(all_series): table = table + r'& '
        table = table + trend_str[str(s1+1)]
        table = table + r'\\ '
    table = table + r'\hline \multirow[t]{2}{1.8cm}{\centering E. Weddell 258-m temperature anomaly} & & & & & & & & ' + \
                    r'Jan. 2008 - Jun. 2016 & {0:.2f}\degree C & {1:.2f} \\ & & & & & & & & ' \
                    r'Jul. 2002 - Jun. 2016 & {2:.2f}\degree C & {3:.2f} \\ ' \
                    r''.format(ew_tmp_anomalies_slope_end * 365.24 * 10,ew_tmp_anomalies_p_end,
                               ew_tmp_anomalies_slope * 365.24 * 10,ew_tmp_anomalies_p)
    table = table + r'\hline \multirow[t]{2}{1.8cm}{\centering E. Weddell 250-1000-m temperature anomaly} & & & & & & & & ' + \
                    r'Jan. 2008 - Jun. 2016 & {0:.2f}\degree C & {1:.2f} \\ & & & & & & & & ' \
                    r'Jul. 2002 - Jun. 2016 & {2:.2f}\degree C & {3:.2f} \\ ' \
                    r''.format(ew_tmp_depth_range_anomalies_slope_end * 365.24 * 10,ew_tmp_depth_range_anomalies_p_end,
                               ew_tmp_depth_range_anomalies_slope * 365.24 * 10,ew_tmp_depth_range_anomalies_p)
    table = table + r'\hline \multicolumn{' + r'{0}'.format(len(all_series)+4) + r'}{l}{\makecell[cl]{\\ ' \
            + r'*{0}-month (winter only) running mean applied; '.format(storm_corr_rm_mo) \
            + r'\textsuperscript{\dag}Series detrended ' \
            + r'(original series found to have significant trend, i.e. two-sided \emph{p} < 0.05) \\ }} \\ ' \
            + r'\end{tabular} \end{table} \end{document}'

    text_file = open(current_results_dir + 'ED_table_1.tex','w')
    text_file.write(table)
    text_file.close()

# Extended Data Fig. 1. Locations of observations used to construct hydrographic climatologies for the Maud Rise 
#                       and eastern Weddell regions. 
if plot_ED_fig_1:
    plot_for_paper = True

    # IMPORTANT: requires compiled hydrography pickle from Fig. 2
    e_weddell_obs_all = pickle.load(open(figure_pickle_dir + 'fig_2_e_weddell_obs','rb'))
    e_weddell_obs = e_weddell_obs_all['detailed_info']['psal']['ml_avg']
    mr_obs_all = pickle.load(open(figure_pickle_dir + 'fig_2_mr_obs','rb'))
    mr_obs = mr_obs_all['detailed_info']['psal']['ml_avg']
    mr_center = [-65.0,3.0]       # summit of Maud Rise (65.0°S, 3.0°E)
    mr_hydro_radius = 250         # search radii (km) from Maud Rise
    e_weddell_hydro_radius = 500

    if plot_for_paper:
        epochs = [1970,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019]
        map_dimensions = [1150000,1200000,-65.25,3.0]  # width,height,lat_center,lon_center
        subplot_rows = 5

        for index in range(len(epochs)-1):
            if index == 0: first_subplot = True;  label_contours = False
            else:          first_subplot = False; label_contours = False
            if index == len(epochs) - 1 - 1: last_subplot = True;  add_legend = True
            else:                            last_subplot = False; add_legend = False
            subplot_columns = int(ceil((len(epochs)-1) / subplot_rows))
            which_subplot = [subplot_rows,subplot_columns,index + 1]
            if index + 1 <= subplot_columns: lon_labels = [0,0,1,0]
            else:                            lon_labels = [0,0,0,0]
            if (index * subplot_rows) % (subplot_rows * subplot_columns) == 0: lat_labels = [1,0,0,0]
            else:                                                              lat_labels = [0,0,0,0]

            if first_subplot: master_fig = plt.figure(figsize=(7.5,10))
            plt.gcf().add_subplot(*which_subplot)

            if epochs[index] == epochs[index+1]-1: epoch_title = str(epochs[index])
            else:                                  epoch_title = '{0}-\n{1}'.format(epochs[index],epochs[index+1]-1)

            m = pt.prof_locations_map(current_results_dir,data_dir,e_weddell_obs,map_dimensions,
                                      toi_range=[datetime(epochs[index],1,1),datetime(epochs[index+1]-1,12,31)],
                                      manual_list_of_types=['ship','float','seal'],
                                      manual_labels_for_types=['Shipboard cast','Float profile','Instrumented seal cast'],
                                      manual_markers_for_types=['s','o','^'],manual_marker_open_for_types=[True,False,True],
                                      grid_lats=arange(-80,60,5),grid_lons=arange(-80,50,10),
                                      lon_labels=lon_labels,lat_labels=lat_labels,label_contours=label_contours,
                                      add_epoch_title=epoch_title,fontsize=5,
                                      add_circ_patch=[[mr_center[1],mr_center[0],e_weddell_hydro_radius],
                                                      [mr_center[1],mr_center[0],mr_hydro_radius]],
                                      add_legend=add_legend,legend_pos='outside_right',
                                      create_new_fig=False,use_existing_basemap=None,return_basemap=True,save_as=None)

            pt.prof_locations_map(current_results_dir,data_dir,mr_obs,None,
                                  toi_range=[datetime(epochs[index],1,1),datetime(epochs[index+1]-1,12,31)],
                                  manual_list_of_types=['ship','float','seal'],
                                  manual_markers_for_types=['s','o','^'],manual_marker_open_for_types=[True,False,True],
                                  grid_lats=None,lon_labels=None,lat_labels=None,label_contours=None,
                                  add_epoch_title=None,add_circ_patch=None,add_legend=False,
                                  create_new_fig=False,use_existing_basemap=m,return_basemap=False,save_as=None)

            if last_subplot:
                plt.subplots_adjust(wspace=0.05,hspace=0.05)
                plt.savefig(current_results_dir + 'ED_figure_1.pdf')
                plt.close()

# Extended Data Fig. 2. Sea ice concentration during the 2016 polynya.
# Extended Data Fig. 8. Sea ice concentration during the 2017 polynya.
if plot_ED_figs_2_8:
    # wrapper function
    def make_sic_fig(plot_argo_locs_not_trajs,nan_threshold,save_as,plot_as_subplots,subplot_add_colorbar,
                     subplot_rows,map_params,subplot_fig_size,open_threshold,extent_threshold,start_date,end_date,
                     float_toi,lon_bounds,lat_bounds,only_plot_specific_wmoids,specific_wmoids,wmoid_blacklist,
                     prof_blacklist,grid_lats=arange(-80,60,2),grid_lons=arange(-80,50,10),
                     pad=0.25,spacing=0.2,cbar_bottom=0.125,
                     these_dates=None,include_year_in_date=False,use_only_dmsp=False,boundary_width=1):
        if float_toi is not None:
            plot_floats = True
            argo_gdac_index = ldp.argo_gdac_load_index(argo_gdac_dir)
            float_data_selected = []
            for wmoid in argo_gdac_index['wmoids']:
                if only_plot_specific_wmoids and (wmoid not in specific_wmoids): continue
                this_float_meta = ldp.argo_gdac_float_meta(argo_gdac_index['local_prof_index'],wmoid)

                toi_match = logical_and(this_float_meta['prof_datetimes'] >= float_toi[0],
                                        this_float_meta['prof_datetimes'] <= float_toi[1])
                lon_match = logical_and(this_float_meta['prof_lons'] >= lon_bounds[0],
                                        this_float_meta['prof_lons'] <= lon_bounds[1])
                lat_match = logical_and(this_float_meta['prof_lats'] >= lat_bounds[0],
                                        this_float_meta['prof_lats'] <= lat_bounds[1])
                pos_flag_match = this_float_meta['prof_position_flags'] != 9
                prof_match = logical_and(logical_and(logical_and(toi_match,lon_match),lat_match),pos_flag_match)
                if any(prof_match):
                    prof_cut_indices = range(amin(where(prof_match)),amax(where(prof_match)) + 1)
                    prof_match = zeros(len(prof_match),dtype=bool)
                    prof_match[prof_cut_indices] = True
                    if wmoid in wmoid_blacklist:
                        for pnum in [prof_blacklist[idx] for idx in where(array(wmoid_blacklist) == wmoid)[0]]:
                            prof_match[pnum] = False
                    this_float_data = [wmoid,this_float_meta['prof_lons'][prof_match],
                                       this_float_meta['prof_lats'][prof_match],
                                       this_float_meta['prof_position_flags'][prof_match],
                                       this_float_meta['prof_datetimes'][prof_match],
                                       array(this_float_meta['prof_nums'])[prof_match]]
                    float_data_selected.append(this_float_data)
        else:
            float_data_selected = None
            plot_floats = False

        if these_dates is None:
            date_span = tt.dates_in_range(start_date,end_date)
        else:
            date_span = these_dates
        for index,date in enumerate(date_span):
            if date is None: continue
            print(date)
            if sea_ice_data_avail['amsr2'][date][1] and not use_only_dmsp:
                sic_grid = sea_ice_grids['amsr2']
                sic_field = ldp.load_amsr(sea_ice_data_avail['amsr2'][date][0],regrid_to_25km=False)
            elif sea_ice_data_avail['amsre'][date][1] and not use_only_dmsp:
                sic_grid = sea_ice_grids['amsre']
                sic_field = ldp.load_amsr(sea_ice_data_avail['amsre'][date][0],regrid_to_25km=False)
            elif sea_ice_data_avail['dmsp'][date][1]:
                sic_grid = sea_ice_grids['dmsp']
                sic_field = ldp.load_dmsp(sea_ice_data_avail['dmsp'][date][0],date,use_goddard=True)
            elif sea_ice_data_avail['nimbus5'][date][1]:
                sic_grid = sea_ice_grids['nimbus5']
                sic_field = ldp.load_nimbus5(sea_ice_data_avail['nimbus5'][date][0])
            else:
                continue
            if sum(isnan(sic_field)) == size(sic_field): continue

            if type(map_params[0]) == ndarray: this_map_params = map_params[index]
            else:                              this_map_params = map_params

            if plot_as_subplots:
                if index == 0:
                    first_subplot = True
                else:
                    first_subplot = False
                if index == len(date_span) - 1:
                    last_subplot = True
                else:
                    last_subplot = False
                subplot_columns = int(ceil(len(date_span) / subplot_rows))
                which_subplot = [subplot_rows,subplot_columns,index + 1]
                if index + 1 <= subplot_columns:
                    lon_labels = [0,0,1,0]
                else:
                    lon_labels = [0,0,0,0]
                if (index * subplot_rows) % (subplot_rows * subplot_columns) == 0:
                    lat_labels = [1,0,0,0]
                else:
                    lat_labels = [0,0,0,0]
                pt.sea_ice_argo_spatial(data_dir,date,sic_grid,sic_field,float_data_selected,plot_argo_locs_not_trajs,
                                        None,save_as,current_results_dir,*this_map_params,polynya_grid=None,
                                        open_sic=0,as_subplot=True,subplot_fig_size=subplot_fig_size,
                                        first_subplot=first_subplot,last_subplot=last_subplot,
                                        which_subplot=which_subplot,subplot_add_colorbar=subplot_add_colorbar,
                                        plot_floats=plot_floats,include_year_in_date=include_year_in_date,
                                        grid_lats=grid_lats,grid_lons=grid_lons,
                                        subplot_lon_labels=lon_labels,subplot_lat_labels=lat_labels,
                                        subplot_labelsize=5,pad=pad,spacing=spacing,cbar_bottom=cbar_bottom,
                                        boundary_width=boundary_width,bathy_contours=arange(-3500,-100,500))

    # load Antarctic coastline
    circumant_lons,circumant_lats = gt.establish_coastline(coastline_filename_prefix)

    # load sea ice concentration metadata
    [sea_ice_grids,sea_ice_data_avail,sea_ice_all_dates] = ldp.sea_ice_data_prep(nimbus5_dir,dmsp_v3_dir,dmsp_nrt_dir,
                                                                                 amsre_dir,amsr2_dir,amsr_gridfile,
                                                                                 amsr_areafile,nsidc_ps25_grid_dir)

    # plot individual float locations/profiles, or entire trajectories?
    plot_argo_locs_not_trajs = True

    polynya_2016_subplots = True
    if polynya_2016_subplots:
        save_as = 'ED_figure_2'
        plot_as_subplots = True
        subplot_add_colorbar = True
        subplot_rows = 7
        map_params_main = [650000,500000,-65.0,4.5]    # map boundaries: width, height, lat_center, lon_center
        map_params_south = [650000,500000,-68.25,3.5]
        map_params = concatenate([tile(map_params_main,(25,1)),tile(map_params_south,(5,1))])
        subplot_fig_size = (6.5,6.0*1.17)
        start_date_2016 = (2016,7,24)      # date range for (sea ice) plots
        end_date_2016 = (2016,8,17)
        these_dates = [*tt.dates_in_range(start_date_2016,end_date_2016),
                       (2016,10,27),(2016,10,30),(2016,11,9),(2016,11,18),(2016,11,27)]
        float_toi = [20160101000000,20170101000000] # time span to look for float positions
        lon_bounds = [-20,25] # Argo profile search bounds
        lat_bounds = [-75,-60]
        only_plot_specific_wmoids = True     # if False, look at all floats in dataset
        specific_wmoids = [5904471,5904468]  # specific WMOids to plot (format: list of ints)
        wmoid_blacklist = [7900407]          # manual blacklist for position jump
        prof_blacklist = [110] # profile number blacklist, corresponding to float at same index in wmoid_blacklist
        make_sic_fig(plot_argo_locs_not_trajs,None,save_as,plot_as_subplots,subplot_add_colorbar,subplot_rows,
                     map_params,subplot_fig_size,None,None,None,None,float_toi,
                     lon_bounds,lat_bounds,only_plot_specific_wmoids,specific_wmoids,wmoid_blacklist,prof_blacklist,
                     grid_lats=arange(-80,60,2),grid_lons=arange(-80,50,10),
                     pad=0.25,spacing=None,cbar_bottom=0.125,
                     boundary_width=0.5,these_dates=these_dates,include_year_in_date=True)

    polynya_2017_subplots = True
    if polynya_2017_subplots:
        save_as = 'ED_figure_8'
        plot_as_subplots = True
        subplot_add_colorbar = True
        subplot_rows = 9
        map_params = [1100000,750000,-65.5,4.5]  # map boundaries: width, height, lat_center, lon_center
        subplot_fig_size = (1.1*6.5,1.1*7.0)
        start_date_2017 = (2017,8,30) # date range for (sea ice) plots
        end_date_2017 = (2017,12,2)
        these_dates = tt.dates_in_range(start_date_2017,end_date_2017)[::2]
        float_toi = [20170101000000,20180101000000] # time span to look for float positions
        lon_bounds = [-20,25] # Argo profile search bounds
        lat_bounds = [-75,-60]
        only_plot_specific_wmoids = True     # if False, look at all floats in dataset
        specific_wmoids = [5904471,5904468]  # specific WMOids to plot (format: list of ints)
        wmoid_blacklist = [7900407]          # manual blacklist for position jump
        prof_blacklist = [110] # profile number blacklist, corresponding to float at same index in wmoid_blacklist
        make_sic_fig(plot_argo_locs_not_trajs,None,save_as,plot_as_subplots,subplot_add_colorbar,subplot_rows,
                     map_params,subplot_fig_size,None,None,None,None,float_toi,
                     lon_bounds,lat_bounds,only_plot_specific_wmoids,specific_wmoids,wmoid_blacklist,prof_blacklist,
                     grid_lats=arange(-80,60,3),grid_lons=arange(-80,50,10),
                     pad=0.0,spacing=0.07,cbar_bottom=0.10,
                     boundary_width=0.5,these_dates=these_dates,include_year_in_date=True)

# Extended Data Fig. 5. Winds and wind-induced sea ice divergence during the 2016 polynya.
if plot_ED_fig_5:
    use_erai_pickle = True

    save_as = 'ED_figure_5'
    param_names = 'Estimated ice divergence'
    start_end_dates = [(2016,7,22),(2016,8,17)]
    plot_every_nth_day = 1
    subplot_rows = 6
    map_params = [929500,715000,-65.25,3.5]  # map boundaries: width, height, lat_center, lon_center
    grid_lats = arange(-81,60,2)
    grid_lons = arange(-80,50,10)
    subplot_fig_size = (6.5,7.0)
    subplot_labelsize = 5
    bathy_contours = arange(-3500,-100,500)
    day_string_loc = [0.05,0.95]
    day_string_valign = 'top'
    day_string_halign = 'left'

    # load reanalysis
    toi = [datetime(2016,7,19),datetime(2016,8,20)]
    if not use_erai_pickle:
        erai_daily = ldp.load_ecmwf(era_custom_dir,'erai_daily_weddell.nc',datetime_range=toi)
        erai_daily = erai_daily.drop(['msl','sst','skt','t2m','d2m','q2m','ui10','vi10','div','si10'])
        erai_daily.load()
        pickle.dump(erai_daily,open(figure_pickle_dir + 'ED_fig_5_erai_div_ice_u10_v10','wb'))
    else:
        erai_daily = pickle.load(open(figure_pickle_dir + 'ED_fig_5_erai_div_ice_u10_v10','rb'))

    div_ice_new_units = erai_daily['div_ice'].copy()
    div_ice_new_units = div_ice_new_units * 100.0
    div_ice_new_units.attrs['units'] = '10$^{-7}$ s$^{-1}$'

    # load sea ice concentration metadata
    [sea_ice_grids,sea_ice_data_avail,sea_ice_all_dates] = ldp.sea_ice_data_prep(nimbus5_dir,dmsp_v3_dir,dmsp_nrt_dir,
                                                                                 amsre_dir,amsr2_dir,amsr_gridfile,
                                                                                 amsr_areafile,nsidc_ps25_grid_dir)

    date_span = tt.dates_in_range(start_end_dates[0],start_end_dates[1])[::plot_every_nth_day]
    for index,date in enumerate(date_span):
        day_string = '{0}-{1:02d}-{2:02d}'.format(*date)
        print('Working on {0}...'.format(day_string))
        
        average_these_dates = tt.convert_tuple_to_datetime(date)

        # load SIC field for polynya contour
        sic_grid = sea_ice_grids['amsr2']
        sic_field = ldp.load_amsr(sea_ice_data_avail['amsr2'][date][0],regrid_to_25km=False)

        # subplot planning
        if index == 0: first_subplot = True
        else:          first_subplot = False
        if index == len(date_span) - 1: last_subplot = True
        else:                           last_subplot = False
        subplot_columns = int(ceil(len(date_span) / subplot_rows))
        which_subplot = [subplot_rows,subplot_columns,index + 1]
        if index + 1 <= subplot_columns: lon_labels = [0,0,1,0]
        else:                            lon_labels = [0,0,0,0]
        if (index * subplot_rows) % (subplot_rows * subplot_columns) == 0: lat_labels = [1,0,0,0]
        else:                                                                          lat_labels = [0,0,0,0]

        warnings.filterwarnings('ignore',category=mcbook.mplDeprecation)

        # establish canvas
        if first_subplot: master_fig = plt.figure(figsize=subplot_fig_size)
        plt.gcf().add_subplot(*which_subplot)
        master_fig,m = pt.blank_inset_basemap(*map_params,create_new_fig=False,
                                              lon_labels=lon_labels,lat_labels=lat_labels,
                                              labelsize=subplot_labelsize,grid_color='.2',
                                              boundary_width=0.5,coastline_width=0.5,
                                              grid_lats=grid_lats,grid_lons=grid_lons,
                                              resolution='i')

        # plot sea ice divergence and wind vectors
        pcm = pt.era_field(data_dir,None,None,div_ice_new_units,
                           average_these_dates,*map_params,bathy_contours=bathy_contours,
                           contour_lims=[-10,10],use_cmap='RdBu_r',add_cbar=False,existing_canvas=m,
                           return_pcm=True,
                           add_wind_vectors=[erai_daily['u10'],erai_daily['v10']],wind_vector_scale=100,
                           add_wind_vector_key=last_subplot,wind_vector_key_loc=[1.3,0.65],
                           wind_vector_key_fontsize=subplot_labelsize+2,
                           add_sic_contours=[sic_grid['lons'],sic_grid['lats'],sic_field],sic_contours=[50],
                           add_date=day_string,date_string_loc=day_string_loc,
                           date_string_size=subplot_labelsize+1,
                           date_string_valign=day_string_valign,date_string_halign=day_string_halign)

        # subplot and figure formatting
        if last_subplot:
            plt.tight_layout(h_pad=0.25,w_pad=0.25,rect=(0.02,0.02,0.98,0.98))
            plt.gcf().subplots_adjust(bottom=0.15)
            cbar_ax = plt.gcf().add_axes([0.2,0.075,0.6,0.025])
            cbar = plt.gcf().colorbar(pcm,extend='both',orientation='horizontal',cax=cbar_ax)
            units = div_ice_new_units.attrs['units']
            cbar.ax.text(x=0.5,y=1.25,
                         s='{0} ({1})'.format(param_names,units),
                         size=subplot_labelsize+2,horizontalalignment='center',transform=plt.gca().transAxes)
            cbar.ax.tick_params(labelsize=subplot_labelsize+2)
            cbar.outline.set_linewidth(0.5)
            plt.savefig(current_results_dir + save_as + '.pdf',dpi=300)
            plt.close()

print('Script has finished!')
