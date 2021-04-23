# run anaconda prompt
# type >>> conda activate per2py
# type >>> spyder
# open this file in spyder or idle and run with F5
# v.2021.04.23
# changelog:  circular colorspace for phase plots 

from __future__ import division

# imports
import numpy  as np
import scipy as sp
import pandas as pd
from matplotlib import gridspec
import matplotlib.pyplot as plt
import PlotOptions as plo
import Bioluminescence as blu
import DecayingSinusoid as dsin
import CellularRecording as cr
import settings
import winsound
import glob, os
import matplotlib as mpl
import seaborn as sns
import math
import warnings

# call global variables from module settings.py
settings.init()

# if recording 1 frame/hour, set time_factor to 1, if 1 frame/0.25h, set to 0.25
time_factor = 1

# settings for truncate_t variable - plots and analyze only data from this timepoint (h)
treatment = 0

# plots and analyze only data to this timepoint (h or None), end=None >>> complete dataset
end = None 

#
#
#                Code below this line should not be edited.
#

# supress annoying UserWarning: tight_layout: falling back to Agg renderer
def fxn():
    warnings.warn("tight_layout", UserWarning)
    
PULL_FROM_IMAGEJ = True    # this version is for Trackmate data only.

# list all the datasets
all_inputs=[]
for input_fi in settings.INPUT_FILES:
    all_inputs.append(cr.generate_filenames_dict(settings.INPUT_DIR, input_fi,
                                    PULL_FROM_IMAGEJ, input_ij_extension=settings.INPUT_EXT))

# process the data for every set of inputs
for files_dict in all_inputs:
    # assign all filenames to correct local variables
    data_type = files_dict['data_type']
    input_data = files_dict['input_data']
    input_dir = files_dict['input_dir']
    input_ij_extension = files_dict['input_ij_extension']
    input_ij_file = files_dict['input_ij_file']
    output_cosine = files_dict['output_cosine']
    output_cosine_params = files_dict['output_cosine_params']
    output_detrend = files_dict['output_detrend']
    output_zscore = files_dict['output_zscore']
    output_detrend_smooth = files_dict['output_detrend_smooth']
    output_detrend_smooth_xy = files_dict['output_detrend_smooth_xy']
    output_pgram = files_dict['output_pgram']
    output_phases = files_dict['output_phases']
    pull_from_imagej = files_dict['pull_from_imagej']
    raw_signal = files_dict['raw_signal']
    raw_xy = files_dict['raw_xy']

    # does the actual processing of the data
    # I. IMPORT DATA
    # only perform this step if pull_from_imagej is set to True
    if pull_from_imagej:
        cr.load_imagej_file(input_data, raw_signal, raw_xy, time_factor)

    raw_times, raw_data, locations, header = cr.import_data(raw_signal, raw_xy)
    time_factor = time_factor

    # II. INTERPOLATE MISSING PARTS
    # truncate 0 h and interpolate
    interp_times, interp_data, locations = cr.truncate_and_interpolate(
        raw_times, raw_data, locations, truncate_t=0)

    # III. DETREND USING HP Filter
    #(Export data for presentation of raw tracks with heatmap in Prism.)
    detrended_times, detrended_data, trendlines = cr.hp_detrend(
                                        interp_times, interp_data)

    # IV. SMOOTHING USING EIGENDECOMPOSITION
    # try eigendecomposition, if fail due to inadequate number of values, use savgol
    try:
        denoised_times, denoised_data, eigenvalues = cr.eigensmooth(detrended_times, detrended_data, ev_threshold=0.05, dim=40)
    except IndexError:        
        denoised_times, denoised_data, eigenvalues = cr.savgolsmooth(detrended_times, detrended_data, time_factor=time_factor) 
    
    # TRUNCATE from treatment to end, original function is w/o end variable
    #final_times, final_data, locations = cr.truncate_and_interpolate(denoised_times,
    #                                denoised_data, locations, truncate_t=treatment) # truncate_t=12
    
    final_times, final_data, locations = cr.truncate_and_interpolate_before(denoised_times,
                                    denoised_data, locations, truncate_t=treatment, end=end)
    
    # V. LS PERIODOGRAM TEST FOR RHYTHMICITY
    lspers, pgram_data, circadian_peaks, lspeak_periods, rhythmic_or_not = cr.LS_pgram(final_times, final_data)

    # VI. GET A SINUSOIDAL FIT TO EACH CELL
    # use final_times, final_data
    # use forcing to ensure period within 1h of LS peak period
    sine_times, sine_data, phase_data, refphases, periods, amplitudes, decays, r2s, meaningful_phases =\
         cr.sinusoidal_fitting(final_times, final_data, rhythmic_or_not,
                               fit_times=raw_times, forced_periods=lspeak_periods)
    # get metrics
    circadian_metrics = np.vstack([rhythmic_or_not, circadian_peaks, refphases, periods, amplitudes,
                                   decays, r2s])

    # VII. SAVING ALL COMPONENTS
    timer = plo.laptimer()
    print("Saving data... time: ",)

    # detrended
    cell_ids = header[~np.isnan(header)]
    output_array_det = np.nan*np.ones((len(detrended_times)+1, len(cell_ids)+2))
    output_array_det[1:,0] = detrended_times
    output_array_det[1:,1] = np.arange(len(detrended_times))
    output_array_det[0,2:] = refphases
    output_array_det[1:,2:] = detrended_data
    output_df = pd.DataFrame(data=output_array_det,
            columns = ['TimesH', 'Frame']+list(cell_ids))
    output_df.loc[0,'Frame']='RefPhase'
    output_df.to_csv(output_detrend, index=False)
    del output_df # clear it

    # detrended-denoised
    output_array = np.nan*np.ones((len(final_times)+1, len(cell_ids)+2))
    output_array[1:,0] = final_times
    output_array[1:,1] = np.arange(len(final_times))
    output_array[0,2:] = refphases
    output_array[1:,2:] = final_data
    output_df = pd.DataFrame(data=output_array,
            columns = ['TimesH', 'Frame']+list(cell_ids))
    output_df.loc[0,'Frame']='RefPhase'
    output_df.to_csv(output_detrend_smooth, index=False)
    del output_df # clear it

    # Z-Score
    output_array = np.nan*np.ones((len(final_times)+1, len(cell_ids)+2))
    output_array[1:,0] = final_times
    output_array[1:,1] = np.arange(len(final_times))
    output_array[1:,2:] = sp.stats.zscore(final_data, axis=0, ddof=0)
    output_df = pd.DataFrame(data=output_array,
            columns = ['TimesH', 'Frame']+list(cell_ids))
    output_df.loc[0,'Frame']='RefPhase'
    output_df.loc[0,list(cell_ids)]=refphases
    output_df.to_csv(output_zscore, index=False)
    del output_df # clear it

    # LS Pgram
    output_array = np.nan*np.ones((len(lspers), len(pgram_data[0,:])+1))
    output_array[:,0] = lspers
    output_array[:,1:] = pgram_data
    output_df = pd.DataFrame(data=output_array,
            columns = ['LSPeriod']+list(cell_ids))
    output_df.to_csv(output_pgram, index=False)
    del output_df # clear it

    #sinusoids
    output_array = np.nan*np.ones((len(sine_times), len(cell_ids)+2))
    output_array[:,0] = sine_times
    output_array[:,1] = np.arange(len(sine_times))
    output_array[:,2:] = sine_data
    output_df = pd.DataFrame(data=output_array,
            columns = ['TimesH', 'Frame']+list(cell_ids))
    output_df.to_csv(output_cosine, index=False)
    del output_df

    #phases
    output_array = np.nan*np.ones((len(sine_times), len(cell_ids)+2))
    output_array[:,0] = sine_times
    output_array[:,1] = np.arange(len(sine_times))
    output_array[:,2:] = phase_data
    output_df = pd.DataFrame(data=output_array,
            columns = ['TimesH', 'Frame']+list(cell_ids))
    output_df.to_csv(output_phases, index=False)
    del output_df
    
    #trends
    #trend_array = [np.mean(i) for i in trendlines.T]
    #trend_a = np.asarray(trend_array).reshape((1,len(trend_array)))
    
    if end is None:
        end_t = len(raw_times)
    else:
        end_t = int(end * 1/time_factor)    
    trendlines_trunc = trendlines[int(treatment*1/time_factor):end_t, :]
    trend_array = [np.mean(i) for i in trendlines_trunc.T]
    trend_a = np.asarray(trend_array).reshape((1,len(trend_array)))    
    
    # sinusoid parameters and XY locations
    # this gets the locations for each cell by just giving their mean
    # location and ignoring the empty values. this is a fine approximation.
    locs_fixed = np.zeros([2,len(cell_ids)])
    for idx in range(len(cell_ids)):
        locs_fixed[0, idx] = np.nanmean(locations[:,idx*2])
        locs_fixed[1, idx] = np.nanmean(locations[:,idx*2+1])
    output_array = np.nan*np.ones((9, len(cell_ids)))
    output_array = np.concatenate((circadian_metrics,locs_fixed, trend_a), axis=0)  #updated with trend/mesor
    output_array[2,:] *= 360/2/np.pi #transform phase into 360-degree circular format
    output_df = pd.DataFrame(data=output_array,
            columns = list(cell_ids), index=['Rhythmic','CircPeak','Phase','Period','Amplitude',
                                            'Decay','Rsq', 'X', 'Y', 'Trend'])  #updated with trend/mesor
    output_df.T.to_csv(output_cosine_params, index=True)
    del output_df # clear it
    print(str(np.round(timer(),1))+"s")

    print("Generating and saving plots: ",)
    for cellidx, trackid in enumerate(cell_ids.astype(int)):
        cr.plot_result(cellidx, raw_times, raw_data, trendlines,
                    detrended_times, detrended_data, eigenvalues,
                    final_times, final_data, rhythmic_or_not,
                    lspers, pgram_data, sine_times, sine_data, r2s,
                    settings.INPUT_DIR+f'analysis_output_{settings.timestamp}/', data_type, trackid)
                    #INPUT_DIR, data_type)
    print(str(np.round(timer(),1))+"s")

    print("All data saved. Run terminated successfully for "+data_type+'.\n')
    

#############################################################
#############################################################
####### FINAL PLOTS #########################################
#############################################################
#############################################################    

# to change CT to polar coordinates for polar plotting
# 1h = (2/24)*np.pi = (1/12)*np.pi,   circumference = 2*np.pi*radius
# use modulo to get remainder after integer division
def polarphase(x):                                          
    if x < 24:
        r = (x/12)*np.pi        
    else:
        r = ((x % 24)/12)*np.pi
    return r

# stackoverflow filter outliers - change m as needed (2 is default, 10 filters only most extreme)
def reject_outliers(data, m=10.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)
    return data[s < m]

# https://jakevdp.github.io/PythonDataScienceHandbook/04.07-customizing-colorbars.html
def grayscale_cmap(cmap):
    from matplotlib.colors import LinearSegmentedColormap
    """Return a grayscale version of the given colormap"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    
    # convert RGBA to perceived grayscale luminance
    # cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]
        
    return LinearSegmentedColormap.from_list(cmap.name + "_gray", colors, cmap.N)

# CHOOSE color map of plots
cmap="viridis"
#cmap="YlGnBu"
#cmap= grayscale_cmap(cmap)
 
mydir = settings.INPUT_DIR+f'analysis_output_{settings.timestamp}/'

# LOAD DATA FOR PLOTTING FROM ANALYSIS FOLDER
data = pd.read_csv(glob.glob(f'{mydir}*oscillatory_params.csv')[0])
data_dd = pd.read_csv(glob.glob(f'{mydir}*signal_detrend_denoise.csv')[0])
data_raw = pd.read_csv(glob.glob(f'{mydir}*signal.csv')[0])

### To save figs as vector svg with fonts editable in Corel ###
mpl.use('svg')                                                                          #import matplotlib as mpl
new_rc_params = {"font.family": 'Arial', "text.usetex": False, "svg.fonttype": 'none'}  #to store text as text, not as path in xml-coded svg file
mpl.rcParams.update(new_rc_params)


#########################################################################
####### Single Polar Phase Plot #########################################
#########################################################################

# Use amplitude to filter out outliers or nans
#outlier_reindex = ~(np.isnan(reject_outliers(data[['Amplitude']])))['Amplitude']          # need series of bool values for indexing 
outlier_reindex = ~(np.isnan(data['Amplitude']))

data_filt = data[data.columns[:].tolist()][outlier_reindex]                                  # data w/o amp outliers

phaseseries = data_filt['Phase'].values.flatten()                                           # plot Phase
phase_sdseries = 0.1/(data_filt['Rsq'].values.flatten())                                     # plot R2 related number as width

# NAME
genes = data_filt['Unnamed: 0'].values.flatten().astype(int)                      # plot profile name as color
colorcode = plt.cm.nipy_spectral(np.linspace(0, 1, len(genes)))     # gist_ncar, RdYlBu, Accent check>>> https://matplotlib.org/examples/color/colormaps_reference.html

# LENGTH (AMPLITUDE)
amp = data_filt['Amplitude'].values.flatten()                       # plot filtered Amplitude as length
#amp = 1                                                            # plot arbitrary number if Amp problematic

# POSITION (PHASE)
#phase = [polarphase(i) for i in phaseseries]                        # if phase in in hours (cosinor)
phase = np.radians(phaseseries)                                    # if phase is in degrees (per2py))
#phase = [i for i in phaseseries]                                   # if phase is in radians already

# WIDTH (SD, SEM, R2, etc...)
#phase_sd = [polarphase(i) for i in phase_sdseries]                 # if using CI or SEM of phase, which is in hours
phase_sd = [i for i in phase_sdseries]                              # if using Rsq/R2, maybe adjust thickness 


ax = plt.subplot(111, projection='polar')                                                       #plot with polar projection
bars = ax.bar(phase, amp, width=phase_sd, color=colorcode, bottom=0, alpha=0.8)       #transparency-> alpha=0.5, , rasterized = True, bottom=0.0 to start at center, bottom=amp.max()/3 to start in 1/3 circle
#ax.set_yticklabels([])          # this deletes radial ticks
ax.set_theta_zero_location('N') # this puts CT=0 theta=0 to North - points upwards
ax.set_theta_direction(-1)      #reverse direction of theta increases
ax.set_thetagrids((0, 45, 90, 135, 180, 225, 270, 315), labels=('0', '3', '6', '9', '12', '15', '18', '21'), fontweight='bold', fontsize=12)  #set theta grids and labels, **kwargs for text properties
ax.legend(bars, genes, fontsize=8, bbox_to_anchor=(1.1, 1.1))   # legend needs sequence of labels after object bars which contains sequence of bar plots 
ax.set_xlabel("Circadian phase (h)", fontsize=12)
#plt.title("Individual phases plot", fontsize=14, fontstyle='italic')


### To save as vector svg with fonts editable in Corel ###
plt.savefig(f'{mydir}Phase plot.svg', format = 'svg', bbox_inches = 'tight') #if using rasterized = True to reduce size, set-> dpi = 1000
### To save as bitmap png for easy viewing ###
plt.savefig(f'{mydir}Phase plot.png', bbox_inches = 'tight')
#plt.show()
plt.clf()
plt.close()



###############################################################################################
####### Single Polar Histogram of frequency of phases #########################################
###############################################################################################

N_bins = 47                                                     # how much bins, 23 is for 1 bin per hour, depends on distribution
#colorcode = plt.cm.nipy_spectral(np.linspace(0, 1, N_bins))      #gist_ncar, RdYlBu, Accent check>>> https://matplotlib.org/examples/color/colormaps_reference.html
colorcode = sns.husl_palette(256)[0::int(round(len(sns.husl_palette(256)) / N_bins, 0))]

phase_hist, tick = np.histogram(phase, bins = N_bins, range=(0, 2*np.pi))           # need hist of phase in N bins from 0 to 23h
theta = np.linspace(0.0, 2 * np.pi, N_bins, endpoint=False)     # this just creates number of bins spaced along circle, in radians for polar projection, use as x in histogram
width = (2*np.pi) / N_bins                                      # equal width for all bins that covers whole circle

axh = plt.subplot(111, projection='polar')                                                      #plot with polar projection
bars_h = axh.bar(theta, phase_hist, width=width, color=colorcode, bottom=2, alpha=0.8)          # bottom > 0 to put nice hole in centre

axh.set_yticklabels([])          # this deletes radial ticks
axh.set_theta_zero_location('N') # this puts CT=0 theta=0 to North - points upwards
axh.set_theta_direction(-1)      #reverse direction of theta increases
axh.set_thetagrids((0, 45, 90, 135, 180, 225, 270, 315), labels=('0', '3', '6', '9', '12', '15', '18', '21'), fontweight='bold', fontsize=12)  #set theta grids and labels, **kwargs for text properties
axh.set_xlabel("Circadian phase (h)", fontsize=12)
#plt.title("Phase histogram", fontsize=14, fontstyle='italic')

# calculate vector sum of angles and plot "Rayleigh" vector
a_cos = map(lambda x: math.cos(x), phase)
a_sin = map(lambda x: math.sin(x), phase)
uv_x = sum(a_cos)/len(phase)
uv_y = sum(a_sin)/len(phase)
uv_radius = np.sqrt((uv_x*uv_x) + (uv_y*uv_y))
uv_phase = np.angle(complex(uv_x, uv_y))

# Alternative from http://webspace.ship.edu/pgmarr/Geo441/Lectures/Lec%2016%20-%20Directional%20Statistics.pdf
#v_angle = math.atan((uv_y/uv_radius)/(uv_x/uv_radius))

v_angle = uv_phase     # they are the same 
v_length = uv_radius*max(phase_hist)  # because hist is not (0,1) but (0, N in largest bin), need to increase radius
axh.annotate('',xy=(v_angle, v_length), xytext=(v_angle,0), xycoords='data', arrowprops=dict(width=1, color='black')) #add arrow

### To save as vector svg with fonts editable in Corel ###
plt.savefig(f'{mydir}Histogram_Phase.svg', format = 'svg', bbox_inches = 'tight') #if using rasterized = True to reduce size, set-> dpi = 1000
### To save as bitmap png for easy viewing ###
plt.savefig(f'{mydir}Histogram_Phase.png', bbox_inches = 'tight')
#plt.show()
plt.clf()
plt.close()


###############################################################################################
####### Single Histogram of frequency of periods ##############################################
###############################################################################################

#outlier_reindex_per = ~(np.isnan(reject_outliers(data[['Period']])))['Period'] 
#data_filt_per = data_filt[outlier_reindex_per]
data_filt_per = data_filt.copy()

######## Single Histogram ##########
y = "Period"
x_lab = y
y_lab = "Frequency"
ylim = (0, 0.4)
xlim = (math.floor(data_filt['Period'].min() - 1), math.ceil(data_filt_per['Period'].max() + 1))
suptitle_all = f'{x_lab} vs {y_lab}'
x_coord = xlim[0] + (xlim[1]-xlim[0])/8
y_coord = ylim[1] - (ylim[1]/8)


with warnings.catch_warnings():  # supress annoying UserWarning: tight_layout: falling back to Agg renderer
    warnings.simplefilter("ignore")
    fxn()

    allplot = sns.FacetGrid(data_filt_per)
    allplot = allplot.map(sns.distplot, y, kde=False)  #, Freedman–Diaconis rule
    plt.xlim(xlim)
    #plt.legend(title='Sex')
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.text(x_coord, y_coord, f'n = ' + str(data_filt_per[y].size - data_filt_per[y].isnull().sum()) + '\nmean = ' + str(round(data_filt_per[y].mean(), 3)) + ' ± ' + str(round(data_filt_per[y].sem(), 3)) + 'h')
    #loc = plticker.MultipleLocator(base=4.0) # this locator puts ticks at regular intervals
    #allplot.xaxis.set_major_locator(loc)
    
    ### To save as vector svg with fonts editable in Corel ###
    plt.savefig(f'{mydir}' + '\\' + 'Histogram_Period.svg', format = 'svg', bbox_inches = 'tight')
    ### To save as bitmap png for easy viewing ###
    plt.savefig(f'{mydir}' + '\\' + 'Histogram_Period.png', format = 'png', bbox_inches = 'tight')
    plt.clf()
    plt.close()

############################################################
###### XY coordinates Heatmap of phase #####################
############################################################

# round values to 1 decimal
data_filtr = np.round(data_filt[['X', 'Y', 'Phase']], decimals=3)  # adjust decimal points if needed
# pivot and transpose for heatmap format
df_heat = data_filtr.pivot(index='X', columns='Y', values='Phase').transpose()

suptitle1 = "Phase of PER2 expression"
titleA = "XY SCN coordinates"

df_heat_spec3 = df_heat
                                    
fig, axs = plt.subplots(ncols=2, nrows=1, sharex=False, sharey=False,  gridspec_kw={'width_ratios': [20, 1]}) 
heat1 = sns.heatmap(df_heat_spec3.astype(float), xticklabels=10, yticklabels=10, annot=False, square=True, cbar=True, ax=axs[0], cbar_ax=axs[1], cmap=mpl.colors.ListedColormap(sns.husl_palette(256)))  #tell sns which ax to use  #cmap='coolwarm'  #yticklabels=n >> show every nth label

fig.suptitle(suptitle1, fontsize=12, fontweight='bold')
axs[0].set_title(titleA, fontsize=10, fontweight='bold')
axs[0].set(xlabel='X (pixels)', ylabel='Y (pixels)')

### To save as vector svg with fonts editable in Corel ###
plt.rcParams['svg.fonttype'] = 'none'    #to store text as text, not as path in xml-coded svg file, but avoid bugs that prevent rotation of ylabes when used before setting them
plt.savefig(f'{mydir}Heatmap_XY_Phase.svg', format = 'svg', bbox_inches = 'tight')
### To save as bitmap png for easy viewing ###
plt.savefig(f'{mydir}Heatmap_XY_Phase.png', format = 'png')
plt.clf()
plt.close()


############################################################
###### Dual Heatmap of Raw and Denoised Signal #############
############################################################

sns.set_context("paper", font_scale=1)

x_lab = "time"
suptitle = "Single-cell PER2 expression in the SCN"

# NonSorted Raw data Heatmap
newi1 = data_raw.iloc[:-2, 1:].astype(float).set_index('Frame').index.astype(int)
df_heat_spec1  = data_raw.iloc[:-2, 2:].astype(float).set_index(newi1).transpose()
titleA = "raw"

# NonSorted Detrended traces (Looks good)
data_dd.pop('Frame')
newi2 = (data_dd.iloc[1:, :].astype(float).set_index('TimesH').index.astype(int)) + treatment
df_heat_spec2 = data_dd.iloc[1:, 1:].astype(float).set_index(newi2).transpose()
titleB = "interpolated detrended"
# to plot sorted by phases, use first row in data_dd to sort before transposition

fig, axs = plt.subplots(ncols=5, nrows=1, sharex=False, sharey=False,  gridspec_kw={'width_ratios': [20, 1, 1, 20, 1]}) 
heat1 = sns.heatmap(df_heat_spec1.astype(float), xticklabels=24, yticklabels=False, annot=False, cbar=True, ax=axs[0], cbar_ax=axs[1], cmap="viridis")  #tell sns which ax to use  #cmap='coolwarm'  
heat2 = sns.heatmap(df_heat_spec2.astype(float), xticklabels=24, yticklabels=False, annot=False, cbar=True, ax=axs[3], cbar_ax=axs[4], cmap="viridis")    # yticklabels=10 #use every 10th label

fig.suptitle(suptitle, fontsize=12, fontweight='bold')
axs[0].set_title(titleA, fontsize=10, fontweight='bold')
axs[0].set(xlabel='Time (h)')
axs[2].set_axis_off()  # to put more space between plots so that cbar does not overlap, use 3d axis and set it off
axs[3].set_title(titleB, fontsize=10, fontweight='bold')
axs[3].set(xlabel='Time (h)')

plt.rcParams['svg.fonttype'] = 'none'    #to store text as text, not as path in xml-coded svg file, but avoid bugs that prevent rotation of ylabes when used before setting them
plt.savefig(f'{mydir}Dual_Heatmap.svg', format = 'svg', bbox_inches = 'tight')
plt.savefig(f'{mydir}Dual_Heatmap.png', format = 'png')
plt.clf()
plt.close()

#GRAYSACLE version
fig, axs = plt.subplots(ncols=5, nrows=1, sharex=False, sharey=False,  gridspec_kw={'width_ratios': [20, 1, 1, 20, 1]}) 
heat1 = sns.heatmap(df_heat_spec1.astype(float), xticklabels=24, yticklabels=False, annot=False, cbar=True, ax=axs[0], cbar_ax=axs[1], cmap=cmap)  #tell sns which ax to use  #cmap='coolwarm'  
heat2 = sns.heatmap(df_heat_spec2.astype(float), xticklabels=24, yticklabels=False, annot=False, cbar=True, ax=axs[3], cbar_ax=axs[4], cmap=cmap)    # yticklabels=10 #use every 10th label

fig.suptitle(suptitle, fontsize=12, fontweight='bold')
axs[0].set_title(titleA, fontsize=10, fontweight='bold')
axs[0].set(xlabel='Time (h)')
axs[2].set_axis_off()  # to put more space between plots so that cbar does not overlap, use 3d axis and set it off
axs[3].set_title(titleB, fontsize=10, fontweight='bold')
axs[3].set(xlabel='Time (h)')

plt.rcParams['svg.fonttype'] = 'none'    #to store text as text, not as path in xml-coded svg file, but avoid bugs that prevent rotation of ylabes when used before setting them
plt.savefig(f'{mydir}Dual_Heatmap_gray.svg', format = 'svg', bbox_inches = 'tight')
plt.savefig(f'{mydir}Dual_Heatmap_gray.png', format = 'png')
plt.clf()
plt.close()


#Detrended only
fig, axs = plt.subplots(ncols=2, nrows=1, sharex=False, sharey=False,  gridspec_kw={'width_ratios': [20, 1]}) 
heat2 = sns.heatmap(df_heat_spec2.astype(float), xticklabels=24, yticklabels=False, annot=False, cbar=True, ax=axs[0], cbar_ax=axs[1], cmap=cmap)    # yticklabels=10 #use every 10th label

fig.suptitle(suptitle, fontsize=12, fontweight='bold')
axs[0].set_title(titleB, fontsize=10, fontweight='bold')
axs[0].set(xlabel='Time (h)')

plt.rcParams['svg.fonttype'] = 'none'    #to store text as text, not as path in xml-coded svg file, but avoid bugs that prevent rotation of ylabes when used before setting them
plt.savefig(f'{mydir}Dual_Heatmap_gray2.svg', format = 'svg', bbox_inches = 'tight')
plt.savefig(f'{mydir}Dual_Heatmap_gray2.png', format = 'png')
plt.clf()
plt.close()


print(f'Finished Plots at {mydir}')    
    
winsound.Beep(500, 800)
