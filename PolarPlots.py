# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 14:06:38 2020

@author: Martin.Sladek
"""

import pandas as pd
import numpy as np
import glob, os
import datetime  as dt
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib as mpl
#import scipy.stats as stats
#from scipy.optimize import curve_fit
#import scipy.signal
#from statsmodels.stats.anova import anova_lm
#import statsmodels.api as sm
from tkinter import filedialog
#from tkinter import *
import tkinter as tk
import seaborn as sns
import math

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

##################### Tkinter button to open analysis folder ################################
def browse_button():    
    global folder_path                      # Allow user to select a directory and store it in global var folder_path
    filename = filedialog.askdirectory()
    folder_path.set(filename)
    print(filename)
    sourcePath = folder_path.get()
    os.chdir(sourcePath)                    # Provide the path here
    root.destroy()                          # close window after pushing button

root = tk.Tk()
folder_path = tk.StringVar()
lbl1 = tk.Label(master=root, textvariable=folder_path)
lbl1.grid(row=0, column=1)
buttonBrowse = tk.Button(text="Browse folder", command=browse_button)
buttonBrowse.grid()
tk.mainloop()
 
mydir = os.getcwd() + '\\'

# LOAD DATA FOR PLOTTING FROM ANALYSIS FOLDER
data = pd.read_csv(glob.glob('*oscillatory_params.csv')[0])
data_dd = pd.read_csv(glob.glob('*signal_detrend_denoise.csv')[0])
data_raw = pd.read_csv(glob.glob('*signal.csv')[0])


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
#plt.title("Rayleigh-style phase plot", fontsize=14, fontstyle='italic')


### To save as vector svg with fonts editable in Corel ###
plt.savefig(f'{mydir}' + '\\' + f'Phase plot.svg', format = 'svg', bbox_inches = 'tight') #if using rasterized = True to reduce size, set-> dpi = 1000
### To save as bitmap png for easy viewing ###
plt.savefig(f'{mydir}' + '\\' + f'Phase plot.png', bbox_inches = 'tight')
#plt.show()
plt.clf()
plt.close()



###############################################################################################
####### Single Polar Histogram of frequency of phases #########################################
###############################################################################################

N_bins = 47                                                     # how much bins, 23 is for 1 bin per hour, depends on distribution
colorcode = plt.cm.nipy_spectral(np.linspace(0, 1, N_bins))      #gist_ncar, RdYlBu, Accent check>>> https://matplotlib.org/examples/color/colormaps_reference.html

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


### To save as vector svg with fonts editable in Corel ###
plt.savefig(f'{mydir}' + '\\' + f'Histogram_Phase.svg', format = 'svg', bbox_inches = 'tight') #if using rasterized = True to reduce size, set-> dpi = 1000
### To save as bitmap png for easy viewing ###
plt.savefig(f'{mydir}' + '\\' + f'Histogram_Phase.png', bbox_inches = 'tight')
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

allplot = sns.FacetGrid(data_filt_per)
allplot = allplot.map(sns.distplot, y, bins=24)  #, bins=48, bins=24 for 1/2h (for 12h axis), bins='sqrt' for Square root of n, None for Freedman–Diaconis rule
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
data_filtr = np.round(data_filt[['X', 'Y', 'Phase']], decimals=1)
# pivot and transpose for heatmap format
df_heat = data_filtr.pivot(index='X', columns='Y', values='Phase').transpose()

suptitle1 = "Phase of PER2 expression"
titleA = "XY SCN coordinates"

df_heat_spec3 = df_heat                                        

fig, axs = plt.subplots(ncols=2, nrows=1, sharex=False, sharey=False,  gridspec_kw={'width_ratios': [20, 1]}) 
heat1 = sns.heatmap(df_heat_spec3.astype(float), xticklabels=10, yticklabels=10, annot=False, square=True, cbar=True, ax=axs[0], cbar_ax=axs[1], cmap="YlGnBu")  #tell sns which ax to use  #cmap='coolwarm'  #yticklabels=n >> show every nth label

fig.suptitle(suptitle1, fontsize=12, fontweight='bold')
axs[0].set_title(titleA, fontsize=10, fontweight='bold')
axs[0].set(xlabel='X (pixels)', ylabel='Y (pixels)')

### To save as vector svg with fonts editable in Corel ###
plt.rcParams['svg.fonttype'] = 'none'    #to store text as text, not as path in xml-coded svg file, but avoid bugs that prevent rotation of ylabes when used before setting them
plt.savefig(f'{mydir}' + '\\' + f'Heatmap_XY_Phase.svg', format = 'svg', bbox_inches = 'tight')
### To save as bitmap png for easy viewing ###
plt.savefig(f'{mydir}' + '\\' + f'Heatmap_XY_Phase.png', format = 'png')
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
newi2 = data_dd.iloc[1:-2, :].astype(float).set_index('TimesH').index.astype(int)  #removes 2 last cols
df_heat_spec2 = data_dd.iloc[1:-2, 2:].astype(float).set_index(newi2).transpose()  #removes 2 last cols
titleB = "interpolated detrended"
# to plot sorted by phases, use first row in data_dd to sort before transposition

fig, axs = plt.subplots(ncols=5, nrows=1, sharex=False, sharey=False,  gridspec_kw={'width_ratios': [20, 1, 1, 20, 1]}) 
heat1 = sns.heatmap(df_heat_spec1.astype(float), xticklabels=24, yticklabels=False, annot=False, cbar=True, ax=axs[0], cbar_ax=axs[1], cmap="YlGnBu")  #tell sns which ax to use  #cmap='coolwarm'  
heat2 = sns.heatmap(df_heat_spec2.astype(float), xticklabels=24, yticklabels=False, annot=False, cbar=True, ax=axs[3], cbar_ax=axs[4], cmap="coolwarm")    # yticklabels=10 #use every 10th label

fig.suptitle(suptitle, fontsize=12, fontweight='bold')
axs[0].set_title(titleA, fontsize=10, fontweight='bold')
axs[0].set(xlabel='Time (h)')
axs[2].set_axis_off()  # to put more space between plots so that cbar does not overlap, use 3d axis and set it off
axs[3].set_title(titleB, fontsize=10, fontweight='bold')
axs[3].set(xlabel='Time (h)')

plt.rcParams['svg.fonttype'] = 'none'    #to store text as text, not as path in xml-coded svg file, but avoid bugs that prevent rotation of ylabes when used before setting them
plt.savefig(f'{mydir}' + '\\' + f'Dual_Heatmap.svg', format = 'svg', bbox_inches = 'tight')
plt.savefig(f'{mydir}' + '\\' + f'Dual_Heatmap.png', format = 'png')
plt.clf()
plt.close()

print('Finished: ' + mydir.split('\\')[-2])