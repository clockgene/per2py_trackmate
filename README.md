# per2py_trackmate

First, install miniconda3 from here: 

https://docs.conda.io/en/latest/miniconda.html
Choose Python 3.8 Windows 64bit installer of Miniconda3, download, run the file, select admin, for all, select add to PATH, do not make default (unless you have no previous Python environment)

More information on Anaconda and its environments:
https://towardsdatascience.com/a-guide-to-conda-environments-bc6180fc533
https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html


Create per2py environment and install necessary packages:

click in Windows on Start menu, run Anaconda PowerShell prompt (as admin if possible), type and enter each row and wait for installation:
conda create -n per2py
conda activate per2py
conda env list   		
conda install numpy			
conda install scipy matplotlib ipython jupyter pandas seaborn xlrd
conda install spectrum pywavelets lmfit 

# If you have trouble installing from default channel (i.e. you get some error message), type this instead: 
conda install --channel conda-forge spectrum

# seaborn is for heatmaps, xlrd for xlsx imports in latest versions, pectrum pywavelets lmfit for per2py

# to run per2py in IDLE, type this in conda prompt
conda activate per2py
idle

# to use batch file that autoactivates per2py env and starts idle/spyder, right click on desktop, create New text document (not Word, notepad txt),
# rename it to per2py.bat, open it in notepad and copy paste this:

@echo on
call C:\ProgramData\Miniconda3\Scripts\activate.bat
call activate per2py
call python -m idlelib

# C:\ProgramData\... need to be modified to your actual path to miniconda, which depends on admin/user install!


If you have a fairly powerful PC (8GB+ RAM, quadcore+), you can use Spyder IDE. But it is not necessary to install spyder, youvan  use default IDE named IDLE on older PCs.
First you need to install Spyder, it downgrades jedi for some reason:
conda install spyder  

# to run per2py in spyder, type this in conda prompt
conda activate per2py
spyder

# batch file is the same, but instead> call python -m idlelib, type:
spyder.exe

IMPORTANT:
# For CZ Windows users only:
1. change ; to , in Excel CSV coding, like this: Vyhledej: Region (změnit zemi...) – Další možnosti pro datum… -  Změnit datum, čas,… - Další nastavení – Oddělovač seznamu – tam napiš , místo ;
2. change , to . in Excel: run Excel 2016 or newer (for older version it is a bit different, google it): Soubor - Možnosti - Upřesnit - zruš Použít oddělovač systému a do Oddělovač desetinných míst napiš .
 

# Once everything is installed, unzip source file (e.g. per2py_luminoskan.zip) somewhere (or download from github once ready) and follow corresponding instructions (e.g. in file _manual_per2py_luminoskan.txt) to analyze luminiscence data.



1. Run FIJI Trackmate and analyze images, save table as csv file (only 1 output with both signal and XY coordinates).

2. Open file START.py in IDLE (via conda-per2py environment or using batch file, see installation notes)

3. Change time_factor if needed (for 6 SCN samples usually 1, otherwise use 1/2 for 0.5h, 2 for 2 hours, etc...).

4. Change treatment and end_h variables as needed (i.e. start and end times in hours for analysis of selected time intervals).

5. Run by F5.

6. Icon will appear, click on it and browse to file from the Trackmate analysis and select it, then close the icon by clicking the cross.

7. Plots are saved as png (for viewing) and svg (import to Corel Draw), output tables as csv files, create new timestamped subfolder in the same folder as the input Trackmate file.
