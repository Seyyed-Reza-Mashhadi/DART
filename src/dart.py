# DART
# Required libraries to import

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator
import matplotlib as mpl
import fileinput, sys
import shutil
import os
import matplotlib.patches as mpatches
from scipy.interpolate import interp1d
from IPython.display import clear_output
import plotly.express as px
import math
import geopandas as gpd
import pyproj 
from shapely.geometry import Point
import simplekml
import base64
import tempfile
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import MaxNLocator, FuncFormatter   # Import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit

# get_ipython().run_line_magic('matplotlib', 'notebook')
# %matplotlib qt5 
# %matplotlib notebook


##########################################################################################################
# GENERAL USEFUL FUNCTIONS

def get_borehole_names(input_dirc):
    """Getting the name of boreholes by getting the name of files in a directory. 
    You can use the following codes for excluding specific boreholes if needed: a=a.iloc[i1::].drop([i2,i3]).reset_index(drop=True)"""
    names = pd.DataFrame([x for x in os.listdir(input_dirc)])
    names = names[names[0].str.contains(".txt") == False].reset_index(drop=True)
    names.rename(columns={0: 'Borehole'}, inplace=True)
    names['Borehole'] = names['Borehole'].astype('string')
    return names

def merge_txt_files(txt_files_list):
    """ For merging text files including similar structured dataframes, useful for combining the data from various projects for general statistics of NMR parameters"""
    DATA_2=pd.DataFrame()
    for i in txt_files_list:
        DATA = pd.DataFrame(pd.read_csv(i,sep='\t'))
        DATA_2 = pd.concat([DATA, DATA_2], axis=0, join='outer', ignore_index=False)
    return DATA_2

def get_azimuth(start_point, end_point):
    """This function calculates the azimuth of a line based on coordinates of start and end points in a unique UTM zone"""
    return (90 - math.degrees(math.atan2((end_point[1] - start_point[1]), (end_point[0] - start_point[0])))) % 360

def add_coordinates (dataframe, input_file):
    """Dataframe with BNMR data and the input file for XYZ coordinates will be the inputs for this function."""
    XYZ = pd.DataFrame(pd.read_csv(input_file, sep='\t',encoding="ISO-8859-1"))
    XYZ.dropna(inplace=True)
    XYZ['Borehole'] = XYZ['Borehole'].astype('string')
    dataframe['Borehole'] = dataframe['Borehole'].astype('string')
    dataframe = dataframe.merge(XYZ,how='left',on=['Borehole'])
    if 'depth' in dataframe.columns:
        dataframe['elevation'] = dataframe.Z - dataframe.depth
    return dataframe

def remove_duplicates(geolabled_database): 
    geolabled_database.drop_duplicates(subset=['depth', 'Borehole'], keep=False, inplace=True, ignore_index=True)
    # basically ensuring that there are only one BNMR measurement for each depth so it can remove any sort of duplicates
    return geolabled_database

def check_NonNumeric_columns (dataframe, params):
    """This function removes duplicates from the defined columns/parameters and shows non-numeric values in the column which is useful in many cases.
    The params is an array. For geology and many data is suggested to firstly check ['TOP','BOTTOM']. Then, apply it for your desired parameter/column."""
    df = dataframe.copy()
    df.dropna(subset=params, inplace=True)
    for par in params:
        counter = 0
        if len(df[pd.to_numeric(df[par], errors='coerce').isna()])!=0:
            counter +=1
            print(par, 'column has non_numeric values: \n')
            print(df[pd.to_numeric(df[par], errors='coerce').isna()])
            df.drop(df[pd.to_numeric(df[par], errors='coerce').isna()].index, inplace=True)
    if counter != 0:
        print('\nAll non numeric values are dropped from the dataframe')
    else:
        print('There is no non-numeric values')
    return df

def hist_kde_cfd_plot(df, par, x_label='', grid=True, output_dirc=''):
    sorted_values = df[par].sort_values()
    cdf = np.linspace(0, 1, len(sorted_values))
    fig2, ax2 = plt.subplots(1, 3, figsize=(9, 3))
    labels_dict = dict({'totalf':'Total Porosity (%)','soe': 'SOE', 'noise':'Noise (%)', 'mlT2': 'T$_{2ML}$ (s)',
                   'clayf':'Clay-bound Porosity (%)','capf':'Capillary-bound Porosity (%)','freef':'Free Porosity (%)',
                   'immobile': 'Immobile Porosity (%)'})
    x_label=par
    if par in labels_dict.keys():
        x_label = labels_dict[par]
    ax = ax2[0]    # histogram
    ax.set_xlabel(x_label)
    ax.hist(df[par], color='crimson')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram')
    ax.grid(grid)
    ax = ax2[1]     # KDE
    log_scale=False
    if par in ['soe','mlT2']:
        log_scale=True
    sns.kdeplot(df[par], log_scale=log_scale, linewidth=2, color='m', ax=ax)
    ax.set_xlabel(x_label)
    ax.grid(grid)
    ax.set_title('KDE')
    ax = ax2[2]    # CDF
    ax.plot(sorted_values, cdf, linewidth=2, color='darkorange')
    ax.set_xlabel(x_label)
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('CDF')
    ax.set(ylim=(0,1.05))
    ax.grid(grid)
    if par in ['soe','mlT2']:
        ax.set(xscale='log')
    if par in ['totalf','clayf','capf','freef','immobile']:
        ax2[0].set(xlim=(0,None))
        ax2[1].set(xlim=(0,None))
        ax2[2].set(xlim=(0,None))
    plt.tight_layout()
    if output_dirc!='':
        plt.savefig(output_dirc+'\\hist_kde_cfd_plot_'+par+'.png')
        clear_output()
    return fig2

# SECTION PLOTTING FUNCTIONS

def section_plot (dataframe, variable, names_column='Borehole', plot_type='curve', point_start=None, point_end=None, buffer=None, filled=True, secondary_axis=False, max_value= None, min_value= None, 
                  color=None, decimal=0, log_scale=False, scaleX = None, interpolation='none', label_offset=0.3, label_fontsize=12, xlim=None, ylim=None, title=None, color_scale_label=None,  output_dirc='' , output_name=''):
    """ variables to define: 
    1. dataframe: the dataframe format including X,Y,Z and the desired variable.
    2. names_column: the string indicating the points identification ID / column name. It can be Borehole, Geophysical Station, etc.
    3. variable to plot: the name (string) of the desired variable to plot
    4. plot type ('curve' or 'filled_bars')
    5. point_start, point_end (optional): If you have exact coordinates for your profile, you can insert. Otherwise, you will provide two points in the popping up window. 
    6. filled=True (for the curve type)
    7. Secondary=True (for the curve type to be plotted in the reverse direction)
    8. max_value, min_value 
    9. color (or pallete for filled bars)
    10. decimal: number of decimals for the lable of the curve type variable
    11. logarithmic_scale = True/False
    12. interpolation: interpolation method for filled_bars plots.
    13. xlim, ylim: limits of the section plot if needed
    14. title: figure title
    15. scaleX: scaling factor to set the width of each single curve/filled_bars plot. Default values for curve and filled_bars are 0.03 and 0.015, respectively.
    if you do not have topography data yet and want to plot sections, create a Z column with zero values to avoid errors.
    """
    
    if scaleX == None:
        if plot_type == 'filled_bars':
            scaleX = 0.015  
        else:
            scaleX = 0.03  
    if color==None:
        if plot_type == 'curve':
            color = 'b'
        elif plot_type == 'filled_bars':
            color = 'viridis'
    if secondary_axis ==True:
        secondary_index = -1
    else:
        secondary_index = +1
    # introducing the data file
    par = variable
    database_area = dataframe.copy()
    global buffer_distance, sect_st, sect_end
    if (point_start == None) or (point_end == None):
        buffer_distance = None
        # Create Tkinter window
        plt.rcParams['text.antialiased'] = True
        root = tk.Tk()
        root.title("Point Recorder")
        root.geometry("1200x1000")
        class PointRecorder:
            def __init__(self, master):
                self.master = master
                self.start_point = None
                self.end_point = None
                self.buffer_distance = None  # Buffer distance attribute
                self.clicks = 0
                self.click_handler_id = None  # Store the ID of the click event handler
                # Buffer distance entry
                self.buffer_label = tk.Label(master, text="Enter Buffer Distance (m):")
                self.buffer_label.pack()
                self.buffer_entry = tk.Entry(master)
                self.buffer_entry.pack()
                # OK button
                self.ok_button = tk.Button(master, text="OK", command=self.on_ok_click, state=tk.DISABLED)
                self.ok_button.pack()
                # Create a plot
                self.fig, self.ax = plt.subplots(figsize=(9,9))
                self.ax.set_aspect('equal')
                self.ax.set_xlabel('X')
                self.ax.set_ylabel('Y')
                # Plot the sect_df_all data
                self.ax.scatter(database_area.X, database_area.Y, color='k', s=15)
                # Adding annotations to points
                for index, row in database_area.iterrows():
                    plt.annotate(f"{row['Borehole']}", (row['X'], row['Y']),  textcoords="offset points", 
                                xytext=(5,5), ha='center', color='k', fontsize=12, weight='normal')
                # Set gridlines
                self.ax.grid(color='gray', linestyle='--', linewidth=0.5)
                # Set axis tick label format to integers
                self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                self.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                # Set axis tick label formatter to display integers
                self.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
                self.ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0f}'.format(y)))
                # Create a canvas
                self.canvas = FigureCanvasTkAgg(self.fig, master=master)
                self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                # Bind click event
                self.click_handler_id = self.canvas.mpl_connect("button_press_event", self.record_point)
                # Bind <Return> key press event to trigger on_ok_click
                master.bind("<Return>", lambda event: self.on_ok_click())
            def record_point(self, event):
                if self.clicks >= 2:
                    self.canvas.mpl_disconnect(self.click_handler_id)    # Disconnect the click event handler after getting two clicks
                    return
                if event.dblclick:
                    # Ignore double-clicks
                    return
                x, y = event.xdata, event.ydata
                if x is not None and y is not None:
                    if self.clicks == 0:
                        self.start_point = (x, y)
                    elif self.clicks == 1:
                        self.end_point = (x, y)
                    self.clicks += 1
                    self.ax.plot(x, y, 'ro')   # Draw a point on the plot
                    self.canvas.draw()
                    if self.clicks == 2:    # Check if both points are captured
                        self.enable_ok_button()
            def enable_ok_button(self):
                self.ok_button.config(state=tk.NORMAL)
            def on_ok_click(self):
                global sect_st, sect_end, buffer_distance
                sect_st = self.start_point
                sect_end = self.end_point
                buffer_distance = self.buffer_entry.get()
                try:
                    buffer_distance = float(buffer_distance)  # Store buffer_distance in global variable
                    root.destroy()
                except ValueError:
                    tk.messagebox.showerror("Error", "Please enter a valid number for buffer distance.")
        app = PointRecorder(root)   # Create PointRecorder instance
        root.mainloop()   # Run the Tkinter event loop
    else:
        sect_st = point_start
        sect_end = point_end
        if buffer==None:
            buffer_distance = float(input('Set the buffer distance (m): ')) # define the buffer in meters
        else:
            buffer_distance = buffer
    print('Section Start: ', np.round(sect_st,0))
    print('Section End: ', np.round(sect_end,0))
    print('Buffer Distance (m): ', buffer_distance)
    profile_length = ((sect_end[0]-sect_st[0])**2 + (sect_end[1]-sect_st[1])**2)**0.5
    print('Profile Length (m): ', np.round(profile_length,0))
    # create a dataframe for the section
    database_area['Xper']=0.0
    database_area['Yper']=0.0
    # distance from the line
    database_area ['Buffer1'] = np.abs((sect_end[1] - sect_st[1]) * database_area['X'] - (sect_end[0] - sect_st[0]) * database_area['Y'] + sect_end[0] * sect_st[1] - sect_end[1] * sect_st[0]) / ((sect_end[1] - sect_st[1])**2 + (sect_end[0] - sect_st[0])**2)**0.5
    # distance from the center of the diameter of "profile length  + 2*buffer distance"
    database_area['Buffer2'] = np.abs(((database_area['X'] - ((sect_end[0]+sect_st[0])/2))**2 + (database_area['Y'] - ((sect_end[1]+sect_st[1])/2))**2)**0.5)
    # applying the criteria or the buffer zone
    sect_df = database_area[(database_area.Buffer1 <= buffer_distance) & (database_area.Buffer2 <=((profile_length+buffer_distance)/2))].reset_index(drop=True)
    sect_df = sect_df.copy()
    # Calculating the intersection of the perpendicular line from the point to the section
    m = (sect_end[1] - sect_st[1]) / (sect_end[0] - sect_st[0]) if sect_end[0] - sect_st[0] != 0 else np.inf
    if m == np.inf:
        sect_df.Xper = sect_st[0]
        sect_df.Yper = sect_df.Y
    elif m == 0:
        sect_df.Xper = sect_df.X
        sect_df.Yper = sect_st[1]
    else:
        b = sect_st[1] - m * sect_st[0]
        m_perp = -1 / m
        b_perp = sect_df.Y - m_perp * sect_df.X
        sect_df.Xper = (b_perp - b) / (m - m_perp)
        sect_df.Yper = m * sect_df.Xper + b
    # creating the distance column for plotting
    sect_df['Distance'] = ((sect_df.Xper-sect_st[0])**2 + (sect_df.Yper-sect_st[1])**2)**0.5
    # plotting the boreholes, selected versus not selected
    fig0, ax0 = plt.subplots(figsize=(9,9))
    ax0.scatter(database_area.X,database_area.Y, color='r', s=10)
    ax0.scatter(sect_df['X'],sect_df['Y'], color='b', s=10)
    ax0.ticklabel_format(axis='x', style='plain', useOffset=False, useMathText=True, scilimits=(-3, 8))
    ax0.ticklabel_format(axis='y', style='plain', useOffset=False, useMathText=True, scilimits=(-3, 8))
    ax0.tick_params(axis='y', rotation=0)
    ax0.set_aspect('equal')
    ax0.plot([sect_st[0],sect_end[0]],[sect_st[1],sect_end[1]], color='k')
    ax0.scatter(sect_df['Xper'],sect_df['Yper'], color='g', s=10)
    for bhs in sect_df[names_column].unique():
        ax0.annotate(bhs, (sect_df[sect_df[names_column]==bhs].X.mean(),sect_df[sect_df[names_column]==bhs].Y.mean()), fontsize=7)  
    # Plotting the section    
    if min_value == None:
        min_value = sect_df[par].min()
    if max_value == None:
        max_value = sect_df[par].max()
    # normalization of values for plotting
    if (max_value-min_value)!=0:
        if log_scale == False:
            sect_df['Xplot'] = sect_df.Distance + secondary_index*(profile_length*scaleX)*(sect_df[par]-min_value)/(max_value-min_value)
        else:
            sect_df['Xplot'] = sect_df.Distance + secondary_index*(profile_length*scaleX)*(np.log10(sect_df[par])-np.log10(min_value))/(np.log10(max_value)-np.log10(min_value))
    else:
        sect_df['Xplot_1'] = sect_df.Distance
    sect_df['Yplot'] = -1 * sect_df.depth + sect_df.Z
    sect_df = sect_df.replace([np.inf, -np.inf], np.nan).dropna()   
    # creating the figure
    fig, ax = plt.subplots(figsize=(14,6))
    if plot_type == 'curve':
        for bhs in sect_df[names_column].unique():
            if filled == True:
                ax.fill_betweenx(sect_df[sect_df[names_column]==bhs].Yplot , x1= sect_df[sect_df[names_column]==bhs].Distance, x2= sect_df[sect_df[names_column]==bhs].Xplot,
                              alpha=1, color='k', facecolor=color, linewidth=0.5)
            else:
                ax.plot(sect_df[sect_df[names_column]==bhs].Xplot, sect_df[sect_df[names_column]==bhs].Yplot, linestyle='-',
                    alpha=1, color=color,linewidth=1)   
            ax.plot([sect_df[sect_df[names_column]==bhs].Distance.max(),sect_df[sect_df[names_column]==bhs].Distance.max()],
                     [sect_df[sect_df[names_column]==bhs].Z.min(),sect_df[sect_df[names_column]==bhs].Z.min()-sect_df[sect_df[names_column]==bhs].depth.max()],
                     '-k', linewidth=0.5)
        sect_df.sort_values(by='Distance', ascending=False, kind='mergesort', inplace=True)
        ax.plot(sect_df.Distance,sect_df.Z,linestyle='-', color='k', label='Topography')
        ax.plot(sect_df.Distance,sect_df.Z,'ko', markersize=2)
        for bhs in sect_df[names_column].unique():
            ax.annotate(bhs, (sect_df[sect_df[names_column]==bhs].Distance.iloc[0], sect_df[sect_df[names_column]==bhs].Z.iloc[0]+label_offset), fontsize=label_fontsize, rotation=90, horizontalalignment='center', 
                         verticalalignment='bottom')
            ax.plot([sect_df[sect_df[names_column]==bhs].Distance.max(),
                    sect_df[sect_df[names_column]==bhs].Distance.max()+secondary_index*(scaleX*profile_length)*0.5,
                    sect_df[sect_df[names_column]==bhs].Distance.max()+secondary_index*(scaleX*profile_length)*1.0],
                     [sect_df[sect_df[names_column]==bhs].Z.min()-sect_df[sect_df[names_column]==bhs].depth.max(),
                      sect_df[sect_df[names_column]==bhs].Z.min()-sect_df[sect_df[names_column]==bhs].depth.max(),
                     sect_df[sect_df[names_column]==bhs].Z.min()-sect_df[sect_df[names_column]==bhs].depth.max()],
                    marker='|', markersize=5, linestyle='-', color='k', linewidth=0.5)
            if log_scale == False: 
                ax.annotate(np.format_float_positional(min_value, precision=3), xy=(sect_df[sect_df[names_column]==bhs].Distance.max(), sect_df[sect_df[names_column]==bhs].Z.min()-sect_df[sect_df[names_column]==bhs].depth.max()), 
                        xytext=(0,-2), textcoords='offset points', ha='center', va='top', rotation=90, fontsize=6)
                ax.annotate(np.format_float_positional((max_value-min_value)/2,precision=3), xy=(sect_df[sect_df[names_column]==bhs].Distance.max()+(0.5)*secondary_index*(scaleX*profile_length), sect_df[sect_df[names_column]==bhs].Z.min()-sect_df[sect_df[names_column]==bhs].depth.max()), 
                        xytext=(0,-2), textcoords='offset points', ha='center', va='top', rotation=90, fontsize=6)
                ax.annotate(np.format_float_positional(max_value, precision=3), xy=(sect_df[sect_df[names_column]==bhs].Distance.max()+1.0*secondary_index*(scaleX*profile_length), sect_df[sect_df[names_column]==bhs].Z.min()-sect_df[sect_df[names_column]==bhs].depth.max()), 
                        xytext=(0,-2), textcoords='offset points', ha='center', va='top', rotation=90, fontsize=6)
            else:
                ax.annotate(np.format_float_positional(min_value, precision=3), xy=(sect_df[sect_df[names_column]==bhs].Distance.max(), sect_df[sect_df[names_column]==bhs].Z.min()-sect_df[sect_df[names_column]==bhs].depth.max()), 
                        xytext=(0,-2), textcoords='offset points', ha='center', va='top', rotation=90, fontsize=6)
                ax.annotate(np.format_float_positional(min_value*10**((np.log10(max_value/min_value))/2),precision=3), xy=(sect_df[sect_df[names_column]==bhs].Distance.max()+(0.5)*secondary_index*(scaleX*profile_length), sect_df[sect_df[names_column]==bhs].Z.min()-sect_df[sect_df[names_column]==bhs].depth.max()), 
                        xytext=(0,-2), textcoords='offset points', ha='center', va='top',rotation=90,fontsize=6)
                ax.annotate(np.format_float_positional(max_value, precision=3), xy=(sect_df[sect_df[names_column]==bhs].Distance.max()+1.0*secondary_index*(scaleX*profile_length), sect_df[sect_df[names_column]==bhs].Z.min()-sect_df[sect_df[names_column]==bhs].depth.max()), 
                        xytext=(0,-2), textcoords='offset points',ha='center', va='top', rotation=90, fontsize=6)
    if plot_type == 'filled_bars':
        for bhs in sect_df[names_column].unique():
            xmin = sect_df[sect_df[names_column]==bhs].Distance.max()-0.5*secondary_index*(scaleX*profile_length)
            xmax = sect_df[sect_df[names_column]==bhs].Distance.max()+0.5*secondary_index*(scaleX*profile_length)
            ymin = sect_df[sect_df[names_column]==bhs].Yplot.min()
            ymax = sect_df[sect_df[names_column]==bhs].Yplot.max()
            extent = [xmin, xmax, ymin, ymax]
            data_plot = np.array([sect_df[sect_df[names_column]==bhs].Distance, sect_df[sect_df[names_column]==bhs].Yplot, sect_df[sect_df[names_column]==bhs][par]], dtype=float).T
            # Plot additional data using imshow plot
            im = ax.imshow(data_plot[:, 2].reshape((len(data_plot),1)), cmap=color, vmin=min_value, vmax=max_value,
                           interpolation=interpolation, extent=extent, aspect='auto')
        sect_df.sort_values(by='Distance', ascending=False, kind='mergesort', inplace=True)
        ax.plot(sect_df.Distance,sect_df.Z,linestyle='-', color='k', label='Topography')
        ax.plot(sect_df.Distance,sect_df.Z,'ko', markersize=2) 
        for bhs in sect_df[names_column].unique():
            plt.annotate(bhs, (sect_df[sect_df[names_column]==bhs].Distance.iloc[0], sect_df[sect_df[names_column]==bhs].Z.iloc[0]+label_offset), fontsize=label_fontsize, rotation=90, horizontalalignment='center', verticalalignment='bottom')
            ax.plot([sect_df[sect_df[names_column]==bhs].Distance.min() - 0.5*(profile_length*scaleX),
                     sect_df[sect_df[names_column]==bhs].Distance.min() - 0.5*(profile_length*scaleX)],
                     [sect_df[sect_df[names_column]==bhs].Z.max(),
                      sect_df[sect_df[names_column]==bhs].Yplot.min()],'-k', linewidth=0.5)
            ax.plot([sect_df[sect_df[names_column]==bhs].Distance.min() + 0.5*(profile_length*scaleX),
                     sect_df[sect_df[names_column]==bhs].Distance.min() + 0.5*(profile_length*scaleX)],
                     [sect_df[sect_df[names_column]==bhs].Z.max(),
                      sect_df[sect_df[names_column]==bhs].Yplot.min()],'-k', linewidth=0.5)
            ax.plot([sect_df[sect_df[names_column]==bhs].Distance.min() - 0.5*(profile_length*scaleX),
                     sect_df[sect_df[names_column]==bhs].Distance.min() + 0.5*(profile_length*scaleX)],
                     [sect_df[sect_df[names_column]==bhs].Z.max(),
                      sect_df[sect_df[names_column]==bhs].Z.max()],'-k', linewidth=0.5)
            ax.plot([sect_df[sect_df[names_column]==bhs].Distance.min() - 0.5*(profile_length*scaleX),
                     sect_df[sect_df[names_column]==bhs].Distance.min() + 0.5*(profile_length*scaleX)],
                     [sect_df[sect_df[names_column]==bhs].Yplot.min(),
                      sect_df[sect_df[names_column]==bhs].Yplot.min()],'-k', linewidth=0.5)
        # setting the x and y limits of the plot
        xmin, xmax = 0, profile_length
        ymin, ymax = sect_df.Yplot.min(),sect_df.Z.max()
        if ymax < 0:
            ymax *= 0.9
        else:
            ymax *= 1.1
        if ymin < 0:
            ymin *= 1.1
        else:
            ymin *= 0.9 
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.05, aspect=80 )  # Adjust size and pad as needed
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(color_scale_label)

    # Common Lables and Annotations
    ax.set(xlabel='Distance (m)')
    ax.set(ylabel='Elevation (m)')
    ax.set_title(title, y=1.1)
    if xlim == None:
        ax.set_xlim(0 , profile_length)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # Labeling the section Orientation
    # Calculating the Azimuth of the section
    azimuth = (90 - math.degrees(math.atan2((sect_end[1] - sect_st[1]), (sect_end[0] - sect_st[0])))) % 360
    az_threshold = 10
    if (azimuth >= 360-az_threshold) | (azimuth <= 0+az_threshold):
        TEXT1, TEXT2 = 'S', 'N'
    elif (azimuth >= 180-az_threshold) & (azimuth <= 180+az_threshold):
        TEXT1, TEXT2 = 'N', 'S'
    elif (azimuth >= 90-az_threshold) & (azimuth <= 90+az_threshold):
        TEXT1, TEXT2 = 'W', 'E'
    elif (azimuth >= 270-az_threshold) & (azimuth <= 270+az_threshold):
        TEXT1, TEXT2 = 'E', 'W'
    elif (azimuth > 0+az_threshold) & (azimuth < 90-az_threshold):
        TEXT1, TEXT2 = 'SW', 'NE'
    elif (azimuth > 90+az_threshold) & (azimuth < 180-az_threshold):
        TEXT1, TEXT2 = 'NW', 'SE'
    elif (azimuth > 270+az_threshold) & (azimuth < 360-az_threshold):
        TEXT1, TEXT2 = 'SE', 'NW'
    elif (azimuth > 180+az_threshold) & (azimuth < 270-az_threshold):
        TEXT1, TEXT2 = 'NE', 'SW'
    ax.text(x=0.95,y=0.95, s=TEXT2, weight='bold', fontsize=16, transform=fig.transFigure)
    ax.text(x=0.05,y=0.95, s=TEXT1, weight='bold', fontsize=16, transform=fig.transFigure)
    fig = plt.gcf()
    bdf = buffer_distance
    if output_dirc !='':
            plt.savefig(str(output_dirc + '\\' + output_name+'.png'), bbox_inches='tight', dpi=600)
    return fig0, fig, sect_st, sect_end, bdf

def section_plot_geo (dataframe, geo_symbology_file, names_column = 'Borehole', variable='ROCKSYMBOL', point_start=None, point_end=None, buffer=None, scaleX = None,
                      xlim=None, ylim=None, title=None, label_offset=0.3, label_fontsize=12, output_dirc='', output_name=''):
    """ This function is for plotting geological section. Note that you may need to change some of the predefined variables.
    variables to define:
    1. dataframe: the dataframe format including X,Y,Z, TOP,and BOTTOM.
    2. names_column: the string indicating the points identification ID / column name. It can be Borehole, Borehole_ID, etc.
    3. variable: variable to plot, the name (string) of the desired variable to plot i.e. ROCKSYMBOL, etc. (be careful that name of variable should be the same in geosymbology file and in dataframe)
    4. point_start, point_end: If you have exact coordinates for your profile, you can insert. Otherwise, you will provide two points in the popping up window. 
    5. geo_symbology_file: txt file with geological color codes and variable names (Two columns should exist: 'COLORCODE' and variable)
    6. scaleX = scaling factor used to set the width of the plot. default value is 0.007.
    7. xlim, ylim: limits of the section plot if needed.
    8. title: figure title
    if you do not have topography data yet and want to plot sections, create a Z column with zero values to avoid errors"""
    if scaleX == None:
        scaleX = 0.007  
    # introducing the data file
    par = variable
    database_area = dataframe.copy()
    global buffer_distance, sect_st, sect_end
    if (point_start == None) or (point_end == None):
        buffer_distance = None
        # Create Tkinter window
        plt.rcParams['text.antialiased'] = True
        root = tk.Tk()
        root.title("Point Recorder")
        root.geometry("1200x1000")
        class PointRecorder:
            def __init__(self, master):
                self.master = master
                self.start_point = None
                self.end_point = None
                self.buffer_distance = None  # Buffer distance attribute
                self.clicks = 0
                self.click_handler_id = None  # Store the ID of the click event handler
                # Buffer distance entry
                self.buffer_label = tk.Label(master, text="Enter Buffer Distance (m):")
                self.buffer_label.pack()
                self.buffer_entry = tk.Entry(master)
                self.buffer_entry.pack()
                # OK button
                self.ok_button = tk.Button(master, text="OK", command=self.on_ok_click, state=tk.DISABLED)
                self.ok_button.pack()
                # Create a plot
                self.fig, self.ax = plt.subplots(figsize=(9,9))
                self.ax.set_aspect('equal')
                self.ax.set_xlabel('X')
                self.ax.set_ylabel('Y')
                # Plot the sect_df_all data
                self.ax.scatter(database_area.X, database_area.Y, color='k', s=15)
                # Adding annotations to points
                for index, row in database_area.iterrows():
                    plt.annotate(f"{row['Borehole']}", (row['X'], row['Y']),  textcoords="offset points", 
                                xytext=(5,5), ha='center', color='k', fontsize=12, weight='normal')
                # Set gridlines
                self.ax.grid(color='gray', linestyle='--', linewidth=0.5)
                # Set axis tick label format to integers
                self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                self.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                # Set axis tick label formatter to display integers
                self.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
                self.ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0f}'.format(y)))
                # Create a canvas
                self.canvas = FigureCanvasTkAgg(self.fig, master=master)
                self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                # Bind click event
                self.click_handler_id = self.canvas.mpl_connect("button_press_event", self.record_point)
                # Bind <Return> key press event to trigger on_ok_click
                master.bind("<Return>", lambda event: self.on_ok_click())
            def record_point(self, event):
                if self.clicks >= 2:
                    self.canvas.mpl_disconnect(self.click_handler_id)    # Disconnect the click event handler after getting two clicks
                    return
                if event.dblclick:
                    # Ignore double-clicks
                    return
                x, y = event.xdata, event.ydata
                if x is not None and y is not None:
                    if self.clicks == 0:
                        self.start_point = (x, y)
                    elif self.clicks == 1:
                        self.end_point = (x, y)
                    self.clicks += 1
                    self.ax.plot(x, y, 'ro')   # Draw a point on the plot
                    self.canvas.draw()
                    if self.clicks == 2:    # Check if both points are captured
                        self.enable_ok_button()
            def enable_ok_button(self):
                self.ok_button.config(state=tk.NORMAL)
            def on_ok_click(self):
                global sect_st, sect_end, buffer_distance
                sect_st = self.start_point
                sect_end = self.end_point
                buffer_distance = self.buffer_entry.get()
                try:
                    buffer_distance = float(buffer_distance)  # Store buffer_distance in global variable
                    root.destroy()
                except ValueError:
                    tk.messagebox.showerror("Error", "Please enter a valid number for buffer distance.")
        app = PointRecorder(root)   # Create PointRecorder instance
        root.mainloop()   # Run the Tkinter event loop
    else:
        sect_st = point_start
        sect_end = point_end
        if buffer==None:
            buffer_distance = float(input('Set the buffer distance (m): ')) # define the buffer in meters
        else:
            buffer_distance = buffer

    print('Section Start: ', np.round(sect_st,0))
    print('Section End: ', np.round(sect_end,0))
    print('Buffer Distance (m): ', buffer_distance)
    profile_length = ((sect_end[0]-sect_st[0])**2 + (sect_end[1]-sect_st[1])**2)**0.5
    print('Profile Length (m): ', np.round(profile_length,0))
    # create a dataframe for the section
    database_area['Xper']=0.0
    database_area['Yper']=0.0
    # distance from the line
    database_area ['Buffer1'] = np.abs((sect_end[1] - sect_st[1]) * database_area['X'] - (sect_end[0] - sect_st[0]) * database_area['Y'] + sect_end[0] * sect_st[1] - sect_end[1] * sect_st[0]) / ((sect_end[1] - sect_st[1])**2 + (sect_end[0] - sect_st[0])**2)**0.5
    # distance from the center of the diameter of "profile length  + 2*buffer distance"
    database_area['Buffer2'] = np.abs(((database_area['X'] - ((sect_end[0]+sect_st[0])/2))**2 + (database_area['Y'] - ((sect_end[1]+sect_st[1])/2))**2)**0.5)
    # applying the criteria or the buffer zone
    sect_df = database_area[(database_area.Buffer1 <= buffer_distance) & (database_area.Buffer2 <=((profile_length+buffer_distance)/2))].reset_index(drop=True)
    sect_df = sect_df.copy()
    # Calculating the intersection of the perpendicular line from the point to the section
    m = (sect_end[1] - sect_st[1]) / (sect_end[0] - sect_st[0]) if sect_end[0] - sect_st[0] != 0 else np.inf
    if m == np.inf:
        sect_df.Xper = sect_st[0]
        sect_df.Yper = sect_df.Y
    elif m == 0:
        sect_df.Xper = sect_df.X
        sect_df.Yper = sect_st[1]
    else:
        b = sect_st[1] - m * sect_st[0]
        m_perp = -1 / m
        b_perp = sect_df.Y - m_perp * sect_df.X
        sect_df.Xper = (b_perp - b) / (m - m_perp)
        sect_df.Yper = m * sect_df.Xper + b
    # creating the distance column for plotting
    sect_df['Distance'] = ((sect_df.Xper-sect_st[0])**2 + (sect_df.Yper-sect_st[1])**2)**0.5

    # plotting the boreholes, selected versus not selected
    fig0, ax0 = plt.subplots(figsize=(9,9))
    ax0.scatter(database_area.X,database_area.Y, color='r', s=10)
    ax0.scatter(sect_df['X'],sect_df['Y'], color='b', s=10)
    ax0.ticklabel_format(axis='x', style='plain', useOffset=False, useMathText=True, scilimits=(-3, 8))
    ax0.ticklabel_format(axis='y', style='plain', useOffset=False, useMathText=True, scilimits=(-3, 8))
    ax0.tick_params(axis='y', rotation=0)
    ax0.set_aspect('equal')
    ax0.plot([sect_st[0],sect_end[0]],[sect_st[1],sect_end[1]], color='k')
    ax0.scatter(sect_df['Xper'],sect_df['Yper'], color='g', s=10)
    for bhs in sect_df[names_column].unique():
        ax0.annotate(bhs, (sect_df[sect_df[names_column]==bhs].X.mean(),sect_df[sect_df[names_column]==bhs].Y.mean()), fontsize=7)
        
    # Plotting the geo-section    
    # creating the figure
    fig, ax = plt.subplots(figsize=(14,6))
    Geo_Symbology = pd.DataFrame(pd.read_csv(geo_symbology_file, sep='\t',encoding= 'unicode_escape'))
    symbol_color_dict = dict(zip(Geo_Symbology[variable], Geo_Symbology.COLORCODE))
    # loop over boreholes and intervals
    for bhs in sect_df[names_column].unique():
        for i in range(len(sect_df[sect_df[names_column]==bhs])):
            symbol = sect_df[sect_df[names_column]==bhs][variable].iloc[i]
            symbol_color = symbol_color_dict.get(symbol, '#F8F8FF')
            xx = [sect_df[sect_df[names_column]==bhs].Distance.iloc[i] - (profile_length*scaleX), sect_df[sect_df[names_column]==bhs].Distance.iloc[i] + (profile_length*scaleX),
                  sect_df[sect_df[names_column]==bhs].Distance.iloc[i] + (profile_length*scaleX), sect_df[sect_df[names_column]==bhs].Distance.iloc[i] - (profile_length*scaleX)]
            yy = [sect_df[sect_df[names_column]==bhs].Z.iloc[i] - sect_df[sect_df[names_column]==bhs].TOP.iloc[i], sect_df[sect_df[names_column]==bhs].Z.iloc[i] - sect_df[sect_df[names_column]==bhs].TOP.iloc[i],
                  sect_df[sect_df[names_column]==bhs].Z.iloc[i] - sect_df[sect_df[names_column]==bhs].BOTTOM.iloc[i], sect_df[sect_df[names_column]==bhs].Z.iloc[i] - sect_df[sect_df[names_column]==bhs].BOTTOM.iloc[i]]
            ax.fill(xx, yy, color=symbol_color, linewidth=0, label=Geo_Symbology[variable][Geo_Symbology.COLORCODE==symbol_color].to_string(index=False))      
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(),bbox_to_anchor=(1.1, 1), loc="upper right",title="LEGEND", labelspacing = 1, fontsize=8, fancybox=True, borderpad=1)
            ax.set_ylabel('Elevation (m)')
            ax.set_xlabel('Distance (m)')

    # Labeling and Boreder of logs            
    for bhs in sect_df[names_column].unique():
        plt.annotate(bhs, (sect_df[sect_df[names_column]==bhs].Distance.iloc[0], sect_df[sect_df[names_column]==bhs].Z.iloc[0]+label_offset), fontsize=label_fontsize, rotation=90, horizontalalignment='center', verticalalignment='bottom')
        ax.plot([sect_df[sect_df[names_column]==bhs].Distance.min() - (profile_length*scaleX),
                 sect_df[sect_df[names_column]==bhs].Distance.min() - (profile_length*scaleX)],
                 [sect_df[sect_df[names_column]==bhs].Z.min(),
                  sect_df[sect_df[names_column]==bhs].Z.min()-sect_df[sect_df[names_column]==bhs].BOTTOM.max()],'-k', linewidth=0.5)
        ax.plot([sect_df[sect_df[names_column]==bhs].Distance.min() + (profile_length*scaleX),
                 sect_df[sect_df[names_column]==bhs].Distance.min() + (profile_length*scaleX)],
                 [sect_df[sect_df[names_column]==bhs].Z.min(),
                  sect_df[sect_df[names_column]==bhs].Z.min()-sect_df[sect_df[names_column]==bhs].BOTTOM.max()],'-k', linewidth=0.5)
        ax.plot([sect_df[sect_df[names_column]==bhs].Distance.min() - (profile_length*scaleX),
                 sect_df[sect_df[names_column]==bhs].Distance.min() + (profile_length*scaleX)],
                 [sect_df[sect_df[names_column]==bhs].Z.min(),
                  sect_df[sect_df[names_column]==bhs].Z.min()],'-k', linewidth=0.5)
        ax.plot([sect_df[sect_df[names_column]==bhs].Distance.min() - (profile_length*scaleX),
                 sect_df[sect_df[names_column]==bhs].Distance.min() + (profile_length*scaleX)],
                 [sect_df[sect_df[names_column]==bhs].Z.min()-sect_df[sect_df[names_column]==bhs].BOTTOM.max(),
                  sect_df[sect_df[names_column]==bhs].Z.min()-sect_df[sect_df[names_column]==bhs].BOTTOM.max()],'-k', linewidth=0.5)

    # Plotting the topography. Note that it uses the selected boreholes not the trace of the section
    sect_df.sort_values(by='Distance', ascending=False, kind='mergesort', inplace=True)
    ax.plot(sect_df.Distance,sect_df.Z,linestyle='-', color='k', label='Topography')
    ax.plot(sect_df.Distance,sect_df.Z,'ko', markersize=2)
    if xlim == None:
        xmin, xmax = 0, profile_length
        ax.set_xlim(xmin,xmax)
    else:
        ax.set_xlim(xlim)
    
    if ylim == None:
        ymin, ymax = min(sect_df.Z - sect_df.BOTTOM),sect_df.Z.max()
        if ymax < 0:
            ymax *= 0.9
        else:
            ymax *= 1.1
        if ymin < 0:
            ymin *= 1.1
        else:
            ymin *= 0.9 
        ax.set_ylim(ymin, ymax)
    else:
        ax.set_ylim(ylim) 

    # Common Lables and Annotations
    ax.set(xlabel='Distance (m)')
    ax.set(ylabel='Elevation (m)')
    ax.set_title(title, y=1.1)
    
    #Labeling the section Orientation
    azimuth = (90 - math.degrees(math.atan2((sect_end[1] - sect_st[1]), (sect_end[0] - sect_st[0])))) % 360
    threshold = 10
    if (azimuth >= 360-threshold) | (azimuth <= 0+threshold):
        TEXT1, TEXT2 = 'S', 'N'
    elif (azimuth >= 180-threshold) & (azimuth <= 180+threshold):
        TEXT1, TEXT2 = 'N', 'S'
    elif (azimuth >= 90-threshold) & (azimuth <= 90+threshold):
        TEXT1, TEXT2 = 'W', 'E'
    elif (azimuth >= 270-threshold) & (azimuth <= 270+threshold):
        TEXT1, TEXT2 = 'E', 'W'
    elif (azimuth > 0+threshold) & (azimuth < 90-threshold):
        TEXT1, TEXT2 = 'SW', 'NE'
    elif (azimuth > 90+threshold) & (azimuth < 180-threshold):
        TEXT1, TEXT2 = 'NW', 'SE'
    elif (azimuth > 270+threshold) & (azimuth < 360-threshold):
        TEXT1, TEXT2 = 'SE', 'NW'
    elif (azimuth > 180+threshold) & (azimuth < 270-threshold):
        TEXT1, TEXT2 = 'NE', 'SW'
    ax.text(x=0.95,y=0.95, s=TEXT2, weight='bold', fontsize=16, transform=fig.transFigure)
    ax.text(x=0.05,y=0.95, s=TEXT1, weight='bold', fontsize=16, transform=fig.transFigure)   
    fig = plt.gcf()
    bdf = buffer_distance
    if output_dirc !='':
            plt.savefig(str(output_dirc + '\\' + output_name+'.png'), bbox_inches='tight', dpi=600)
    return fig0, fig, sect_st, sect_end, bdf
    
def section_plot_water_partitioning_data (dataframe, names_column='Borehole', point_start=None, point_end=None, buffer=None, max_value= 100, min_value= 0, scaleX = 0.03, 
                                          xlim=None, ylim=None, title=None, label_offset=0.3, label_fontsize=12, output_dirc='', output_name='water_partitioning_data'):
    """ variables to define: 
    1. dataframe: the dataframe including X,Y,Z and the desired variables.
    2. names_column: the string indicating the points identification ID / column name. 'Borehole' is the default.
    3. point_start, point_end (optional): If you have exact coordinates for your profile, you can insert. Otherwise, you will provide two points in the popping up window. 
    4. max_value, min_value 
    5. xlim, ylim: limits of the section plot if needed
    6. title
    7. scaleX: scaling factor to set the width of each single curve/filled_bars plot. Default values for curve and filled_bars are 0.03 and 0.015, respectively.
    """
    # introducing the data file
    database_area = dataframe.copy()
    global buffer_distance, sect_st, sect_end
    if (point_start == None) or (point_end == None):
        buffer_distance = None
        # Create Tkinter window
        plt.rcParams['text.antialiased'] = True
        root = tk.Tk()
        root.title("Point Recorder")
        root.geometry("1200x1000")
        class PointRecorder:
            def __init__(self, master):
                self.master = master
                self.start_point = None
                self.end_point = None
                self.buffer_distance = None  # Buffer distance attribute
                self.clicks = 0
                self.click_handler_id = None  # Store the ID of the click event handler
                # Buffer distance entry
                self.buffer_label = tk.Label(master, text="Enter Buffer Distance (m):")
                self.buffer_label.pack()
                self.buffer_entry = tk.Entry(master)
                self.buffer_entry.pack()
                # OK button
                self.ok_button = tk.Button(master, text="OK", command=self.on_ok_click, state=tk.DISABLED)
                self.ok_button.pack()
                # Create a plot
                self.fig, self.ax = plt.subplots(figsize=(9,9))
                self.ax.set_aspect('equal')
                self.ax.set_xlabel('X')
                self.ax.set_ylabel('Y')
                # Plot the sect_df_all data
                self.ax.scatter(database_area.X, database_area.Y, color='k', s=15)
                # Adding annotations to points
                for index, row in database_area.iterrows():
                    plt.annotate(f"{row['Borehole']}", (row['X'], row['Y']),  textcoords="offset points", 
                                xytext=(5,5), ha='center', color='k', fontsize=12, weight='normal')
                # Set gridlines
                self.ax.grid(color='gray', linestyle='--', linewidth=0.5)
                # Set axis tick label format to integers
                self.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                self.ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                # Set axis tick label formatter to display integers
                self.ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '{:.0f}'.format(x)))
                self.ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0f}'.format(y)))
                # Create a canvas
                self.canvas = FigureCanvasTkAgg(self.fig, master=master)
                self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                # Bind click event
                self.click_handler_id = self.canvas.mpl_connect("button_press_event", self.record_point)
                # Bind <Return> key press event to trigger on_ok_click
                master.bind("<Return>", lambda event: self.on_ok_click())
            def record_point(self, event):
                if self.clicks >= 2:
                    self.canvas.mpl_disconnect(self.click_handler_id)    # Disconnect the click event handler after getting two clicks
                    return
                if event.dblclick:
                    # Ignore double-clicks
                    return
                x, y = event.xdata, event.ydata
                if x is not None and y is not None:
                    if self.clicks == 0:
                        self.start_point = (x, y)
                    elif self.clicks == 1:
                        self.end_point = (x, y)
                    self.clicks += 1
                    self.ax.plot(x, y, 'ro')   # Draw a point on the plot
                    self.canvas.draw()
                    if self.clicks == 2:    # Check if both points are captured
                        self.enable_ok_button()
            def enable_ok_button(self):
                self.ok_button.config(state=tk.NORMAL)
            def on_ok_click(self):
                global sect_st, sect_end, buffer_distance
                sect_st = self.start_point
                sect_end = self.end_point
                buffer_distance = self.buffer_entry.get()
                try:
                    buffer_distance = float(buffer_distance)  # Store buffer_distance in global variable
                    root.destroy()
                except ValueError:
                    tk.messagebox.showerror("Error", "Please enter a valid number for buffer distance.")
        app = PointRecorder(root)   # Create PointRecorder instance
        root.mainloop()   # Run the Tkinter event loop
    else:
        sect_st = point_start
        sect_end = point_end
        if buffer==None:
            buffer_distance = float(input('Set the buffer distance (m): ')) # define the buffer in meters
        else:
            buffer_distance = buffer

    print('Section Start: ', np.round(sect_st,0))
    print('Section End: ', np.round(sect_end,0))
    print('Buffer Distance (m): ', buffer_distance)
    profile_length = ((sect_end[0]-sect_st[0])**2 + (sect_end[1]-sect_st[1])**2)**0.5
    print('Profile Length (m): ', np.round(profile_length,0))
    # create a dataframe for the section
    database_area['Xper']=0.0
    database_area['Yper']=0.0
    # distance from the line
    database_area ['Buffer1'] = np.abs((sect_end[1] - sect_st[1]) * database_area['X'] - (sect_end[0] - sect_st[0]) * database_area['Y'] + sect_end[0] * sect_st[1] - sect_end[1] * sect_st[0]) / ((sect_end[1] - sect_st[1])**2 + (sect_end[0] - sect_st[0])**2)**0.5
    # distance from the center of the diameter of "profile length  + 2*buffer distance"
    database_area['Buffer2'] = np.abs(((database_area['X'] - ((sect_end[0]+sect_st[0])/2))**2 + (database_area['Y'] - ((sect_end[1]+sect_st[1])/2))**2)**0.5)
    # applying the criteria or the buffer zone
    sect_df = database_area[(database_area.Buffer1 <= buffer_distance) & (database_area.Buffer2 <=((profile_length+buffer_distance)/2))].reset_index(drop=True)
    sect_df = sect_df.copy()
    # Calculating the intersection of the perpendicular line from the point to the section
    m = (sect_end[1] - sect_st[1]) / (sect_end[0] - sect_st[0]) if sect_end[0] - sect_st[0] != 0 else np.inf
    if m == np.inf:
        sect_df.Xper = sect_st[0]
        sect_df.Yper = sect_df.Y
    elif m == 0:
        sect_df.Xper = sect_df.X
        sect_df.Yper = sect_st[1]
    else:
        b = sect_st[1] - m * sect_st[0]
        m_perp = -1 / m
        b_perp = sect_df.Y - m_perp * sect_df.X
        sect_df.Xper = (b_perp - b) / (m - m_perp)
        sect_df.Yper = m * sect_df.Xper + b
    # creating the distance column for plotting
    sect_df['Distance'] = ((sect_df.Xper-sect_st[0])**2 + (sect_df.Yper-sect_st[1])**2)**0.5

    # plotting the boreholes, selected versus not selected
    fig0, ax0 = plt.subplots(figsize=(9,9))
    ax0.scatter(database_area.X,database_area.Y, color='r', s=10)
    ax0.scatter(sect_df['X'],sect_df['Y'], color='b', s=10)
    ax0.ticklabel_format(axis='x', style='plain', useOffset=False, useMathText=True, scilimits=(-3, 8))
    ax0.ticklabel_format(axis='y', style='plain', useOffset=False, useMathText=True, scilimits=(-3, 8))
    ax0.tick_params(axis='y', rotation=0)
    ax0.set_aspect('equal')
    ax0.plot([sect_st[0],sect_end[0]],[sect_st[1],sect_end[1]], color='k')
    ax0.scatter(sect_df['Xper'],sect_df['Yper'], color='g', s=10)
    for bhs in sect_df[names_column].unique():
        ax0.annotate(bhs, (sect_df[sect_df[names_column]==bhs].X.mean(),sect_df[sect_df[names_column]==bhs].Y.mean()), fontsize=7)
    
    # Plotting the section
    fig, ax6 = plt.subplots(figsize=(14,6),)
    # normalization
    if (max_value-min_value)!=0:
        sect_df['Xplot_1'] = sect_df.Distance + (scaleX*profile_length)*(sect_df['clayf']-min_value)/(max_value-min_value)
        sect_df['Xplot_2'] = sect_df.Distance + (scaleX*profile_length)*((sect_df['clayf']+sect_df['capf'])-min_value)/(max_value-min_value)
        sect_df['Xplot_3'] = sect_df.Distance + (scaleX*profile_length)*(sect_df['totalf']-min_value)/(max_value-min_value)
    else:
        sect_df['Xplot_1'] = sect_df.Distance
    sect_df['Yplot'] = -1 * sect_df.depth + sect_df.Z
    sect_df = sect_df.replace([np.inf, -np.inf], np.nan).dropna()

    for bhs in sect_df[names_column].unique():
        ax6.fill_betweenx(sect_df[sect_df[names_column]==bhs].Yplot , x1= sect_df[sect_df[names_column]==bhs].Xplot_2, 
                          x2= sect_df[sect_df[names_column]==bhs].Xplot_3, alpha=1, color='k', facecolor='#0000FF',linewidth=0.3, label='Free')
        ax6.fill_betweenx(sect_df[sect_df[names_column]==bhs].Yplot , x1= sect_df[sect_df[names_column]==bhs].Xplot_1, 
                          x2= sect_df[sect_df[names_column]==bhs].Xplot_2, alpha=1, color='#0A0A0A', facecolor='#00FFFF', linewidth=0.3, label='Capillary-bound')
        ax6.fill_betweenx(sect_df[sect_df[names_column]==bhs].Yplot , x1= sect_df[sect_df[names_column]==bhs].Distance, 
                          x2= sect_df[sect_df[names_column]==bhs].Xplot_1, alpha=1, color='#0A0A0A', facecolor='#DEB887', linewidth=0.3, label='Clay-bound')
        ax6.plot([sect_df[sect_df[names_column]==bhs].Distance.max(),sect_df[sect_df[names_column]==bhs].Distance.max()],
                 [sect_df[sect_df[names_column]==bhs].Z.min(),sect_df[sect_df[names_column]==bhs].Z.min()-sect_df[sect_df[names_column]==bhs].depth.max()],
                 '-k', linewidth=0.5)
        ax6.plot([sect_df[sect_df[names_column]==bhs].Distance.max(),
                sect_df[sect_df[names_column]==bhs].Distance.max()+0.5*(scaleX*profile_length),
                sect_df[sect_df[names_column]==bhs].Distance.max()+1.0*(scaleX*profile_length)],
                 [sect_df[sect_df[names_column]==bhs].Z.min()-sect_df[sect_df[names_column]==bhs].depth.max(),
                  sect_df[sect_df[names_column]==bhs].Z.min()-sect_df[sect_df[names_column]==bhs].depth.max(),
                 sect_df[sect_df[names_column]==bhs].Z.min()-sect_df[sect_df[names_column]==bhs].depth.max()],
                marker='|', markersize=5, linestyle='-', color='k', linewidth=0.5)
        ax6.annotate(np.format_float_positional(min_value, precision=0), xy=(sect_df[sect_df[names_column]==bhs].Distance.max(),
                           sect_df[sect_df[names_column]==bhs].Z.min()-sect_df[sect_df[names_column]==bhs].depth.max()), 
                    xytext=(0,-10), textcoords='offset points', fontsize=6)
        ax6.annotate(np.format_float_positional(max_value, precision=0), xy=(sect_df[sect_df[names_column]==bhs].Distance.max()+1*(scaleX*profile_length),
                           sect_df[sect_df[names_column]==bhs].Z.min()-sect_df[sect_df[names_column]==bhs].depth.max()), 
                    xytext=(0,-10), textcoords='offset points', fontsize=6)

    sect_df.sort_values(by='Distance', ascending=False, kind='mergesort', inplace=True)
    ax6.plot(sect_df.Distance,sect_df.Z,linestyle='-', color='k')
    ax6.plot(sect_df.Distance,sect_df.Z,'ko', markersize=2)
    
    for bhs in sect_df[names_column].unique():
        ax6.annotate(bhs, (sect_df[sect_df[names_column]==bhs].Distance.iloc[0], sect_df[sect_df[names_column]==bhs].Z.iloc[0]+label_offset), fontsize=label_fontsize, rotation=90, horizontalalignment='center', 
                     verticalalignment='bottom')
    
    # Common Lables and Annotations
    ax6.set(xlabel='Distance (m)')
    ax6.set(ylabel='Elevation (m)')
    ax6.set_title(title, weight='bold', fontsize=14, y=1.1)
    if xlim == None:
        ax6.set_xlim(0 , profile_length)
    ax6.set_xlim(xlim)
    ax6.set_ylim(ylim)
        
    # remove duplicate legends
    def legend(ax):
        handles, labels = ax.get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        ax.legend(*zip(*unique), bbox_to_anchor=(1.1, 1), loc="upper right",title="Porosity Class", labelspacing = 1, 
                  fontsize=6, fancybox=True, borderpad=1)
    legend(ax6)
    
    # Labeling the section Orientation
    # Calculating the Azimuth of the section
    azimuth = (90 - math.degrees(math.atan2((sect_end[1] - sect_st[1]), (sect_end[0] - sect_st[0])))) % 360
    az_threshold = 10
    if (azimuth >= 360-az_threshold) | (azimuth <= 0+az_threshold):
        TEXT1, TEXT2 = 'S', 'N'
    elif (azimuth >= 180-az_threshold) & (azimuth <= 180+az_threshold):
        TEXT1, TEXT2 = 'N', 'S'
    elif (azimuth >= 90-az_threshold) & (azimuth <= 90+az_threshold):
        TEXT1, TEXT2 = 'W', 'E'
    elif (azimuth >= 270-az_threshold) & (azimuth <= 270+az_threshold):
        TEXT1, TEXT2 = 'E', 'W'
    elif (azimuth > 0+az_threshold) & (azimuth < 90-az_threshold):
        TEXT1, TEXT2 = 'SW', 'NE'
    elif (azimuth > 90+az_threshold) & (azimuth < 180-az_threshold):
        TEXT1, TEXT2 = 'NW', 'SE'
    elif (azimuth > 270+az_threshold) & (azimuth < 360-az_threshold):
        TEXT1, TEXT2 = 'SE', 'NW'
    elif (azimuth > 180+az_threshold) & (azimuth < 270-az_threshold):
        TEXT1, TEXT2 = 'NE', 'SW'
    ax6.text(x=0.95,y=0.95, s=TEXT2, weight='bold', fontsize=16, transform=fig.transFigure)
    ax6.text(x=0.05,y=0.95, s=TEXT1, weight='bold', fontsize=16, transform=fig.transFigure)
    fig = plt.gcf()
    bdf = buffer_distance
    if output_dirc !='':
            plt.savefig(str(output_dirc + '\\' + output_name+'.png'), bbox_inches='tight', dpi=600)
    return fig0, fig, sect_st, sect_end, bdf

def section_plot_all (dataframe, dataframe_geo, geo_symbology_file, variable, variable_geo='ROCKSYMBOL', names_column='Borehole', point_start=None, point_end=None, 
                      xlim=None, ylim=None, label_offset=0.3, label_fontsize=12, output_dirc='', dict_subplot_geo={'scaleX' : None, 'title':None}, dict_subplot_wp={'min_value': 0, 'max_value': 100, 'scaleX' : 0.03},
                      dict_subplot_data={'plot_type':'curve', 'interpolation':'none', 'min_value':None, 'max_value': None, 'decimal':0, 'color':None,  'log_scale':False, 'scaleX':None, 
                                         'filled':True, 'secondary_axis':False, 'color_scale_label':None}):
        """ A function to plot all types of common sections for BNMR data"""
        # Plotting Porosity Section
        fig0, fig, sect_st, sect_end, bd = section_plot_water_partitioning_data (dataframe=dataframe, names_column=names_column, point_start=point_start, point_end=point_end,
                                            min_value= dict_subplot_wp['min_value'], max_value= dict_subplot_wp['max_value'], title=None, 
                                            scaleX = dict_subplot_wp['scaleX'], xlim=xlim, ylim=ylim, output_dirc=output_dirc, 
                                            output_name='Section_Porosity', label_offset=label_offset, label_fontsize=label_fontsize)
        # Plotting Selected Variable Section
        section_plot (dataframe=dataframe, variable=variable, names_column='Borehole', point_start=sect_st, point_end=sect_end, buffer=bd,
                      xlim=xlim, ylim=ylim, title=None, output_dirc=output_dirc , output_name=str('Section_'+variable), interpolation=dict_subplot_data['interpolation'],
                      max_value= dict_subplot_data['max_value'], min_value= dict_subplot_data['min_value'], plot_type=dict_subplot_data['plot_type'], 
                      color=dict_subplot_data['color'], decimal=dict_subplot_data['decimal'], log_scale=dict_subplot_data['log_scale'], scaleX = dict_subplot_data['scaleX'],  
                        color_scale_label=dict_subplot_data['color_scale_label'], label_offset=label_offset, label_fontsize=label_fontsize)
        # Plotting Geology Section
        section_plot_geo (dataframe=dataframe_geo, geo_symbology_file=geo_symbology_file, names_column = names_column, variable=variable_geo,
                        point_start=sect_st, point_end=sect_end, scaleX = dict_subplot_geo['scaleX'], xlim=xlim, ylim=ylim, buffer=bd, 
                        title=dict_subplot_geo['title'], output_dirc=output_dirc, output_name='Section_Geology', label_offset=label_offset, label_fontsize=label_fontsize)
        print('All sections are created')
        return sect_st, sect_end, bd


# CREATE AVERAGE SHAPE FILES FROM BNMR DATA

def create_avg_BNMR_shape_files (geolabled_database, output_directory,  variable='ROCKSYMBOL', coordinate_system='epsg:32632', apply_filterings = False, multiple_geolables=False, max_noise_threshold = 20, min_depth = 0.11):
    """The input database should include XYZ coordiantes (you should create geolabled database in advance, and add_coordinates to it). This function creates shape file for mean, minimum, and maximum of BNMR parameters for: (I) entire borehole 
    and (II) each geology in the borehole. Note that you may filter the dataset and remove noisy measurements before using the function. Each area will have separate shape files. Don't forget to set the correct UTM zone."""
    # Creating average min max value of BNMR parameters for "entire borehole" at different sites                 
    BNMR_df = geolabled_database.copy()
    # Keeping only one of the duplicated rows / measurements with mixed geologies (because the measurements that are in more than one geology are repeated in multiple rows)
    BNMR_df = BNMR_df.drop_duplicates(subset=['depth', 'Borehole','Study_Area'], keep='first')
    if apply_filterings == True:
        BNMR_df = BNMR_df[BNMR_df.depth > min_depth]  # Removing the data that is too close to the surface (i.e., part of the sensitive zone above the surface)
        BNMR_df = BNMR_df[BNMR_df.noise < max_noise_threshold]  #removing highly noisy data

    if ('X' not in BNMR_df.columns) & ('Y' not in BNMR_df.columns) :
        print('Data does not have coordinates. You may use add_coordiante function to add XYZ to the current dataframe')    
    else:
        for std in BNMR_df.Study_Area.unique():
            i = -1
            avg_df = pd.DataFrame(columns=['Borehole','X','Y','Z', 'depth_max', 'depth_min'])
            for bh in BNMR_df[(BNMR_df.Study_Area==std)].Borehole.unique():
                i += 1
                avg_df.loc[i,['Borehole','X','Y','Z', 'depth_max', 'depth_min', 'clayf_min','clayf_max','clayf_avg',
                              'capf_min','capf_max','capf_avg', 'freef_min','freef_max','freef_avg',
                              'imob_min','imob_max','imob_avg', 'totf_min','totf_max','totf_avg',
                              'mlT2_min','mlT2_max','mlT2_avg',]] = [bh, 
                                BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)].X.iloc[0], 
                                BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)].Y.iloc[0],
                                BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)].Z.iloc[0],
                                BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)].depth.max(), 
                                BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)].depth.min(),
                                BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)].clayf.min(), 
                                BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)].clayf.max(),
                                BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)].clayf.mean(),
                                BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)].capf.min(), 
                                BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)].capf.max(),
                                BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)].capf.mean(), 
                                BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)].freef.min(), 
                                BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)].freef.max(),
                                BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)].freef.mean(),
                                BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)].immobile.min(), 
                                BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)].immobile.max(),
                                BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)].immobile.mean(),
                                BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)].totalf.min(), 
                                BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)].totalf.max(),
                                BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)].totalf.mean(),
                                BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)].mlT2.min(), 
                                BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)].mlT2.max(),
                                BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)].mlT2.mean(),]
            avg_df = avg_df.fillna(0)
            # Convert the dataframe to a GeoDataFrame & export to a shapefile
            gdf = gpd.GeoDataFrame(avg_df, crs=pyproj.CRS(coordinate_system), geometry=gpd.points_from_xy(avg_df.X, avg_df.Y)) 
            gdf.to_file(output_directory+'\\'+std+'_Avg_BNMR_Borehole.shp')

            # Creating average min max value of BNMR parameters for each geology at different boreholes
            BNMR_df = geolabled_database.copy()
            if apply_filterings == True:
                if multiple_geolables==False:
                    BNMR_df = BNMR_df.drop_duplicates(subset=['depth', 'Borehole','Study_Area'], keep=False)
                BNMR_df = BNMR_df[BNMR_df.depth > min_depth]  # Removing the data that is too close to the surface (i.e., part of the sensitive zone above the surface)
                BNMR_df = BNMR_df[BNMR_df.noise < max_noise_threshold]  #removing highly noisy data
            
            for std in BNMR_df.Study_Area.unique():
                i = -1
                for rck in BNMR_df[(BNMR_df.Study_Area==std)][variable].unique():
                    avg_df = pd.DataFrame(columns=['Borehole','X','Y','Z'])
                    for bh in BNMR_df[(BNMR_df[variable]==rck)&(BNMR_df.Study_Area==std)].Borehole.unique():
                        i += 1
                        avg_df.loc[i,['Borehole','X','Y','Z', 'clayf_min','clayf_max','clayf_avg',
                                      'capf_min','capf_max','capf_avg', 'freef_min','freef_max','freef_avg',
                                      'imob_min','imob_max','imob_avg', 'totf_min','totf_max','totf_avg',
                                      'mlT2_min','mlT2_max','mlT2_avg',]] = [bh, 
                                        BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)&(BNMR_df[variable]==rck)].X.iloc[0], 
                                        BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)&(BNMR_df[variable]==rck)].Y.iloc[0],
                                        BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)&(BNMR_df[variable]==rck)].Z.iloc[0],
                                        BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)&(BNMR_df[variable]==rck)].clayf.min(), 
                                        BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)&(BNMR_df[variable]==rck)].clayf.max(),
                                        BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)&(BNMR_df[variable]==rck)].clayf.mean(),
                                        BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)&(BNMR_df[variable]==rck)].capf.min(), 
                                        BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)&(BNMR_df[variable]==rck)].capf.max(),
                                        BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)&(BNMR_df[variable]==rck)].capf.mean(), 
                                        BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)&(BNMR_df[variable]==rck)].freef.min(), 
                                        BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)&(BNMR_df[variable]==rck)].freef.max(),
                                        BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)&(BNMR_df[variable]==rck)].freef.mean(),
                                        BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)&(BNMR_df[variable]==rck)].immobile.min(), 
                                        BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)&(BNMR_df[variable]==rck)].immobile.max(),
                                        BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)&(BNMR_df[variable]==rck)].immobile.mean(),
                                        BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)&(BNMR_df[variable]==rck)].totalf.min(), 
                                        BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)&(BNMR_df[variable]==rck)].totalf.max(),
                                        BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)&(BNMR_df[variable]==rck)].totalf.mean(),
                                        BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)&(BNMR_df[variable]==rck)].mlT2.min(), 
                                        BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)&(BNMR_df[variable]==rck)].mlT2.max(),
                                        BNMR_df[(BNMR_df.Borehole==bh)&(BNMR_df.Study_Area==std)&(BNMR_df[variable]==rck)].mlT2.mean(),]
                    avg_df = avg_df.fillna(0)
                    # Convert the dataframe to a GeoDataFrame & export to a shapefile
                    gdf = gpd.GeoDataFrame(avg_df, crs=pyproj.CRS(coordinate_system), geometry=gpd.points_from_xy(avg_df.X, avg_df.Y))
                    gdf.to_file(output_directory+'\\'+std+'_Avg_BNMR_'+rck+'.shp')
        print ('All shape files are created.')


def add_new_lable (BNMR_dataframe, dataframe_new , parameters, output_dirc=''):
    """This function is for adding a new data or column (string or numerical variable) to the BNMR or geo_labeled BNMR dataframe.
    The parameters should be a list including the name of new columns (format: list). If there is no TOP and BOTTOM, the columns will be created. Moreover, the function highlights the problems in TOP, BOTTOM column of the new dataframe to help checking the input file and 
    correct the mistakes if necessary. Note that the columns with NAN, etc. values are not removed from the final output as it is problematic if we have multiple 
    parameters where we have value for one and not for the others. """
    final_dataframe = pd.DataFrame()
    No_data = []
    # dropping problematic rows (non numeric values)
    if (len(dataframe_new[pd.to_numeric(dataframe_new['TOP'], errors='coerce').isna()])!=0) | (len(dataframe_new[pd.to_numeric(dataframe_new['BOTTOM'], errors='coerce').isna()])!=0):
        print( 'Warning! The top and bottom of new dataframe has non numeric values. It is dropped from the dataframe to run this function.')
        print( 'Check the new dataframe for possible errors.')
        print('\nTOP column:\n' , dataframe_new[pd.to_numeric(dataframe_new['TOP'], errors='coerce').isna()])
        print('\n\nBOTTOM column:\n' ,dataframe_new[pd.to_numeric(dataframe_new['BOTTOM'], errors='coerce').isna()])
        dataframe_new.drop(dataframe_new[pd.to_numeric(dataframe_new['TOP'], errors='coerce').isna()].index, inplace=True)
        dataframe_new.drop(dataframe_new[pd.to_numeric(dataframe_new['BOTTOM'], errors='coerce').isna()].index, inplace=True)
    # checking if BNMR data has TOP and BOTTOM columns
    if ('TOP' not in BNMR_dataframe.columns) | ('BOTTOM' not in BNMR_dataframe.columns):
        step = 0.22  # height of the sensitive zone or vertical resolution
        BNMR_dataframe['TOP'] = BNMR_dataframe.depth - step/2
        BNMR_dataframe.TOP = BNMR_dataframe.TOP.round(3)
        BNMR_dataframe['BOTTOM'] = BNMR_dataframe.depth + step/2
        BNMR_dataframe.BOTTOM = BNMR_dataframe.BOTTOM.round(3)

    for Borehole_name in BNMR_dataframe.Borehole.unique():
        if dataframe_new[dataframe_new['Borehole']==Borehole_name].empty == True:
            No_data.append(Borehole_name)
            continue                
        d9 = BNMR_dataframe[BNMR_dataframe.Borehole==Borehole_name].copy()
        # Labling the  data
        d9_new = pd.DataFrame()
        for par in parameters:
            d9[par]=None
        counter = -1
        for i in range(len(d9)):
            for j in range(len(dataframe_new[dataframe_new['Borehole']==Borehole_name])):
                if (pd.Interval(d9.TOP.iloc[i], d9.BOTTOM.iloc[i])).overlaps(pd.Interval(dataframe_new[dataframe_new['Borehole']==Borehole_name].TOP.iloc[j],dataframe_new[dataframe_new['Borehole']==Borehole_name].BOTTOM.iloc[j])):
                    counter+=1
                    d9_new = pd.concat([d9_new[:],d9.iloc[i]], axis=1, join='outer', ignore_index=True)
                    for par in parameters:
                        d9_new.transpose().loc[counter, par]= dataframe_new[dataframe_new['Borehole']==Borehole_name][par].iloc[j]
        # Removing the duplicatd rows (those that all columns are the same due to any problem in borehole geology data)
        d9_new = d9_new.transpose()
        d9_new.drop_duplicates(keep='first',inplace=True)
        d9_new.reset_index(drop=True,inplace=True)
        # Appending the new labled boreholes to previous ones
        final_dataframe = pd.concat([d9_new,final_dataframe], axis=0)
        final_dataframe.reset_index(drop=True,inplace=True)
    print('\nTask completed.')
    if No_data !=[]:
        print('The list of Boreholes that exist in BNMR dataframe but are not present in the new dataframe:\n', No_data)        
    if output_dirc!='':
        print('Enter file name: ')
        file_name=str(input())
        clear_output()
        final_dataframe.to_csv(output_dirc+'\\'+file_name+'.txt', sep='\t', index=False)
        print('The data file is exported.')
    return final_dataframe



##########################################################################################################
# PYTHON CLASSES 

class Data:
    def __init__(self, boreholes_list):       
        self.boreholes_list = boreholes_list
        # for converting the input variable from dataframe to list (code will work for both input types)
        if 'DataFrame' in str(type(self.boreholes_list)):
            self.boreholes_list = np.squeeze(self.boreholes_list.values.tolist())
    def import_coordinates(self, boreholes_coordinates_file):
        """The input file should be a tab delimited text with columns named as Borehole, X, Y, and Z."""
        self.XYZ = pd.DataFrame(pd.read_csv(boreholes_coordinates_file, sep='\t',encoding="ISO-8859-1"))
        return self.XYZ        

    def import_data(self, Borehole_name, input_dirc, Geology=True, Hydrology=False, Groundwater=False):
        """ It is not required to call this function separately prior to any operation. It is used in all other commonly used functions such as plotting etc. when needed
        All files must be in the VC export directory and named as: Borehole_Geology.txt, Geological_Symbols.txt, Hydrology_data.txt, Groundwater_depth.txt"""
        self.input_dirc = input_dirc # directory with Vista Clara processed files (exports)
        self.input_dirc.replace('\\\\','\\')
        self.input_dirc = input_dirc             
        """ VISTA CLARA PROCESSED DATA"""
        # SE_decay file (Spin Echo Decay measurements)
        self.d1 = pd.DataFrame(pd.read_csv(self.input_dirc +'\\'+ Borehole_name +'\\'+ Borehole_name +'_SE_decay.txt',sep='\t', header=None))
        # SE_decay_uniform
        self.d2 = pd.DataFrame(pd.read_csv(self.input_dirc +'\\'+ Borehole_name +'\\'+ Borehole_name +'_SE_decay_time.txt',sep='\t', header=None))
        # SE_decay_time (Time of SE measurements)
        self.d3 = pd.DataFrame(pd.read_csv(self.input_dirc +'\\'+ Borehole_name +'\\'+ Borehole_name +'_SE_decay_uniform.txt',sep='\t', header=None))
        # 1Dvectors (different parameters extracted by BNMR data including WC, mlT2, K,T, etc.)
        self.d4 = pd.DataFrame(pd.read_csv(self.input_dirc +'\\'+ Borehole_name +'\\'+ Borehole_name +'_1Dvectors.txt',sep='\t', skipinitialspace=True))
        self.d4[['clayf','capf','freef','totalf']] = self.d4[['clayf','capf','freef','totalf']].mul(100)
        # 1Dvectors_uniform (same as d4!)
        self.d5 = pd.DataFrame(pd.read_csv(self.input_dirc +'\\'+ Borehole_name +'\\'+ Borehole_name +'_1Dvectors_uniform.txt',sep='\t',skipinitialspace=True))
        self.d5[['clayf','capf','freef','totalf']] = self.d5[['clayf','capf','freef','totalf']].mul(100)
        # T2_bins_log10s (T2 distribution bins)
        self.d6 = pd.DataFrame(pd.read_csv(self.input_dirc +'\\'+ Borehole_name +'\\'+ Borehole_name +'_T2_bins_log10s.txt',sep='\t', header=None))
        # T2_dist (T2 distribution values of the histogram)
        self.d7 = pd.DataFrame(pd.read_csv(self.input_dirc +'\\'+ Borehole_name +'\\'+ Borehole_name +'_T2_dist.txt',sep='\t', header=None))
        # T2_dist_uniform (same as d7!)
        self.d8 = pd.DataFrame(pd.read_csv(self.input_dirc +'\\'+ Borehole_name +'\\'+ Borehole_name +'_T2_dist_uniform.txt',sep='\t', header=None))
        print('VC data files are imported.')
        
        # GEOLOGY DATA + GEOLOGY SYMBOLOGY 
        if (Geology==True):
            self.BH_Geo = pd.DataFrame(pd.read_csv(input_dirc + "\\" + "Borehole_Geology.txt", sep='\t',encoding= 'unicode_escape'))
            self.BH_Geo.Borehole = self.BH_Geo.Borehole.astype('string')
            self.Geo_Symbology = pd.DataFrame(pd.read_csv(input_dirc + "\\" + "Geological_Symbols.txt", sep='\t',encoding= 'unicode_escape'))
            print('Boreholes Geology data is imported.')

        # HYDROLOGY DATA 
        if (Hydrology==True):
            self.Hydro_data = pd.DataFrame(pd.read_csv(input_dirc + "\\" + "Hydrology.txt", sep='\t',encoding="ISO-8859-1"))
            self.Hydro_data.Borehole = self.Hydro_data.Borehole.astype('string')
            print('Hydrological data is imported.')   

        # GROUNDWATER-LEVEL DATA
        if (Groundwater==True):
            self.water_depth = pd.DataFrame(pd.read_csv(input_dirc + "\\" + "Groundwater.txt", sep='\t',encoding="ISO-8859-1"))
            self.water_depth.Borehole = self.water_depth.Borehole.astype('string')
            print('Groundwater level data is imported.')
        return self
    
    def logplot (self, input_dirc, output_dirc='', Geology=True, Hydrology=False, Groundwater=False, wc_range=(0,100), T2D_plot_type = 0, Geo_variable='ROCKSYMBOL', Geology_Description='DESCRIPTION', 
                 K_plot_params = {'plotting':'None', 'k_field_method':'Slug test', 'max_noise_BNMR': 20, 'k_limit':{'kmin':None, 'kmax':None}, 'SDR_dict': {'show':False,'b':0,'m':0,'n':0}, 'TC_dict': {'show':False, 'c':0}, 'SOE_dict' : {'show':False, 'c':0,'d':0}}): 
        """ Remember to specify the output directory in case the list includes more than one borehole. Hydrology or K data should be in m/s. 
         The plotting in K_plot_params can be "auto" for Vista clara values, "None" to not show K data, and "calibrated" in case the calibration results were good"""
        if isinstance(self.boreholes_list, str): # converts string to array if the input is a string (this is just for a simpler use of logplot function)
            self.boreholes_list = np.array(self.boreholes_list.split())
        default_K_plot_params = {'plotting':'None', 'k_field_method':'Slug test', 'max_noise_BNMR': 20, 'k_limit':{'kmin':None, 'kmax':None}, 'SDR_dict': {'show':False,'b':0,'m':0,'n':0}, 
                     'TC_dict': {'show':False, 'c':0}, 'SOE_dict' : {'show':False, 'c':0,'d':0}}
        if K_plot_params is None: # that is for enabling the function call to run without adding all details of K_plot_params 
            K_plot_params = default_K_plot_params
        else:
            for key, value in default_K_plot_params.items():
                if key not in K_plot_params:
                    K_plot_params[key] = value
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if sub_key not in K_plot_params[key]:
                            K_plot_params[key][sub_key] = sub_value
        bh_with_no_geo =[]
        for Borehole_name in self.boreholes_list:
            self.import_data(Borehole_name,input_dirc, Geology=Geology, Hydrology=Hydrology, Groundwater=Groundwater) # importing data
            if Geology==True: # checking the boreholes if all have geo data
                if self.BH_Geo[self.BH_Geo['Borehole']==Borehole_name].empty:
                    bh_with_no_geo.append(Borehole_name)

            # PLOTTING DETAILS
            if (K_plot_params['plotting']=='None') & (Hydrology==False):
                number_of_plots = 5
            else:
                number_of_plots = 6
            marker_size = 15
            legend_font= 9.5
            wratio = np.ones(number_of_plots)*3
            wratio [0] = 1
            fig, axes = plt.subplots(1,number_of_plots, sharey=True, figsize=(14,6), gridspec_kw={'width_ratios': wratio})
            fig.suptitle(str(Borehole_name), fontsize=16)
            
            # Geology / Lithology Plot of the borehole
            ax= axes[0]
            if Geology==True: 
                for i in range(len(self.BH_Geo[self.BH_Geo['Borehole']==Borehole_name])):
                    xx=[0,10,10,0]
                    yy=[self.BH_Geo[self.BH_Geo['Borehole']==Borehole_name].TOP.iloc[i],
                        self.BH_Geo[self.BH_Geo['Borehole']==Borehole_name].TOP.iloc[i],
                        self.BH_Geo[self.BH_Geo['Borehole']==Borehole_name].BOTTOM.iloc[i],
                        self.BH_Geo[self.BH_Geo['Borehole']==Borehole_name].BOTTOM.iloc[i]]
                    symbol_color=self.Geo_Symbology[self.Geo_Symbology['ROCKSYMBOL']==self.BH_Geo[self.BH_Geo['Borehole']==Borehole_name][Geo_variable].iloc[i]].COLORCODE.to_string(index=False)
                    ax.fill(xx,yy, color=symbol_color,linewidth=0, label=self.Geo_Symbology[Geology_Description][self.Geo_Symbology.COLORCODE==symbol_color].to_string(index=False))
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(),bbox_to_anchor=(-1, 1), loc="upper right",title="LEGEND", labelspacing = 1, fontsize=8, fancybox=True, borderpad=1)
            ax.set_ylabel('Depth (m)')
            ax.set_ylim(self.d5.depth.min(), self.d5.depth.max())
            ax.set_xlim(0,10)
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_xlabel('Borehole \nGeology')

            # Groundwater_depth data
            if (Groundwater==True):
                ax.plot((0,10), (self.water_depth[self.water_depth['Borehole']==Borehole_name].water_depth, self.water_depth[self.water_depth['Borehole']==Borehole_name].water_depth),
                color='k', linewidth=2, linestyle='--', label='Groundwater')
            
            #################################################
            # Creating a dataframe for spin echo decay
            depth = [] 
            time = []
            SE_decay = []
            SE_decay_uniform = []
            for i in range(len(self.d1)):
                for j in range(len(self.d2)):
                    depth.append(self.d1[0][i])
                    time.append(self.d2[0][j])
                    SE_decay.append(self.d1[j+1][i])
                    SE_decay_uniform.append(self.d1[j+1][i])
            self.df_signal = pd.DataFrame(data=[depth, time, SE_decay, SE_decay_uniform])
            self.df_signal = self.df_signal.transpose()
            self.df_signal.rename(columns={0:'depth', 1:'time', 2:'SE_decay', 3:'SE_decay_uniform'}, inplace=True)
            self.df_signal = self.df_signal.drop(self.df_signal[self.df_signal['depth']<0].index)  # dropping all data above topography
           # dropping all data above topography from the main data file
            self.d4 = self.d4.drop(self.d4[self.d4['depth']<0].index)  
            self.d5 = self.d5.drop(self.d5[self.d5['depth']<0].index) 
            # Creating  a dataframe for the T2-distribution plotting (since T2bins and values are in separate text files) 
            depth2 = []
            time2 = []
            T2 = []
            for i in range(len(self.d7)):
                for j in range(len(self.d6.transpose())):
                    depth2.append(self.d7[0][i])
                    time2.append(self.d6[j][0])
                    T2.append(self.d7[j+1][i])
            self.T2_dist = pd.DataFrame(data=[depth2, time2, T2])
            self.T2_dist = self.T2_dist.transpose()
            self.T2_dist.rename(columns={0:'depth', 1:'time', 2:'T2'}, inplace=True)
            self.T2_dist.time = 10**(self.T2_dist.time)
            self.T2_dist = self.T2_dist.drop(self.T2_dist[self.T2_dist['depth']<0].index) # dropping all data above topography
            #################################################
            # PLOTTING THE LOGS #
            #################################################
            # Spin Echo train data - NMR Signal Decay
            ax=axes[1]

            depth = np.unique(np.array(self.df_signal.depth))
            time  = np.unique(np.array(self.df_signal.time*1000))
            SE_decay = np.array(self.df_signal.SE_decay).reshape(depth.shape+time.shape)
            pcm = ax.pcolor(time,depth, SE_decay,  shading='nearest', vmin=0, vmax=50)
            ax.set_xlabel('t (ms)') 
            ax.set_xlim(0,max(time))
            #ax.set_title('Spin Echo Decay', fontsize=legend_font)
            # plt.colorbar(pcm, ax=axes[1], orientation='horizontal')
            #ax.set_ylabel('Depth (m)')
            #ax.clim(0,50) 
            #ax.colorbar(shrink=0.5)

            #################################################
            # Water Content Plot
            ax = axes[2]

            ax.fill_betweenx(self.d5.depth , x1= 0, x2= self.d5.freef, alpha=1, color='#0A0A0A', facecolor='#0000FF',linewidth=0.3, label='Free')
            ax.fill_betweenx(self.d5.depth , x1= self.d5.freef, x2= (self.d5.freef + self.d5.capf), alpha=1, color='#0A0A0A', facecolor='#00FFFF', linewidth=0.3, label='Capillary-bound')
            ax.fill_betweenx(self.d5.depth , x1= (self.d5.freef + self.d5.capf), x2= (self.d5.freef + self.d5.capf + self.d5.clayf), alpha=1, color='#0A0A0A', facecolor='#DEB887', linewidth=0.3, label='Clay-bound')
            ax.plot(self.d5.totalf ,self.d5.depth, color='#050505',linewidth=1.5, label='Total')
            ax.legend(loc ="upper right", fontsize=legend_font-2.5)
            #plt.grid(axis='x', color = '#666666', linestyle = '--', linewidth = 0.3)
            ax.set_xlim(wc_range)
            ax.set_ylim(self.d5.depth.min(), self.d5.depth.max())
            ax.invert_yaxis()
            ax.set_xlabel('Water Content (%)')

            #################################################
            # T2 distribution  
            ax = axes[3]
            if T2D_plot_type == 0:  # (PLOTTING OPTION: curve)
                overlap = 0.0
                T2_bins = np.array(self.d6)[0]
                T2_dist_raw = np.array(self.d7)
                depth = T2_dist_raw[:, 0]
                T2_dist = T2_dist_raw[:, 1:]
                T2_mean = np.sum(T2_bins*T2_dist, axis=1)/np.sum(T2_dist, axis=1)
                int_f = interp1d(depth, T2_mean, kind='nearest')
                depth_int = np.linspace(depth.min(), depth.max(), 1001)
                T2mean_int= int_f(depth_int)
                dz = (depth.max()-depth.min()) / len(depth)
                dz*=(1+overlap)
                T2_dist/=T2_dist.max()
                #for i_z, c_depth in zip(range(len(depth)-1, -1, -1), depth[::-1]):
                for i_z, c_depth in enumerate(depth):    
                    cT2_mean = T2_dist[i_z]
                    #cT2_mean/=
                    cT2_mean*=dz
                    #ax.plot(T2_bins, c_depth-cT2_mean, 'k')
                    ax.fill_between(T2_bins, c_depth, c_depth-cT2_mean, facecolor='g', edgecolor='g')
                ax.invert_yaxis()
                xticks = ax.get_xticks()
                x_labels = [10**x for x in xticks]
                ax.set_xticks(xticks)
                ax.set_xticklabels(x_labels)
                ax.set_xlabel('$T_2$ (s)')
                #ax.plot(T2mean_int, depth_int, 'k-')
                ax.plot(np.log10(self.d5.mlT2), self.d5.depth, 'ok', markersize=4, linestyle='-', label='$T_{2ML}$')
                ax.legend(loc ="upper right", fontsize=legend_font-1)
            else: # (PLOTTING OPTION 1: Filled image + step-like T2ML)          
                T2_bins = np.array(self.d6)[0]
                T2_dist_raw = np.array(self.d7)
                depth = T2_dist_raw[:, 0]
                T2_dist = T2_dist_raw[:, 1:]
                T2_mean = np.sum(T2_bins*T2_dist, axis=1)/np.sum(T2_dist, axis=1)
                int_f = interp1d(depth, T2_mean, kind='nearest')
                depth_int = np.linspace(depth.min(), depth.max(), 1001)
                T2mean_int= int_f(depth_int)
                ax.pcolor(T2_bins, depth, T2_dist, shading='nearest')
                ax.plot(T2mean_int, depth_int, 'k-',label='$T_{2ML}$')    
                xticks = ax.get_xticks()
                x_labels = [10**x for x in xticks]
                ax.set_xticklabels(x_labels)
                ax.set_xlabel('$T_2$ (ms)')
                ax.legend(loc ="upper right", fontsize=legend_font-1)

            #################################################
            # Hydraulic Conductivity            
            if K_plot_params['plotting']=='Auto':
                ax = axes[4]
                ax.plot(self.d5.Ksdr, self.d5.depth , color='#EE1289',linewidth=2, label='$K_{SDR}$')
                ax.scatter(self.d5.Ksdr, self.d5.depth, s=marker_size, color='#EE1289')
                ax.plot(self.d5.Ktc, self.d5.depth , color='#7FFF00',linewidth=2, label='$K_{TC}$')
                ax.scatter(self.d5.Ktc, self.d5.depth, s=marker_size, color='#7FFF00')
                ax.plot(self.d5.Ksoe, self.d5.depth , color='#FFD700',linewidth=2, label='$K_{SOE}$')
                ax.scatter(self.d5.Ksoe, self.d5.depth, s=marker_size, color='#FFD700')
            elif K_plot_params['plotting']=='Calibrated':
                ax = axes[4]
                if (Groundwater==True): # to make sure we only show K values for the saturated zone
                    water_depth = self.water_depth.loc[self.water_depth.Borehole==Borehole_name , 'water_depth'].values[0] 
                else:
                    water_depth = 0
                if K_plot_params['SDR_dict']['show']==True: # SDR plot
                    dictionary = K_plot_params['SDR_dict']
                    self.d5.loc[:,'Ksdr'] = dictionary['b'] * ((self.d5.loc[:,'totalf']/100)**dictionary['m']) * (self.d5.loc[:,'mlT2']**dictionary['n'])
                    self.d5.loc[self.d5.noise > K_plot_params['max_noise_BNMR'],'Ksdr'] = np.nan
                    ax.plot(self.d5.Ksdr[self.d5.depth>=water_depth], self.d5.depth[self.d5.depth>=water_depth] , color='#EE1289',linewidth=2, label='$K_{SDR}$')
                    ax.scatter(self.d5.Ksdr[self.d5.depth>=water_depth], self.d5.depth[self.d5.depth>=water_depth], s=marker_size, color='#EE1289')
                if K_plot_params['TC_dict']['show'] == True: # TC plot
                    dictionary = K_plot_params['TC_dict']
                    self.d5.loc[:,'Ktc'] =  ((self.d5.loc[:,'freef']/(self.d5.loc[:,'capf']+self.d5.loc[:,'clayf']))*(((self.d5.loc[:,'totalf']/100)/dictionary['c'])**2))**2
                    self.d5.loc[self.d5.noise > K_plot_params['max_noise_BNMR'],'Ktc'] = np.nan
                    ax.plot(self.d5.Ktc[self.d5.depth>=water_depth], self.d5.depth[self.d5.depth>=water_depth] , color='#7FFF00',linewidth=2, label='$K_{TC}$')
                    ax.scatter(self.d5.Ktc[self.d5.depth>=water_depth], self.d5.depth[self.d5.depth>=water_depth], s=marker_size, color='#7FFF00')
                if K_plot_params['SOE_dict']['show'] == True: # SOE plot
                    dictionary = K_plot_params['SOE_dict']
                    self.d5.loc[:,'Ksoe'] = dictionary['c'] * (self.d5.loc[:,'soe']**dictionary['d'])
                    self.d5.loc[self.d5.noise > K_plot_params['max_noise_BNMR'],'Ksoe'] = np.nan
                    ax.plot(self.d5.Ksoe[self.d5.depth>=water_depth], self.d5.depth[self.d5.depth>=water_depth] , color='#FFD700',linewidth=2, label='$K_{SOE}$')
                    ax.scatter(self.d5.Ksoe[self.d5.depth>=water_depth], self.d5.depth[self.d5.depth>=water_depth], s=marker_size, color='#FFD700')

            if (Hydrology==True): # plotting field test hydraulic conductivity if available 
                ax = axes[4]
                K_plot_params['plotting']=='Auto'
                Hydro_d = self.Hydro_data
                Hydro_d = Hydro_d[(Hydro_d.Borehole==Borehole_name) & (Hydro_d.K>0)]
                k_label = "$K_{"+ K_plot_params['k_field_method']+"}$"
                ax.scatter(Hydro_d.K,(Hydro_d.TOP+Hydro_d.BOTTOM)/2, color='k', marker='o', s=marker_size, label=k_label)
                for i in range(len(Hydro_d)):
                    ax.plot([Hydro_d.K,Hydro_d.K],[Hydro_d.TOP,Hydro_d.BOTTOM], linewidth=2, color='black')
                    i+=1

            if number_of_plots != 5:
                ax.legend(loc ="upper right", fontsize=legend_font)
                ax.grid(axis='x', color = 'black', linestyle = '-', linewidth = 0.4)
                ax.grid(axis='x', which='minor', color = 'grey', linestyle = '--', linewidth = 0.4)
                ax.set_ylim(self.d5.depth.min(), self.d5.depth.max())
                ax.set_xscale('log')
                ax.invert_yaxis()
                ax.set_xlabel('K (m/s)')
                ax.set_xlim(K_plot_params['k_limit'].get('kmin'),K_plot_params['k_limit'].get('kmax'))

            # Noise plot
            if number_of_plots == 5:
                ax=axes[4] 
            else:
                ax=axes[5]

            ax.plot(self.d5.noise, self.d5.depth, color='#FF6103')
            ax.scatter(self.d5.noise, self.d5.depth, s=marker_size, color='#FF6103',label='Noise')
            ax.plot(self.d5.noise, self.d5.depth , color='#FF6103',linewidth=2)
            #ax.legend(loc ="upper right", fontsize=10)
            ax.grid(axis='x', which='major', color = '#171717', linestyle = '--', linewidth = 0.4)
            ax.set_ylim(self.d5.depth.min(), self.d5.depth.max())
            #plt.set_xlim(self.d5.noise.min()/1.2, self.d5.noise.max()*1.2)
            ax.set_xscale('linear')
            ax.invert_yaxis()
            ax.set_xlabel('Noise (%)')
            
            #ax.set_ylabel('Depth (m)')
            #ax.set_title('Noise', fontsize=12)

            #################################################
            # Transmissivity 
            # ax=axes[5]

            # ax.plot(self.d5.Tsdr, self.d5.depth , color='#EE1289',linewidth=2, label='$T_{SDR}$')
            # ax.scatter(self.d5.Tsdr, self.d5.depth, s=marker_size, color='#EE1289')
            # ax.plot(self.d5.Ttc, self.d5.depth , color='#7FFF00',linewidth=2, label='$T_{TC}$')
            # ax.scatter(self.d5.Ttc, self.d5.depth, s=marker_size, color='#7FFF00')
            # ax.plot(self.d5.Tsoe, self.d5.depth , color='#FFD700',linewidth=2, label='$T_{SOE}$')
            # ax.scatter(self.d5.Tsoe, self.d5.depth, s=marker_size, color='#FFD700')
            # ax.legend(loc ="upper right", fontsize=legend_font)
            # ax.grid(axis='x', color = 'black', linestyle = '-', linewidth = 0.4)
            # ax.grid(axis='x', which='minor', color = 'grey', linestyle = '--', linewidth = 0.4)
            # ax.set_ylim(self.d5.depth.min(), self.d5.depth.max())
            # ax.set_xlim(0, )
            # ax.set_xscale('linear')
            # ax.invert_yaxis()
            # ax.set_xlabel('T ($m^{2}$/day)')

            #################################################
            # SOE (Some Of Echoes) plot
            # ax=axes[6]

            # ax.plot(self.d5.soe, self.d5.depth, markersize=marker_size, color='#00EE00')
            # #ax.plot(np.sum(T2_dist, axis=1), depth)
            # ax.scatter(self.d5.soe, self.d5.depth, s=marker_size, color='#00EE00',label='SOE')
            # ax.plot(self.d5.soe, self.d5.depth , color='#00EE00',linewidth=2)
            # #ax.legend(loc ="upper right", fontsize=10)
            # ax.grid(axis='x', which='major', color = '#171717', linestyle = '--', linewidth = 0.4)
            # ax.grid(axis='x', which='minor', color = '#171717', linestyle = '--', linewidth = 0.2)
            # ax.set_ylim(self.d5.depth.min(), self.d5.depth.max())
            # ax.set_xscale('log')
            # ax.invert_yaxis()
            # ax.set_xlabel('SOE')

            ################################################
            #Saving the figure in a file
            fig.set_facecolor("white")
            if output_dirc!='':
                plt.savefig(output_dirc+'\\'+Borehole_name+'.png')
                clear_output()
        if output_dirc!='':
            print('Creating .png exports for all boreholes is completed.')
        if bh_with_no_geo: # Check if there is any borehole with no geology data
            print("Boreholes without geology data:\n", bh_with_no_geo)
        return
       
    def export_las (self, input_dirc, boreholes_coordinates_file, output_dirc):        
            """ Both input and output directories must be specified.
            Note that the .las template which is a text file should be in the same directory as vista clara export files"""
            # Remember that there are some reserved letters in .las format and more importantly, there should not be any extra "enter or \n" in the txt or las file
            # after creating the text file, we should change the txt to las and that's it (included in the code)
            
            self.import_coordinates(boreholes_coordinates_file)
            for Borehole_name in self.boreholes_list:
                self.import_data(Borehole_name,input_dirc, Geology=False, Hydrology=False, Groundwater=False)
                # The informatation about borehole that should be added to the las file
                BOREHOLENAME = Borehole_name  # this is defined in the second cell 
                XCORDUTM = float(self.XYZ.X[self.XYZ['Borehole']==Borehole_name])
                YCORDUTM = float(self.XYZ.Y[self.XYZ['Borehole']==Borehole_name]) 
                MINIMUMDEPTH = self.d5.depth.min()
                MAXIMUMDEPTH = self.d5.depth.max()
                MEASUREMENTSTEP = 0.22
                #Using the blank template & replaciung the values in it (for each borehole, a new file is created in the specified directory)
                #copying the blank template 
                Blank_Template = self.input_dirc + "\\" + "Template.txt"
                NewFile = output_dirc + "\\" + Borehole_name + ".txt"
                shutil.copyfile(Blank_Template, NewFile)
                #Inserting values in the file
                for line in fileinput.input([NewFile], inplace=True):
                    line = line.replace('MINIMUMDEPTH',str(MINIMUMDEPTH)).replace('MAXIMUMDEPTH',str(MAXIMUMDEPTH)).replace('MEASUREMENTSTEP',
                            str(MEASUREMENTSTEP)).replace('BOREHOLENAME',str(BOREHOLENAME)).replace('XCORDUTM',str(XCORDUTM)).replace('YCORDUTM',str(YCORDUTM))
                    # sys.stdout is redirected to the file
                    sys.stdout.write(line)   
                # Merging the data into the txt file
                self.d5 = self.d5.drop(columns={'unix_time','board_temp','magnet_temp'})
                self.d5 = self.d5.drop(self.d5[self.d5['depth']<0].index) 
                tfile = open(NewFile, 'a')
                tfile.write(self.d5.to_string(index=False))
                tfile.close()
                NewFile2 = output_dirc + "\\" + Borehole_name +".las"
                os.replace(NewFile, NewFile2)
                clear_output()
            print('The .las file export completed.')      

    def create_BNMR_dataframe (self, input_dirc, output_dirc='', Groundwater=False, study_area='', project_name=''):
        """This function can create BNMR dataframe in cases that we don'g have geology or there is limited geology information in the site. You can enter output directory to export the results as a txt file"""
        BNMR_dataframe = pd.DataFrame()
        No_Geo = []
        for Borehole_name in self.boreholes_list:
            self.import_data(Borehole_name,input_dirc, Geology=False, Hydrology=False,Groundwater=Groundwater)
            clear_output()
            d9 = self.d4
            d9 = d9.drop(d9[d9['depth']<0].index) 
            d9['TOP'] = d9.depth - 0.22/2
            d9.TOP = d9.TOP.round(3)
            d9['BOTTOM'] = d9.depth + 0.22/2
            d9.BOTTOM = d9.BOTTOM.round(3)
            d9.loc[d9.TOP<0,'TOP']=0
            d9['Borehole'] = Borehole_name
            d9.loc[:, 'Study_Area'] = study_area
            d9.loc[:,'Project'] = project_name
            d9.loc[:,'immobile'] = d9.loc[:,'clayf']+d9.loc[:,'capf']
            # create saturation column
            if Groundwater==True:
                d9.loc[d9.TOP > self.water_depth.loc[self.water_depth.Borehole==Borehole_name , 'water_depth'].values[0] ,'saturated'] = 'yes'
                d9.loc[d9.TOP <= self.water_depth.loc[self.water_depth.Borehole==Borehole_name , 'water_depth'].values[0] ,'saturated'] = 'no'
            else:
                d9.loc[:,'saturated']=''
            # Appending the new labled boreholes to previous ones
            BNMR_dataframe = pd.concat([d9,BNMR_dataframe], axis=0)
            BNMR_dataframe.reset_index(drop=True,inplace=True)
        print('Task completed.')
        if output_dirc!='':
            BNMR_dataframe.to_csv(output_dirc+'\\BNMR_Dataframe(before_Geo-labeling).txt', sep='\t', index=False)
            print('The data file is exported.')
        return BNMR_dataframe
     
    def geo_lable(self, input_dirc, output_dirc='', Groundwater=True, study_area='', project_name=''):
        """This function can geo-lable all of them. You can enter output directory to export the results as a txt file"""
        d_geo_lable = pd.DataFrame()
        No_Geo = []
        for Borehole_name in self.boreholes_list:
            self.import_data(Borehole_name,input_dirc, Geology=True ,Groundwater=Groundwater)
            clear_output()
            if self.BH_Geo[self.BH_Geo['Borehole']==Borehole_name].empty == True:
                No_Geo.append(Borehole_name)
                continue                
            d9 = self.d4
            d9 = d9.drop(d9[d9['depth']<0].index) 
            d9['TOP'] = d9.depth - 0.22/2
            d9.TOP = d9.TOP.round(3)
            d9['BOTTOM'] = d9.depth + 0.22/2
            d9.BOTTOM = d9.BOTTOM.round(3)
            d9.loc[d9.TOP<0,'TOP']=0
            d9['Borehole'] = Borehole_name
            d9['ROCKSYMBOL'] = ''
            d9.loc[:,'Study_Area'] = study_area
            d9.loc[:,'Project'] = project_name
            d9.loc[:,'immobile'] = d9.loc[:,'clayf']+d9.loc[:,'capf']
            # create saturation column
            if Groundwater==True:
                d9.loc[d9.TOP > self.water_depth.loc[self.water_depth.Borehole==Borehole_name , 'water_depth'].values[0] ,'saturated'] = 'yes'
                d9.loc[d9.TOP <= self.water_depth.loc[self.water_depth.Borehole==Borehole_name , 'water_depth'].values[0] ,'saturated'] = 'no'
            else:
                d9.loc[:,'saturated']=''
            # Labling the  data
            d9_new = pd.DataFrame()
            counter = -1
            for i in range(len(d9)):
                for j in range(len(self.BH_Geo[self.BH_Geo['Borehole']==Borehole_name])):
                    if (pd.Interval(d9.TOP.iloc[i], d9.BOTTOM.iloc[i])).overlaps(pd.Interval(self.BH_Geo[self.BH_Geo['Borehole']==Borehole_name].TOP.iloc[j],self.BH_Geo[self.BH_Geo['Borehole']==Borehole_name].BOTTOM.iloc[j])):
                        counter+=1
                        d9_new = pd.concat([d9_new[:],d9.iloc[i]], axis=1, join='outer', ignore_index=True)
                        d9_new.transpose()['ROCKSYMBOL'].iloc[counter] = str(self.BH_Geo[self.BH_Geo['Borehole']==Borehole_name].ROCKSYMBOL.iloc[j]) 
            # Removing the duplicatd rows (those that all columns are the same due to any problem in borehole geology data, such as  repeating a same geology twice in a 22cm zone that is possible in some cases)
            d9_new = d9_new.transpose()
            d9_new.drop_duplicates(keep='first',inplace=True)
            d9_new.reset_index(drop=True,inplace=True)
            # Appending the new labled boreholes to previous ones
            d_geo_lable = pd.concat([d9_new,d_geo_lable], axis=0)
            d_geo_lable.reset_index(drop=True,inplace=True)
        print('Task completed.')
        if No_Geo !=[]:
            print('This Borehole has no geology data:', No_Geo)        
        if output_dirc!='':
            d_geo_lable.to_csv(output_dirc+'\\BNMR_Dataframe(with_Geo-labels).txt', sep='\t', index=False)
            print('The data file is exported.')
        return d_geo_lable
      
    def geo_lable_T2D(self, input_dirc, output_dirc='', Groundwater=True, study_area='', project_name=''):
        """This function create a dataframe for comparing T2 distribution for all depths and all geologies. You can enter output directory to export the results as a txt file"""
        T2D = pd.DataFrame()
        for Borehole_name in self.boreholes_list:
            self.import_data(Borehole_name,input_dirc, Geology=True, Groundwater=True)
            clear_output()
            No_Geo = []
            if self.BH_Geo[self.BH_Geo['Borehole']==Borehole_name].empty == True:
                No_Geo.append(Borehole_name)
                continue        
            self.d7.rename(columns={0:'depth'}, inplace=True)
            self.d7 = self.d7.drop(self.d7[self.d7['depth']<0].index)
            d9 = self.d7
            d9['TOP'] = self.d7.depth - 0.22/2
            d9.TOP = self.d7.TOP.round(3)
            d9['BOTTOM'] = self.d7.depth + 0.22/2
            d9.BOTTOM = d9.BOTTOM.round(3)
            d9.loc[d9.TOP<0,'TOP']=0
            d9['Borehole'] = Borehole_name
            d9['ROCKSYMBOL'] = ''
            d9.loc[:,'Study_Area'] = study_area
            d9.loc[:,'Project'] = project_name
            # create saturation column
            if Groundwater==True:
                d9.loc[d9.TOP > self.water_depth.loc[self.water_depth.Borehole==Borehole_name , 'water_depth'].values[0] ,'saturated'] = 'yes'
                d9.loc[d9.TOP <= self.water_depth.loc[self.water_depth.Borehole==Borehole_name , 'water_depth'].values[0] ,'saturated'] = 'no'
            else:
                d9.loc[:,'saturated']=''
            # Labling the  data
            d9_new = pd.DataFrame()
            counter = -1
            for i in range(len(d9)):
                for j in range(len(self.BH_Geo[self.BH_Geo['Borehole']==Borehole_name])):
                    if (pd.Interval(d9.TOP.iloc[i], d9.BOTTOM.iloc[i])).overlaps(pd.Interval(self.BH_Geo[self.BH_Geo['Borehole']==Borehole_name].TOP.iloc[j],self.BH_Geo[self.BH_Geo['Borehole']==Borehole_name].BOTTOM.iloc[j])):
                        counter+=1
                        d9_new = pd.concat([d9_new[:],d9.iloc[i]], axis=1, join='outer', ignore_index=True)
                        d9_new.transpose()['ROCKSYMBOL'].iloc[counter] = str(self.BH_Geo[self.BH_Geo['Borehole']==Borehole_name].ROCKSYMBOL.iloc[j])
            # Removing the duplicatd rows (those that all columns are the same due to any problem in borehole geology data, such as  repeating a same geology twice in a 22cm zone that is possible in some cases)
            d9_new = d9_new.transpose()
            d9_new.drop_duplicates(keep='first',inplace=True)
            d9_new.reset_index(drop=True,inplace=True)
            # adding the bins
            d11 = 10**(self.d6)
            d10 = pd.DataFrame()
            for j in range(len(d9_new.depth)):
                for i in range(len(d11.T)):
                    new_row = pd.DataFrame([d9_new.depth.iloc[j], d11[i].iloc[0], d9_new[i+1].iloc[j], d9_new['Borehole'].iloc[j],
                                            d9_new['ROCKSYMBOL'].iloc[j], d9_new['Study_Area'].iloc[j], d9_new['Project'].iloc[j],
                                            d9_new[i+1].iloc[j] / d9_new.T[j].iloc[1:101].max(), d9_new[i+1].iloc[j] / d9_new.T[j].iloc[1:101].sum()]).T
                    d10 = pd.concat([d10,new_row], axis=0, join='outer', ignore_index=True)
            d10.rename(columns={0:'depth',1:'bincenter',2:'binvalue',3:'Borehole',4:'ROCKSYMBOL',5:'Study_Area',6:'Project', 7:'binvalue_n_max',8:'binvalue_n_area'}, inplace=True)
            # Appending the new labled boreholes to previous ones
            T2D = pd.concat([d10,T2D], axis=0)
            T2D.reset_index(drop=True,inplace=True)      
        print('Task completed.')
        if No_Geo !=[]:
            print('This Borehole has no geology data:', No_Geo)
        if output_dirc!='':
            T2D.to_csv(output_dirc+'\\T2D_Dataframe(with_Geo-labels).txt', sep='\t', index=False)
            print('The data file is exported.')
        return T2D
    
    def geo_lable_SpinEchoDecay(self, input_dirc, output_dirc='', Groundwater=True, study_area='', project_name=''):
        """This function create a dataframe for the spin echo decay data for all depths and all geologies. You can enter output directory to export the results as a txt file"""
        SpinEchoes = pd.DataFrame()
        for Borehole_name in self.boreholes_list:
            self.import_data(Borehole_name,input_dirc, Geology=True, Groundwater=True)
            clear_output()
            No_Geo = []
            if self.BH_Geo[self.BH_Geo['Borehole']==Borehole_name].empty == True:
                No_Geo.append(Borehole_name)
                continue        
            self.d1.rename(columns={0:'depth'}, inplace=True)
            self.d1 = self.d1.drop(self.d1[self.d1['depth']<0].index)
            d9 = self.d1
            d9['TOP'] = self.d1.depth - 0.23/2
            d9.TOP = self.d1.TOP.round(3)
            d9['BOTTOM'] = self.d1.depth + 0.23/2
            d9.BOTTOM = d9.BOTTOM.round(3)
            d9.loc[d9.TOP<0,'TOP']=0
            d9['Borehole'] = Borehole_name
            d9['ROCKSYMBOL'] = ''
            d9.loc[:,'Study_Area'] = study_area
            d9.loc[:,'Project'] = project_name
            # create saturation column
            if Groundwater==True:
                d9.loc[d9.TOP > self.water_depth.loc[self.water_depth.Borehole==Borehole_name , 'water_depth'].values[0] ,'saturated'] = 'yes'
                d9.loc[d9.TOP <= self.water_depth.loc[self.water_depth.Borehole==Borehole_name , 'water_depth'].values[0] ,'saturated'] = 'no'
            else:
                d9.loc[:,'saturated']=''
            # Labling the  data
            d9_new = pd.DataFrame()
            counter = -1
            for i in range(len(d9)):
                for j in range(len(self.BH_Geo[self.BH_Geo['Borehole']==Borehole_name])):
                    if (pd.Interval(d9.TOP.iloc[i], d9.BOTTOM.iloc[i])).overlaps(pd.Interval(self.BH_Geo[self.BH_Geo['Borehole']==Borehole_name].TOP.iloc[j],self.BH_Geo[self.BH_Geo['Borehole']==Borehole_name].BOTTOM.iloc[j])):
                        counter+=1
                        d9_new = pd.concat([d9_new[:],d9.iloc[i]], axis=1, join='outer', ignore_index=True)
                        d9_new.transpose()['ROCKSYMBOL'].iloc[counter] = str(self.BH_Geo[self.BH_Geo['Borehole']==Borehole_name].ROCKSYMBOL.iloc[j])
            # Removing the duplicatd rows (those that all columns are the same due to any problem in borehole geology data, such as  repeating a same geology twice in a 22cm zone that is possible in some cases)
            d9_new = d9_new.transpose()
            d9_new.drop_duplicates(keep='first',inplace=True)
            d9_new.reset_index(drop=True,inplace=True)
            # adding the bins
            d11 = self.d2
            d10 = pd.DataFrame()
            for j in range(len(d9_new.depth)):
                for i in range(len(d11)):
                    new_row = pd.DataFrame([d9_new.depth.iloc[j], d11[0].iloc[i], d9_new[i+1].iloc[j], d9_new['Borehole'].iloc[j],
                                            d9_new['ROCKSYMBOL'].iloc[j], d9_new['Study_Area'].iloc[j], d9_new['Project'].iloc[j]]).T
                    d10 = pd.concat([d10,new_row], axis=0, join='outer', ignore_index=True)
            d10.rename(columns={0:'depth',1:'time',2:'signal',3:'Borehole',4:'ROCKSYMBOL',5:'Study_Area',6:'Project'}, inplace=True)
            # Appending the new labled boreholes to previous ones
            SpinEchoes = pd.concat([d10,SpinEchoes], axis=0)
            SpinEchoes.reset_index(drop=True,inplace=True)      
        print('Task completed.')
        if No_Geo !=[]:
            print('This Borehole has no geology data:', No_Geo)
        if output_dirc!='':
            SpinEchoes.to_csv(output_dirc+'\\SpinEchoDecay_Dataframe(with_Geo-labels).txt', sep='\t', index=False)
            print('The data file is exported.')
        return SpinEchoes
    
    def export_kml (self, kml_export_directory, input_dirc, boreholes_coordinates_file, coordinate_system = "EPSG:32632", Geology=True, Hydrology=False, Groundwater=False, wc_range=(0,100), T2D_plot_type=0, Geo_variable='ROCKSYMBOL', Geology_Description='DESCRIPTION',
                    K_plot_params = {'plotting':'None', 'k_field_method':'Slug test', 'k_limit':{'kmin':None, 'kmax':None}, 'SDR_dict': {'show':False,'b':0,'m':0,'n':0}, 'TC_dict': {'show':False, 'c':0}, 'SOE_dict' : {'show':False, 'c':0,'d':0}}):
        """To create a kml file with BNMR PNG plots embedded into it, input_data (the VC export directory), boreholes coordinates file, and coordinate system. You can also specify if you have geology and hydrology data inputs available to consider them in the plots."""
        # creating temporary exports to add them in kml file
        with tempfile.TemporaryDirectory() as tmp_dir:
            XYZ_Boreholes = self.import_coordinates(boreholes_coordinates_file)
            # create a geometry column with Point objects to convert dataframe to geodataframe
            geometry = [Point(xy) for xy in zip(XYZ_Boreholes['X'], XYZ_Boreholes['Y'])]
            gdf = gpd.GeoDataFrame(XYZ_Boreholes, geometry=geometry, crs=pyproj.CRS(coordinate_system))
            # converting the current coordiante system of the data to geographical coordinate system 
            if coordinate_system != "EPSG:4326":
                utm_proj = pyproj.CRS(coordinate_system)
                geo_proj = pyproj.CRS("EPSG:4326")
                transformer = pyproj.Transformer.from_crs(crs_from=utm_proj, crs_to=geo_proj)
                for i in range(len(gdf)):
                    gdf.loc[i,['latitude','longitude']] = transformer.transform(xx=gdf.loc[i,'X'], yy=gdf.loc[i,'Y'])
                # the path to the png files
                gdf['PNG_path_file']= tmp_dir + '/' + gdf['Borehole'].astype(str)+ '.png'
                # specific path format that works in kml file
                gdf['kml_path'] = 'file:///' + gdf['PNG_path_file']
            # creating the PNG plots
            self.logplot(input_dirc, output_dirc=tmp_dir, Geology=Geology, Hydrology=Hydrology, Groundwater=Groundwater, wc_range=wc_range, T2D_plot_type=T2D_plot_type, Geo_variable=Geo_variable,Geology_Description=Geology_Description,  K_plot_params = K_plot_params)          
            # Create a KML object
            kml = simplekml.Kml()
            # Add a folder to hold the images
            folder = kml.newfolder(name='Boreholes')
            # Loop over the points in the geodatabase
            for i in range(len(gdf)):
                # Get the coordinates of the point
                longitude, latitude = gdf.loc[i,'longitude'], gdf.loc[i,'latitude']
                # Get the name of the corresponding image file
                image_file = gdf.loc[i, 'PNG_path_file']
                # Check if the image file exists
                if os.path.isfile(image_file):
                    # Create a placemark for the point
                    placemark = folder.newpoint(name=gdf.loc[i, 'Borehole'], coords=[(longitude, latitude)])
                    # Set the color and size of the point
                    placemark.style.iconstyle.color = simplekml.Color.green  # set the color to red
                    placemark.style.iconstyle.scale = 1  # set the size to 1
                    with open(image_file, 'rb') as f:
                        image_data = f.read()
                        image_b64 = base64.b64encode(image_data).decode()
                    # Create a description for the placemark with the image
                    description = f'<img style="max-width:1200px;" src="data:image/png;base64,{image_b64}">'
                    placemark.description = description
            # Save the KML file
            kml.save(kml_export_directory + '/BNMR_Logplots.kml')
            print('KML file created.')
        return


##########################################################################################################
class Statistics:
    """This class is created to so statistical analysis and several types of plots. Note that the input is the filtered geo-labled database. You may also filter the 
    geological units that are not commonly found, i.e. rare occurences"""
    def __init__(self, geolabled_database):
        self.geolabled_database = geolabled_database  # a dataframe containing all of the data for statistical analysis 

    def qc_dataset (self, bin_size = 2.5, thresholds=[10,15,20], output_dirc=''):
        """It could be BNMR or geolabeled BNMR dataframe. Threshold defines the threshold between noises in the histogram plot. 
        Please ensure that the thresholds are integer multiples of the bin_size."""
        df2 = self.geolabled_database
        df1 = df2.drop_duplicates(subset=['depth', 'Borehole','Study_Area'], keep=False).reset_index(drop=True)
        print( '****************************************************************************************************************')
        print('*** Basic info about the dataset ***')
        print('Number of rows (including duplicates): ', len(df2))
        print('Number of rows: ', len(df1))
        print('Note that the plots are based on the dataset without duplicates.')
        if len(df1[df1.totalf>100]) !=0:
            print('Warning: \nThere are ', len(df1[df1.totalf>=100]), ' BNMR measurements with WC \u2265 100% !')
        print( '****************************************************************************************************************')
        # Define table data
        col1 = ['Total Number\n of Data']
        col2 = [str('Data with \nnoise \u2266 '+str(thresholds[0])+'%')]
        col3 = [str('Data with \n  '+str(thresholds[0])+'% < noise \u2266 '+str(thresholds[1])+'%')]
        col4 = [str('Data with \n '+str(thresholds[1])+'% < noise \u2266 '+str(thresholds[2])+'%')]
        col5 = [str('Data with \nnoise > '+str(thresholds[2])+'%')]
        columns_names = col1 + col2 + col3 + col4 + col5
        rows_names = ['Entire \nBNMR dataset']
        table_values = [[len(df1), len(df1[df1.noise <= thresholds[0]]), len(df1[(df1.noise > thresholds[0]) & (df1.noise <= thresholds[1])]),
                        len(df1[(df1.noise > thresholds[1]) & (df1.noise <= thresholds[2])]), len(df1[df1.noise > thresholds[2]])]]
        # Create table plot
        fig, ax = plt.subplots(figsize=(5, 3))
        colColours=['lime','yellow','orange','red']
        ax.axis('off')
        ax.set_title('Noise in the BNMR Dataset', y=0.7)
        table = ax.table(cellText=table_values, rowLabels=rows_names, colLabels=columns_names, 
                        cellLoc='center', rowLoc='center', loc='center', fontsize=100,
                        colColours=['silver']+colColours, rowColours=['silver','silver'])
        table.scale(2,2)
        if output_dirc !='':
                plt.savefig(str(output_dirc + '\\' + 'QC_Table.png'), bbox_inches='tight', dpi=600)
        plt.show()
        # Histogram & CFD of noise
        fig2, ax3 = plt.subplots(1, 2, figsize=(9, 4))
        data_A = df1['noise']
        sorted_data_A = data_A.sort_values()
        cumulative_A = sorted_data_A.rank() / len(sorted_data_A)
        ax3[1].plot(sorted_data_A, cumulative_A, '-o', markersize=3, color='tab:olive')
        ax3[1].set_xlabel('Noise (%)')
        ax3[1].set_ylabel('Cumulative Probability')
        ax3[1].set_title('CDF')
        ax3[1].set(ylim=(0,1.05))
        ax3[1].grid(True)
        bin_range = (0, ((df1['noise'].max() // bin_size) + 1) * bin_size)
        bins = np.arange(bin_range[0], bin_range[1] + bin_size, bin_size)
        for color_label, condition in zip(colColours, [data_A < thresholds[0], (data_A < thresholds[1]) & (data_A >= thresholds[0]), 
                                                       (data_A < thresholds[2]) & (data_A >= thresholds[1]), data_A >= thresholds[2]]):
            ax3[0].hist(data_A[condition], edgecolor='k', linewidth=0.5, bins=bins, alpha=1, color=color_label)
        ax3[0].set_xlabel('Noise (%)')
        ax3[0].set(xlim=(0,None))
        ax3[0].set_ylabel('Frequency')
        ax3[0].set_title('Histogram')
        plt.suptitle('CFD and Histogram of Noise')
        if output_dirc !='':
                plt.savefig(str(output_dirc + '\\' + 'QC_Figure.png'),bbox_inches='tight', dpi=600)
        return
          
    def statistical_report(self, output_dirc='', grouping_variable = 'ROCKSYMBOL'):
        """This function provides basic statistic results in an excel file"""
        
        #Calculations
        self.geolabled_database[['totalf', 'clayf', 'capf', 'freef', 'immobile', 'mlT2']]= self.geolabled_database[['totalf', 'clayf', 'capf', 'freef', 'immobile', 'mlT2']].astype(float)
        basic_stats = self.geolabled_database[['totalf', 'clayf', 'capf', 'freef', 'immobile', 'mlT2', grouping_variable]].groupby(grouping_variable).describe().T
        skewness_data = self.geolabled_database[['totalf', 'clayf', 'capf', 'freef', 'immobile', 'mlT2', grouping_variable]].groupby(grouping_variable).skew().T
        kurtosis_data = self.geolabled_database[['totalf', 'clayf', 'capf', 'freef', 'immobile', 'mlT2', grouping_variable]].groupby(grouping_variable).apply(pd.DataFrame.kurtosis).T
        median_data = self.geolabled_database[['totalf', 'clayf', 'capf', 'freef', 'immobile', 'mlT2', grouping_variable]].groupby(grouping_variable).apply(pd.DataFrame.median).T
        def mad(x): # a function to calculate the Median Absolute Deviation (MAD)
            median = x.median()
            return (x - median).abs().median()
        mad_data = self.geolabled_database[['totalf', 'clayf', 'capf', 'freef', 'immobile', 'mlT2', grouping_variable]].groupby(grouping_variable).apply(mad).T
        print('Main Statistical Parameters:')
        print(basic_stats)
        print('\nMedian:')
        print(median_data)
        print('\nMedian Absolute Deviation (MAD):')
        print(mad_data)
        print('\nSkewness:')
        print(skewness_data)
        print('\nKurtosis:')
        print(kurtosis_data)
        if output_dirc!='':      
            with pd.ExcelWriter(output_dirc+'\\'+'Statistical_report_Grouped_by_'+grouping_variable+'.xlsx', engine='xlsxwriter') as writer:
                basic_stats.to_excel(writer, sheet_name='Basic Stats', index=True)
                skewness_data.to_excel(writer, sheet_name='Skewness', index=True)
                kurtosis_data.to_excel(writer, sheet_name='Kurtosis', index=True)
                median_data.to_excel(writer, sheet_name='Median', index=True)
                mad_data.to_excel(writer, sheet_name='Median Absolute Deviation', index=True)
            writer.save()

    def boxplot_WC(self, variable = 'ROCKSYMBOL', variable_list=[], output_dirc='', legend_title = 'Geology'):
        """Boxplot for water partioning data for each geology. Variable is the column that grouping is based upon. Note that you can set whis=1.5 and showfliers to True to get outliers plotted."""
        if variable_list == []:
            variable_list = self.geolabled_database[variable].unique()
        for i in variable_list:
            v = self.geolabled_database[self.geolabled_database[variable] == i]
            subdata = v[['clayf','capf','freef','totalf']].astype(float)
            columns_names = ['Clay-bound','Capillary-bound','Free water','Total water']
            rows_names = ['Minimum','Maximum','Mean','Median','STD']
            values = [[subdata.clayf.min() , subdata.capf.min() , subdata.freef.min() , subdata.totalf.min()],
                      [subdata.clayf.max() , subdata.capf.max() , subdata.freef.max() , subdata.totalf.max()],
                     [subdata.clayf.mean() , subdata.capf.mean() , subdata.freef.mean() , subdata.totalf.mean()],
                      [subdata.clayf.median() , subdata.capf.median() , subdata.freef.median() , subdata.totalf.median()],
                      [subdata.clayf.std() , subdata.capf.std() , subdata.freef.std() , subdata.totalf.std()]]
            #table size
            table_size=0.6
            fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7,8))
            # rectangular box plot
            boxprops = dict(linestyle='-', linewidth=1, color='k')
            medianprops = dict(linestyle='--', linewidth=1, color='k')
            meanprops = dict(marker='x', markerfacecolor='k', markeredgecolor='k')
            bplot1 = ax1.boxplot(subdata,vert=True, patch_artist=True, showfliers=False, whis=[0, 100], boxprops=boxprops, medianprops=medianprops,
                                 showmeans=True ,meanprops=meanprops) 
            colors = [ '#DEB887', '#00FFFF', '#0000FF' , '#8A8A8A'] 
            for patch, color in zip(bplot1['boxes'], colors):
                patch.set_facecolor(color)
            plt.ylim(0,100)
            plt.xticks([])
            plt.ylabel('Porosity (%)')
            table1 = plt.table(cellText=np.round(values,2), rowLabels=rows_names, colLabels=columns_names, edges='closed', 
                      cellLoc = 'center', rowLoc = 'center', bbox=[0,-1*table_size,1,table_size])
            table1.set_fontsize(14)
            # table1.scale(1, 0.5)
            plt.plot([],[], linestyle=medianprops['linestyle'], linewidth=medianprops['linewidth'], color=medianprops['color'], label='Median')
            plt.plot([],[], linestyle='',marker=meanprops['marker'], markerfacecolor=meanprops['markerfacecolor'], markeredgecolor=meanprops['markeredgecolor'], label='Mean')
            plt.figtext(0.5,0.95, "Boxplot of Water Partitioning Data", ha="center", va="top", fontsize=13, color="k")
            plt.figtext(0.5,0.91, str(legend_title + ": " + i) , ha="center", va="top", fontsize=12, weight='bold', color="k")
            nod = "n = "+ str(v[v[variable] == i].totalf.count())
            plt.figtext(0.91,0.91, nod, ha="center", va="top", fontsize=8, color="k")
            plt.legend(fontsize=9)
            plt.subplots_adjust(bottom=0.4)
            clear_output()
            if output_dirc !='':
                plt.savefig(str(output_dirc + '\\' + 'Boxplot_' + str(i) + '.png'), bbox_inches='tight', dpi=600)
        print('Boxplot_WC completed.')
        fig = plt.gcf()
        return fig

    def bhdepth_violinplot(self, scale= "count", bw=0.1, swarmplot='No', output_dirc=''):
        """ The violin plot of depth in different areas. You can define the output for saving the figure"""
        d = self.geolabled_database
        d_new = pd.DataFrame()
        for study_area in d['Study_Area'].unique():
            d2=d[d.Study_Area==study_area]
            d2.reset_index(drop=True,inplace=True)
            for bhi in d2.Borehole.unique():  
                d_new = pd.concat([d_new,d2.iloc[d2[d2.Borehole ==bhi].depth.astype(float).idxmax(axis=0)]], axis=1)
        d_new = d_new.transpose().reset_index(drop=True) 
        # final table with maximum depth
        d_new = d_new[['Study_Area','depth','Project','Borehole']]
        table_data = []
        columns_names = d_new['Study_Area'].unique().tolist()
        rows_names = ['Total Boreholes','Minimum Depth','Maximum Depth','Mean Depth']
        for i in range(len(columns_names)):
            subdata1 = d_new[d_new['Study_Area']==columns_names[i]] 
            table_data.append([np.round((len(subdata1)),0), subdata1.depth.min(),subdata1.depth.max(),subdata1.depth.mean()])
        table_data = [list(x) for x in zip(*table_data)]
        d_new['depth'] = d_new['depth'].astype(float)
        # table size
        table_size=0.5
        table1 = plt.table(cellText=np.round(table_data,2), rowLabels=rows_names, colLabels=columns_names, edges='closed', 
        cellLoc = 'center', rowLoc = 'center', bbox=[0,-1*table_size,1,table_size])
        table1.set_fontsize(9)
        table1.scale(1, 0.5)
        ax1 = sns.violinplot(data=d_new, x="Study_Area", y="depth", color='grey', bw=bw, width=1, cut=0, scale= "count", inner=None,
        trim=True, linewidth=0.1, showmeans = False, showextrema = False, showmedians = False)
        ax1.set(xlabel="")
        if swarmplot!='No':
            ax1 = sns.swarmplot(data=d_new, x="Study_Area", y="depth", color='k', size=2.5)
        ax1.set(xlabel="")
        plt.ylim(0,)
        plt.xticks([])
        plt.ylabel('Depth (m)')
        plt.figtext(0.5,0.95, "Depth of Measured Boreholes", ha="center", va="top", fontsize=12, color="k")
        plt.subplots_adjust(left=0.25, bottom=0.4)
        if output_dirc !='':
            plt.savefig(str(output_dirc + '\\' + 'Violinplot_depth.png'), bbox_inches='tight', dpi=600)
        fig = plt.gcf()
        clear_output()
        return fig 

    def water_fractions_pairplot (self, diag_kind="kde", hue='ROCKSYMBOL', hue_dict_hue='tab10', s=30, alpha=0.5, output_dirc='', legend_title = 'Geology'):
        """The inputs are: diag_kind="kde"or"hist", hue_dict_hue, s (symbol size), alpha, and output_dirc"""
        d = self.geolabled_database
        if diag_kind =="hist":
            diag_kws={"bins": 40, "fill":False, "linewidth":1}
        elif diag_kind =="kde":
            diag_kws={"fill":False, "linewidth":1, "common_norm": False}
        ax = sns.pairplot(data = d, x_vars=['clayf','capf','freef','totalf'], y_vars=['clayf','capf','freef','totalf'], hue=hue, aspect=1,height=2,
                          diag_kind=diag_kind, diag_kws=diag_kws, palette=hue_dict_hue, plot_kws=dict(s=s, alpha=alpha), corner=True )
        ax.set(xlim=[0,100])
        ax.set(ylim=[0,100])
        ax.set(xticks=[0,20,40,60,80,100]) 
        ax.set(yticks=[0,20,40,60,80,100])
        ax.tight_layout()
        ax.axes[0][0].set_ylabel('Clay-bound (%)')
        ax.axes[1][0].set_ylabel('Capillary-bound (%)')
        ax.axes[2][0].set_ylabel('Free (%)')
        ax.axes[3][0].set_ylabel('Total (%)')
        ax.axes[3][0].set_xlabel('Clay-bound (%)')
        ax.axes[3][1].set_xlabel('Capillary-bound (%)')
        ax.axes[3][2].set_xlabel('Free (%)')
        ax.axes[3][3].set_xlabel('Total (%)')
        ax.axes[1][0].plot([0,100],[100,0], linestyle='--',color='k')
        ax.axes[1][0].annotate('Non-physical',xy=(40,40),rotation=-45)
        ax.axes[2][0].plot([0,100],[100,0], linestyle='--',color='k')
        ax.axes[2][1].annotate('Non-physical',xy=(40,40),rotation=-45)
        ax.axes[2][1].plot([0,100],[100,0], linestyle='--',color='k')
        ax.axes[2][0].annotate('Non-physical',xy=(40,40),rotation=-45)
        ax.axes[3][0].plot([0,100],[0,100], linestyle='--',color='k')
        ax.axes[3][0].annotate('Non-physical',xy=(50,20),rotation=45)        
        ax.axes[3][1].plot([0,100],[0,100], linestyle='--',color='k')
        ax.axes[3][1].annotate('Non-physical',xy=(50,20),rotation=45)        
        ax.axes[3][2].plot([0,100],[0,100], linestyle='--',color='k')
        ax.axes[3][2].annotate('Non-physical',xy=(50,20),rotation=45)     
        ax.fig.suptitle('Water Fractions Pairplot', y=0.98, fontsize=13)    
        ax.axes[0][0].plot([0,0],[0,100], linestyle='-',color='k')
        ax.axes[1][1].plot([0,0],[0,100], linestyle='-',color='k')
        ax.axes[2][2].plot([0,0],[0,100], linestyle='-',color='k')
        ax.axes[3][3].plot([0,0],[0,100], linestyle='-',color='k')
        ax._legend.set_title(legend_title)
        ax.tight_layout()
        fig = plt.gcf()
        clear_output()
        if output_dirc !='':
            plt.savefig(str(output_dirc + '\\' + 'water_fractions_pairplot.png'), bbox_inches='tight', dpi=600)
        return fig 
    
    def mobile_immobile_pairplot (self, diag_kind="kde", hue='ROCKSYMBOL', hue_dict_hue='tab10', s=30, alpha=0.5, output_dirc='', legend_title = 'Geology'):
        """The inputs are: diag_kind="kde"or"hist", hue_dict_hue, s (symbol size), alpha, and output_dirc"""
        d = self.geolabled_database
        if diag_kind =="hist":
            diag_kws={"bins": 40, "fill":False, "linewidth":1}
        elif diag_kind =="kde":
            diag_kws={"fill":False, "linewidth":1, "common_norm": False}
        ax = sns.pairplot(data = d, x_vars=['immobile','freef','totalf'], y_vars=['immobile','freef','totalf'],
                          hue=hue, aspect=1, height=2, diag_kind=diag_kind, corner=True, diag_kws=diag_kws,
                          palette=hue_dict_hue, plot_kws=dict(s=s, alpha=alpha) )
        ax.set(xlim=[0,100])
        ax.set(ylim=[0,100])
        ax.set(xticks=[0,20,40,60,80,100]) 
        ax.set(yticks=[0,20,40,60,80,100])
        ax.axes[0][0].set_ylabel('Immobile (%)')             
        ax.axes[1][0].set_ylabel('Mobile (%)')
        ax.axes[1][0].plot([100,0],[0,100], linestyle='--',color='k')
        ax.axes[1][0].annotate('Non-physical',xy=(40,40),rotation=-45)
        ax.axes[2][0].set_ylabel('Total (%)')
        ax.axes[2][0].set_xlabel('Immobile (%)')
        ax.axes[2][0].plot([0,100],[0,100], linestyle='--',color='k')
        ax.axes[2][0].annotate('Non-physical',xy=(50,20),rotation=45)
        ax.axes[2][1].set_xlabel('Mobile (%)')
        ax.axes[2][1].plot([0,100],[0,100], linestyle='--',color='k')
        ax.axes[2][1].annotate('Non-physical',xy=(50,20),rotation=45)
        ax.axes[2][2].set_xlabel('Total (%)')        
        ax.axes[0][0].plot([0,0],[0,100], linestyle='-',color='k')
        ax.axes[1][1].plot([0,0],[0,100], linestyle='-',color='k')
        ax.axes[2][2].plot([0,0],[0,100], linestyle='-',color='k')
        ax.fig.suptitle('Mobile vs. Immobile Porosity Pairplot', y=0.98, fontsize=13)
        ax._legend.set_title(legend_title)
        ax.tight_layout()
        fig = plt.gcf()
        clear_output()
        if output_dirc !='':
            plt.savefig(str(output_dirc + '\\' + 'mobile_immobile_pairplot.png'), bbox_inches='tight', dpi=600)
        return fig  
    
    def water_fractions_pairgrid (self, nlevels=5 , fill=True, alpha= 0.5, hue='ROCKSYMBOL', output_dirc='', legend_title = 'Geology'):
        d = self.geolabled_database
        ax2 = sns.PairGrid(d[['clayf','capf','freef','totalf']+[hue]], diag_sharey=False, hue=hue, corner=True)
        ax2.map_upper(sns.scatterplot, s=30, alpha=alpha)
        ax2.map_lower(sns.kdeplot, levels=nlevels, thresh=0.2, alpha=alpha, fill=fill)
        ax2.map_diag(sns.histplot , kde=False, bins=50, fill=False, linewidth=0.5, alpha=alpha)
        ax2.set(xlim=[0,None])
        ax2.set(ylim=[0,None])
        ax2.axes[0][0].set_ylabel('Clay-bound (%)')
        ax2.axes[1][0].set_ylabel('Capillary-bound (%)')
        ax2.axes[2][0].set_ylabel('Free (%)')
        ax2.axes[3][0].set_ylabel('Total (%)')
        ax2.axes[3][0].set_xlabel('Clay-bound (%)')
        ax2.axes[3][1].set_xlabel('Capillary-bound (%)')
        ax2.axes[3][2].set_xlabel('Free (%)')
        ax2.axes[3][3].set_xlabel('Total (%)')
        ax2.fig.suptitle('Water Fractions PairGrid', y=0.98, fontsize=13)
        ax2._legend.set_title(legend_title)
        ax2.tight_layout()
        fig = plt.gcf()
        clear_output()
        if output_dirc !='':
            plt.savefig(str(output_dirc + '\\' + 'water_fractions_pairgrid.png'), bbox_inches='tight', dpi=600)
        return fig   
         
    def mobile_immobile_pairgrid (self, nlevels=5 , fill=True, alpha=0.5, hue='ROCKSYMBOL', output_dirc='', legend_title = 'Geology'):
        d = self.geolabled_database
        d['immobile']=d['clayf']+d['capf']
        ax2 = sns.PairGrid(d[['immobile','freef','totalf']+[hue]],diag_sharey=False, hue=hue, corner=True)
        ax2.map_upper(sns.scatterplot, s=30, alpha=alpha)
        ax2.map_lower(sns.kdeplot, levels=nlevels, thresh=0.2, alpha=alpha, fill=fill)
        ax2.map_diag(sns.histplot , kde=False, bins=50, fill=False, linewidth=0.5, alpha=alpha)
        ax2.set(xlim=[0,None])
        ax2.set(ylim=[0,None])
        ax2.axes[0][0].set_ylabel('Immobile (%)')
        ax2.axes[1][0].set_ylabel('Mobile (%)')
        ax2.axes[2][0].set_ylabel('Total (%)')
        ax2.axes[2][0].set_xlabel('Immobile (%)')
        ax2.axes[2][1].set_xlabel('Mobile (%)')
        ax2.axes[2][2].set_xlabel('Total (%)')
        ax2.fig.suptitle('Mobile vs. Immobile Porosity Parigrid', y=0.98, fontsize=13)
        ax2._legend.set_title(legend_title)
        ax2.tight_layout()
        fig = plt.gcf()
        clear_output()
        if output_dirc !='':
            plt.savefig(str(output_dirc + '\\' + 'mobile_immobile_pairgrid.png'), bbox_inches='tight', dpi=600)
        return fig    

    def ternary_plot (self, opacity=0.7, hue='ROCKSYMBOL', hue_dict_hue=None, output_dirc='',  legend_title='Geology'):
        """Ternary of porosity classes."""
        d = self.geolabled_database
        d['symbol_size']=10
        fig = px.scatter_ternary(d, a="clayf", b="capf", c="freef", color=hue, size='symbol_size', size_max=7, 
                                 opacity=opacity, title= 'Ternary Plot of Water Partitioning Data' , color_discrete_map=hue_dict_hue,
                                labels={"clayf":"Clay-bound Water", "capf":"Capillary-bound Water", "freef":"Free water"},)
        fig.update_layout(legend_title_text=legend_title)
        if output_dirc !='':
                plt.savefig(str(output_dirc + '\\' + 'ternary_plot.png'), bbox_inches='tight', dpi=600)
        return fig
    
    def mlT2_vs_water_fractions (self, hue_dict_hue=None, hue="ROCKSYMBOL", alpha=0.5 ,output_dirc='', legend_title='Geology'):
        """"Plots T2ML versus porosity classes. The input parameters are hue_dict_hue, alpha, and output_dirc"""
        d = self.geolabled_database
        fig, ax = plt.subplots(2, 2, figsize=(8,8))
        fig.suptitle('Water fractions vs. $mlT_{2}$')
        axs=ax[0][0]
        sns.scatterplot(data=d, x="mlT2", y="clayf", hue=hue, palette=hue_dict_hue, alpha=alpha , ax=axs, legend=False)
        axs.set_xlim(0.003,1)
        axs.set_xscale('log')
        axs.set_xlabel('$mlT_{2}$ (s)')
        axs.set_ylim(0,)
        axs.set_ylabel('Clay-bound Water (%)')
        axs.xaxis.set_tick_params(labelbottom=True)
        axs=ax[0][1]
        sns.scatterplot(data=d, x="mlT2", y="capf", hue=hue, palette=hue_dict_hue, alpha=alpha , ax=axs, legend=True)
        axs.set_xlim(0.003,1)
        axs.set_xscale('log')
        axs.set_xlabel('$mlT_{2}$ (s)')
        axs.set_ylim(0,)
        axs.set_ylabel('Capillary-bound Water (%)')
        axs.legend(title=legend_title, bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0)
        axs=ax[1][0]
        sns.scatterplot(data=d, x="mlT2", y="freef", hue=hue, palette=hue_dict_hue, alpha=alpha , ax=axs, legend=False)
        axs.set_xlim(0.003,1)
        axs.set_xscale('log')
        axs.set_xlabel('$mlT_{2}$ (s)')
        axs.set_ylim(0,)
        axs.set_ylabel('Free Water (%)')
        axs=ax[1][1]
        sns.scatterplot(data=d, x="mlT2", y="totalf", hue=hue, palette=hue_dict_hue, alpha=alpha , ax=axs, legend=False)
        axs.set_xlim(0.003,1)
        axs.set_xscale('log')
        axs.set_xlabel('$mlT_{2}$ (s)')
        axs.set_ylim(0,)
        axs.set_ylabel('Total Water (%)')
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        fig = plt.gcf()
        clear_output()
        if output_dirc !='':
            plt.savefig(str(output_dirc + '\\' + 'mlT2_vs_water_fractions.png'), bbox_inches='tight', dpi=600)
        return fig  
    
    def mlT2_vs_other_parameters (self, hue_dict_hue=None, hue="ROCKSYMBOL", alpha=0.5, output_dirc='', legend_title='Geology'):
        """"Plots Noise, SOE, and TotalF versus T2ML. The input parameters are hue_dict_hue, alpha, and output_dirc"""
        d = self.geolabled_database
        fig, ax = plt.subplots(1, 3, figsize=(12,5))
        fig.suptitle('Porosity, SOE, and Noise vs. $mlT_{2}$')
        axs=ax[0]
        sns.scatterplot(data=d, x="mlT2", y="totalf", hue=hue, palette=hue_dict_hue, alpha=alpha , ax=axs, legend=False)
        axs.set_xlim(0.003,1)
        axs.set_xscale('log')
        axs.set_xlabel('$mlT_{2}$ (s)')
        axs.set_ylim(0,100)
        axs.set_ylabel('Porosity (%)')
        axs.xaxis.set_tick_params(labelbottom=True)
        axs=ax[1]
        sns.scatterplot(data=d, x="mlT2", y="soe", hue=hue, palette=hue_dict_hue, alpha=alpha , ax=axs, legend=False)
        axs.set_xlim(0.003,1)
        axs.set_xscale('log')
        axs.set_xlabel('$mlT_{2}$ (s)')
        axs.set_yscale('log')
        axs.set_ylim(0,None)
        axs.set_ylabel('Sum Of Echoes')
        axs=ax[2]
        sns.scatterplot(data=d, x="mlT2", y="noise", hue=hue, palette=hue_dict_hue, alpha=alpha , ax=axs, legend=True)
        axs.set_xlim(0.003,1)
        axs.set_xscale('log')
        axs.set_xlabel('$mlT_{2}$ (s)')
        axs.set_ylim(0,None)
        axs.set_ylabel('Noise (%)')
        axs.legend(title=legend_title, bbox_to_anchor=(1.03, 1), loc='upper left', borderaxespad=0)
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        fig = plt.gcf()
        clear_output()
        if output_dirc !='':
            plt.savefig(str(output_dirc + '\\' + 'mlT2_vs_other_parameters.png'), bbox_inches='tight', dpi=600)
        return fig
    
    def mlT2_kde_hist(self, bins=50, hue='ROCKSYMBOL', hue_dict_hue=None, alpha=0.5, output_dirc=''):
        d = self.geolabled_database
        ax6 = sns.histplot(data=d,x='mlT2', bins=bins, alpha=alpha, hue=hue, palette=hue_dict_hue, fill=False, shrink=1, kde=True, log_scale=True)
        ax6.set(xlim=[0.005,1])
        ax6.set(title='Histogram of Mean Log Transverse Relaxation Rate (ml$T_{2}$)')
        ax6.set(xscale='log')
        fig = plt.gcf()
        clear_output()
        if output_dirc !='':
            plt.savefig(str(output_dirc + '\\' + 'mlT2_kde_hist.png'), bbox_inches='tight', dpi=600)
        return fig
    
    def porosity_vs_T2ML (self, hue='ROCKSYMBOL', hue_order=None, legend_title='Geology',alpha=0.7, s=50, n_min=0, hue_dict_hue=None, output_dirc=''):
        """This function plots total porosity versus T2ML. Input parameters are hue, hue_order, legend_title,alpha, s (symbol size), , n_min is the minimum number of points for each hue class to be plotted in the figure, hue_dict_hue, output_dirc """
        if hue_order!=None:
            self.geolabled_database.sort_values(by=hue, ascending=False, kind='mergesort', inplace=True)
            self.geolabled_database = self.geolabled_database[self.geolabled_database[hue].isin(hue_order)]
            self.geolabled_database.sort_values(by=hue, ascending=False, kind='mergesort', inplace=True)
        subset_data = self.geolabled_database.groupby(hue).filter(lambda x: len(x) >= n_min)
        fig1 = sns.JointGrid(data=subset_data, x='mlT2', y='totalf', hue=hue, palette=hue_dict_hue,
                            hue_order=hue_order, space=0.2, xlim=[0.001,1], ylim=[0,100], ratio=4 )
        fig1.fig.set_size_inches(5, 5)
        fig1 = fig1.plot_joint(sns.scatterplot, hue=hue, palette=hue_dict_hue, hue_order=hue_order, s=s, alpha=alpha)
        sns.kdeplot(data=subset_data, x='mlT2', hue=hue, log_scale=True, palette=hue_dict_hue, legend=False,
                    fill=False, alpha=1, common_norm=False, ax=fig1.ax_marg_x)
        sns.kdeplot(data=subset_data, y='totalf', hue=hue, log_scale=False, palette=hue_dict_hue, legend=False,
                        fill=False, alpha=1, common_norm=False, ax=fig1.ax_marg_y)                                               
        fig1.ax_joint.legend(title=legend_title, loc='center left', bbox_to_anchor=(1.05, 1.15), ncol=1, fontsize=8)
        # drawing the edges of marginal plots
        fig1.ax_marg_x.plot([fig1.ax_marg_x.get_xlim()[0],fig1.ax_marg_x.get_xlim()[0]],fig1.ax_marg_x.get_ylim(),'-k')
        fig1.ax_marg_x.plot([fig1.ax_marg_x.get_xlim()[1],fig1.ax_marg_x.get_xlim()[1]],fig1.ax_marg_x.get_ylim(),'-k')
        fig1.ax_marg_y.plot(fig1.ax_marg_y.get_xlim(),[fig1.ax_marg_y.get_ylim()[0],fig1.ax_marg_y.get_ylim()[0]],'-k')
        fig1.ax_marg_y.plot(fig1.ax_marg_y.get_xlim(),[fig1.ax_marg_y.get_ylim()[1],fig1.ax_marg_y.get_ylim()[1]],'-k')
        xt, yt = 0.6 , 100
        for i in subset_data[hue].unique():
                count=str(len(subset_data[(subset_data[hue]==i)][hue]))
                yt -= 4
                if hue_dict_hue==None:
                    hue_categories = subset_data[hue].unique()    # Get unique hue categories
                    def rgb_to_hex(rgb):
                        """Convert RGB tuple to hexadecimal color code."""
                        return "#{:02x}{:02x}{:02x}".format(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
                    default_palette = sns.color_palette()   # Create color dictionary
                    hue_dict_hue = {label: color for label, color in zip(hue_categories, default_palette)}
                    hue_dict_hue = {label: rgb_to_hex(rgb) for label, rgb in hue_dict_hue.items()}
                fig1.ax_joint.annotate(str("n="+count), xy=(xt, yt), fontsize=7, weight="bold", color=hue_dict_hue[i])
        fig1.ax_joint.set_xlabel('$T_{2ML}$ (s)')
        fig1.ax_joint.set_ylabel('Porosity (%)')
        if n_min>0:
            print(f'The hue values that have less than {n_min} datapoints are not plotted in the figure')
        if output_dirc !='':
            if n_min>0:
                plt.savefig(str(output_dirc + '\\' + 'porosity_vs_T2ML_'+hue+'_n-min_'+str(n_min)+'.png'), bbox_inches='tight', dpi=600)
            else:
                plt.savefig(str(output_dirc + '\\' + 'porosity_vs_T2ML_'+hue+'.png'), bbox_inches='tight', dpi=600)
        return fig1
    
    def kde_comparison_plot(self, hue='ROCKSYMBOL', legend_title='Geology', hue_dict_hue=None, n_min=0, output_dirc=''):
        columns_to_plot = ['totalf', 'freef', 'immobile', 'clayf', 'capf', 'mlT2', 'soe', 'noise']
        subplot_titles = ['Total Porosity', 'Free Porosity', 'Immobile Porosity', 'Clay-bound Porosity',
                        'Capillary-bound Porosity', 'T$_{2ML}$', 'Sum of Echoes', 'Noise']
        subplot_x_titles = ['Total Porosity (%)', 'Free Porosity (%)', 'Immobile Porosity (%)', 'Clay-bound Porosity (%)',
                            'Capillary-bound Porosity (%)', 'T$_{2ML}$', 'Sum of Echoes', 'Noise (%)']
        if hue_dict_hue is None:
            hue_dict_hue = {}
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        axes = axes.flatten()
        legend_labels = []
        for i, column in enumerate(columns_to_plot):
            ax = axes[i]
            if hue:
                kde_data = self.geolabled_database.groupby(hue).filter(lambda x: len(x) >= n_min)
                for hue_value in kde_data[hue].unique():
                    color = hue_dict_hue.get(hue_value, None)  # Get color from dictionary, default to None
                    if column in ['soe', 'mlT2']:
                        sns.kdeplot(data=kde_data[kde_data[hue] == hue_value][column], ax=ax, label=hue_value,
                                    common_norm=False, log_scale=True, linewidth=2, color=color)
                    else:
                        sns.kdeplot(data=kde_data[kde_data[hue] == hue_value][column], ax=ax, label=hue_value,
                                    common_norm=False, log_scale=None, linewidth=2, color=color)
                    if i == 0:  # Only add labels once
                        legend_labels.append(f"{hue_value} ({len(kde_data[kde_data[hue] == hue_value])})")
            else:
                sns.kdeplot(data=kde_data[column], ax=ax, fill=True)
            ax.set(title=subplot_titles[i])
            ax.set(xlabel=subplot_x_titles[i])
            if column not in ['soe', 'mlT2']:
                ax.set_xlim(0, None)
        fig.legend(title=legend_title, labels=legend_labels, loc='center right', bbox_to_anchor=(1.1, 0.8))
        plt.tight_layout()
        if n_min>0:
            print(f'The hue values that have less than {n_min} datapoints are not plotted in the figure')
        if output_dirc !='':
            if n_min>0:
                plt.savefig(str(output_dirc + '\\' + 'kde_comparison_plot_'+hue+'_n-min_'+str(n_min)+'.png'), bbox_inches='tight', dpi=600)
            else:
                plt.savefig(str(output_dirc + '\\' + 'kde_comparison_plot_'+hue+'.png'), bbox_inches='tight', dpi=600)
        return fig

    def calibrate_K_estimation_models (self, K_data_file, min_total_overlap=0, min_individual_overlap=0 , hue=None, hue_dict=None, add_K_to_BNMR_data=False , best_methods=[], output_dirc=''):
        """ Function for calibarting K estimation models that provides calibrated coefficient constants in separate dataframes for SDR, SOE, and TC models:
        - K_data_file: screen intervals (Borehole, TOP, BOTTOM, K, and if available ROCKSYMBOL and K_std). Note that TOP and BOTTOM values are in meters 
        and K in m/s.
        - method= 'all', 'SDR', 'TC', 'SOE'. the default is all.
        - hue can be ROCKSYMBOl, None, or etc. 
        - hue_dict is not included in the code yet"""
        if hue == None:
            hue2 = 'ROCKSYMBOL'
        else:
            hue2 = hue
        if K_data_file[-3:] == 'txt':
            K_data = pd.DataFrame(pd.read_csv(K_data_file, sep='\t'))
        else:
            K_data = pd.DataFrame(pd.read_excel(K_data_file))
        K_data.loc[:,'SCREEN_LENGTH'] = K_data.loc[:,'BOTTOM'] - K_data.loc[:,'TOP']
        K_data.loc[:,'CENTER'] = (K_data.loc[:,'BOTTOM'] + K_data.loc[:,'TOP'])/2
        # Convert specified columns to float
        # Selecting specific columns
        K_data = K_data[['Borehole', 'BOTTOM', 'TOP', 'CENTER', 'SCREEN_LENGTH', 'K', 'K_std']]
        # Specifying columns for string conversion
        str_cols1 = ['Borehole', 'Study_Area']
        # Specifying columns for float conversion
        flt_cols1 = ['BOTTOM', 'TOP', 'CENTER', 'SCREEN_LENGTH', 'K', 'K_std']
        for col in str_cols1:
            if col not in K_data.columns:
                K_data[col] = ''
        for col in str_cols1:
            if col in K_data.columns:
                K_data[col] = K_data[col].astype(str)
        for col in flt_cols1:
            if col in K_data.columns:
                K_data[col] = K_data[col].astype(np.float64)
        data1 = self.geolabled_database.copy(deep=True)
        str_cols2 = [col for col in ['Borehole','ROCKSYMBOL','Study_Area', 'Project','saturated','Geo_Simple'] if col in data1.columns]
        data1[str_cols2] = data1[str_cols2].astype(str)
        flt_cols2 = ['depth', 'totalf', 'clayf', 'capf', 'freef', 'mlT2', 'soe', 'TOP', 'BOTTOM', 'immobile']
        data1[flt_cols2] = data1[flt_cols2].astype(float)
        # Adding the columns in K dataframe to BNMR dataframe
        data1 = add_new_lable(BNMR_dataframe=data1, dataframe_new=K_data, parameters=list(set(K_data.columns) - set(data1.columns)))
        # Building the dataset for curve fitting 
        par = 'K'
        data1 = data1.dropna(subset=[par])
        df = data1
        # dropping problematic rows (non numeric values)
        counter = 0
        if len(df[pd.to_numeric(df[par], errors='coerce').isna()])!=0:
            counter +=1
            print(par, 'column has non_numeric values: \n')
            print(df[pd.to_numeric(df[par], errors='coerce').isna()])
            df.drop(df[pd.to_numeric(df[par], errors='coerce').isna()].index, inplace=True)
        if counter != 0:
            print('\nAll non numeric values are dropped from the BNMR_SCREEN dataframe')
        else:
            print('There is no non-numeric values in BNMR_SCREEN dataframe')
        # Number of points in each geology before aeraging    
        print("\nNumber of K measurements BEFORE averaging for the entire length of screen")
        for rck in data1[hue2].unique():
            print('Number of K measuremtns in', rck, ' : ', len(data1[data1[hue2]==rck]))
        ############################################################
        ### create new database for HC calibration
        ############################################################
        # setting up the types of columns
        numeric_columns = list(set(flt_cols1 + flt_cols2))
        non_numeric_columns = list(set(str_cols1 + str_cols1))
        # take the average measurement of BNMR for each interval of field HC measurement
        screen_df = pd.DataFrame([],columns=data1.columns)
        i = -1
        for bh in data1.Borehole.unique():
            for dph_c in data1[data1.Borehole==bh].CENTER.unique():
                i += 1 
                avg_subset = data1[(data1.Borehole==bh) & (data1.CENTER==dph_c)]
                avg_subset_copy = avg_subset.copy()
                rck_value = sorted(avg_subset[hue2].unique())
                if len(rck_value)>1:
                    rck_value = ','.join(rck_value)
                    rck_value = [f"{rck_value}"]
                screen_df.loc[i,non_numeric_columns] = avg_subset[non_numeric_columns].iloc[0]
                screen_df.loc[i,hue2] = rck_value[0]
                # dropping duplicates due to geological labeling before averaging
                avg_subset_copy.drop_duplicates(subset=[col for col in avg_subset_copy.columns if col != hue2], keep='first', inplace=True)
                
                ## Simple Average
                #screen_df.loc[i,'TOP'] = K_data.loc[(K_data.Borehole == bh)&(K_data.CENTER == dph_c) , 'TOP'].mean()
                #screen_df.loc[i,'BOTTOM'] = K_data.loc[(K_data.Borehole == bh)&(K_data.CENTER == dph_c), 'BOTTOM'].mean()
                #screen_df.loc[i,numeric_columns] = avg_subset_copy[numeric_columns].mean() 

                ### Weighted Average Based on Overlapping Length  ###
                k_top = K_data.loc[(K_data.Borehole == bh)&(K_data.CENTER == dph_c), 'TOP'].mean()
                k_bottom = K_data.loc[(K_data.Borehole == bh)&(K_data.CENTER == dph_c), 'BOTTOM'].mean()
                # Calculate overlap length for each row in avg_subset_copy
                avg_subset_copy["Overlap"] = avg_subset_copy.apply(lambda row: max(0,
                        min(row["BOTTOM"], k_bottom) - max(row["TOP"], k_top)), axis=1 )                
                overlapping_subset = avg_subset_copy[avg_subset_copy["Overlap"] > min_individual_overlap]   # Exclude rows with no overlap
                total_overlap = overlapping_subset["Overlap"].sum()
                screen_df.loc[i, "Total_Overlap"] = total_overlap
                screen_df.loc[i, "Min_individual_Overlap"] = overlapping_subset["Overlap"].min()
                num_contributors = overlapping_subset.shape[0]
                screen_df.loc[i, "Number_BNMR_Points"] = num_contributors

                for col in numeric_columns:
                    if col == 'K': 
                        screen_df.loc[i,'TOP'] = K_data.loc[(K_data.Borehole == bh)&(K_data.CENTER == dph_c) , 'TOP'].mean()
                        screen_df.loc[i,'BOTTOM'] = K_data.loc[(K_data.Borehole == bh)&(K_data.CENTER == dph_c), 'BOTTOM'].mean()
                        screen_df.loc[i, col] = K_data.loc[(K_data.Borehole == bh)&(K_data.CENTER == dph_c), col].mean()
                    else:
                        weighted_sum = (overlapping_subset[col] * overlapping_subset["Overlap"]).sum()
                        weight_total = overlapping_subset["Overlap"].sum()
                        screen_df.loc[i, col] = weighted_sum / weight_total if weight_total != 0 else None

        screen_df = screen_df.astype({col: float for col in numeric_columns})
        screen_df.loc[:,'TOP'] = np.round(screen_df.loc[:,'TOP'],3)
        screen_df.loc[:,'BOTTOM'] = np.round(screen_df.loc[:,'BOTTOM'],3) 

        # defining the minimum required overlap 
        screen_df = screen_df[screen_df.Total_Overlap>=min_total_overlap]      

        # Number of points in each geology (after averaging)
        print("\nNumber of K measurements AFTER averaging for the entire length of screen")
        for rck in screen_df[hue2].unique():
            print('Number of measuremtns in', rck, ' : ', len(screen_df[screen_df[hue2]==rck]))
        print('\nNumber of Boreholes with K measurements:', len(screen_df.Borehole.unique()))
        print('List of Study Areas: ', str(screen_df.Study_Area.unique()).strip('[]'))
        print("\n")
        screen_df.dropna(axis=1, how='all', inplace=True) 
        # Total Porosity & T2ML plots versus K measurements (points after averaging for the screen interval)
        fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True)
        axs = ax[0]  # subplot 1
        sns.scatterplot(data=screen_df, x='K', y='totalf', hue=hue, palette=hue_dict if hue_dict is not None else None, s=60, alpha=.7, ax=axs, legend=True)
        axs.set(xscale='log')
        axs.set(yscale='linear')
        # axs.set(ylim=(0,100))
        axs.set_xlabel('K (m/s)', fontsize=13)
        axs.set_ylabel('Total Porosity (%)', fontsize=13)
        axs.minorticks_on()
        axs.tick_params(axis='x', labelsize=11)
        axs.tick_params(axis='y', labelsize=11)
        # axs.grid(which='major', color='grey', linewidth=0.5)
        axs = ax[1] # subplot 2
        sns.scatterplot(data=screen_df, x='K', y='mlT2', hue=hue, palette=hue_dict if hue_dict is not None else None, s=60, alpha=.7, ax=axs, legend=True)
        axs.set(xscale='log')
        axs.set(yscale='log')
        axs.set_xlabel('K (m/s)', fontsize=13)
        axs.set_ylabel('$T_{2ML}$ (s)', fontsize=13)
        # axs.set(ylim=(1e-3,1))
        axs.minorticks_on()
        axs.tick_params(axis='x', labelsize=11)
        axs.tick_params(axis='y', labelsize=11)
        if hue != None:
            ax[0].legend().remove() # hide the legend of the first subplot
            ax[1].legend().remove()
            # create a legend for the second subplot outside the plot
            handles, labels = axs.get_legend_handles_labels()
            plt.legend(title='LEGEND', handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1, 0.5))
        # axs.grid(which='major', color='grey', linewidth=0.5)
        plt.tight_layout()
        #Saving the figure in a file
        fig.set_facecolor("white")
        if output_dirc!='':
            plt.savefig(output_dirc+'\\'+'K_Calibration_Plot(K_vs_NMR_parameters)', dpi=600)
        plt.show()
        #################################
        ##### SDR Model Calibration #####
        """ The model is: K_sdr = b * PHI**m * T2ML**n """
        MODEL = 'SDR Model'
        SDR_df = pd.DataFrame(columns=['equation','b','m','n','R2',hue])
        i = -1
        if hue == None:
            rck_all = ['None']
        else:
            rck_all = screen_df[hue].unique()
        def K_NMR(x, b, m, n): # Define the function to fit the data
                PHI, T2ML = x
                return np.log10(b) + (PHI*m) + (T2ML* n)
        fig, axs = plt.subplots(1, len(rck_all), figsize=(4*len(rck_all),4), sharey=True, sharex=True)
        for rck in rck_all:
            i += 1
            if len(rck_all)==1:
                selected_screen_df = screen_df.copy()
                ax=axs
            else:
                selected_screen_df = screen_df[screen_df[hue]==rck].copy()
                ax = axs[i]
            print(MODEL)
            print('K = b * PHI**m * T2ML**n')
            if hue == None:
                print('Curve fitting using all points')
            else:
                print('Curve fitting for: ', rck)
            print('Number of datapoints for curve fitting: ', len(selected_screen_df))
            PHI = np.log10(selected_screen_df.totalf.astype(np.float64)/100)
            T2ML = np.log10(selected_screen_df.mlT2.astype(np.float64))
            K = np.log10(selected_screen_df.K.astype(np.float64))
            param_bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
            popt, pcov = curve_fit(K_NMR, (PHI, T2ML), K, p0=[0, 0, 0], bounds=param_bounds) # Perform the curve fit
            # Extract the optimized coefficients
            b = popt[0]
            m = popt[1]
            n = popt[2]
            print('Estimated coefficients')
            print('b (m/s^n+1):', b)
            print('m:', m)
            print('n:', n)
            # plot
            K_pred = K_NMR((PHI, T2ML), b, m, n)
            # Calculate the K_diff_factor
            K_diff_factor = np.mean(np.maximum(((10**K) / (10**K_pred)), ((10**K_pred) / (10**K))))  # Relative difference
            # ax.scatter(10**K, 10**K_pred, color = 'b' if hue_dict is None else hue_dict[rck], s=50, alpha=0.7, label=rck)
            ax.scatter(10**K, 10**K_pred, color = 'b' if hue_dict is None else hue_dict[rck], s=50, alpha=0.7, label=rck)

            ax.set(xscale='log')
            ax.set(yscale='log')
            ax.set_xlabel('K (m/s)', fontsize=13)
            ax.tick_params(axis='x', labelsize=11)
            ax.tick_params(axis='y', labelsize=11)
            ax.set_ylabel('K$_{NMR}$ (m/s)', fontsize=13)
            xlim = [0.5*min(min(10**K),min(10**K_pred)),1.5*max(max(10**K),max(10**K_pred))]
            ylim = xlim
            ax.set(ylim=ylim)
            ax.set(xlim=xlim)
            if len(rck_all)!=1:
                ax.set(title=rck)
            # Y=X and guidelines
            opacity = 0.75
            ax.plot(xlim,xlim, '--k', alpha=0.7, label='K = K$_{NMR}$')
            ax.plot(xlim,[10*xlim[0],10*xlim[1]], linestyle=':', color='gray', alpha=1, label='One OOM higher')
            ax.plot(xlim,[0.1*xlim[0],0.1*xlim[1]], linestyle=':', color='gray', alpha=1, label='One OOM lower')
            # r2 score
            # from sklearn.metrics import r2_score
            # R2 = r2_score(K, K_pred)
            # print('R-squared score:', R2)
            TSS = np.sum((K - np.mean(K))**2)
            RSS = np.sum((K - K_pred)**2)
            R2 = 1 - RSS / TSS
            
            # Mean Squared Error
            MSE = np.mean((K - K_pred)**2)
            print('Mean Squared Error (MSE):', round(MSE,3) , '(m/s)^2')
            # Root Mean Squared Error
            RMSE = np.sqrt(np.mean((K - K_pred)**2))
            NRMSE = RMSE / (max(K)-min(K))
            print('Root Mean Squared Error (RMSE):', round(RMSE,3) , '(m/s)')
            R2_round = str(round(R2,4))
            SDR_df_new_row = pd.DataFrame({"equation":"K = b * PHI**m * T2ML**n", "b":b, "m":m,'n': n, 'R2':R2, 'K_diff_factor': K_diff_factor, 
                                           'MSE': MSE, 'RMSE': RMSE, 'NRMSE': NRMSE, hue:rck},index=[0])
            SDR_df = pd.concat([SDR_df, SDR_df_new_row], ignore_index=True)
            if R2_round == '0':
                R2_round = "{:.3e}".format(R2)
            print('R-squared score:' + R2_round)
            print('Mean K_diff_factor:', round(K_diff_factor,3), '\n')
            annot_text = str('$R^{2}$ = '+ str(R2_round)+'\nMean K$_{diff_factor}$ = '+ str(np.round(K_diff_factor,1)))
            ax.annotate(text=annot_text, xy=(xlim[0]*1.4,ylim[1]*0.5), color='k', fontsize=10)
            # fig.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.suptitle(MODEL, y=1.05)
        #Saving the figure in a file
        fig.set_facecolor("white")
        if output_dirc!='':
            plt.savefig(output_dirc+'\\'+'K_Calibration_Plot(SDR Model)', dpi=600)
        plt.show()
        ###############################################
        ##### Timur-Coates (TC) Model Calibration #####
        """ The model is: K_tc = [(PHI/c)^2 * FFI/BVI]^2  so 0.25 log K = log PHI*(FFI/BVI)**0.5 - log c """
        MODEL = 'TC Model'
        TC_df = pd.DataFrame(columns=['equation','c','R2',hue])
        def K_NMR(X, c):   # Define the function to fit the data
            return 4*np.log10(X/c)  # X=PHI*np.sqrt(FFI/BVI)
        i = -1
        fig, axs = plt.subplots(1,len(rck_all), figsize=(4*len(rck_all),4), sharey=True, sharex=True)
        for rck in rck_all:
            i += 1
            if len(rck_all)==1:
                selected_screen_df = screen_df.copy()
                ax=axs
            else:
                selected_screen_df = screen_df[screen_df[hue]==rck].copy()
                ax = axs[i]
            print(MODEL)
            print('K = [(PHI/c)^2 * FFI/BVI]^2')
            if hue == None:
                print('Curve fitting using all points')
            else:
                print('Curve fitting for: ', rck)
            print('Number of datapoints for curve fitting: ', len(selected_screen_df))
            # Filter out rows where immobile == 0 to avoid division to zero
            removed_rows = selected_screen_df[selected_screen_df.immobile == 0]
            selected_screen_df = selected_screen_df[selected_screen_df.immobile != 0]
            if not removed_rows.empty:
                print("Removed rows:")
                print(removed_rows)
                print('\n')
            X = selected_screen_df.totalf.astype(np.float64) * np.sqrt(selected_screen_df.freef.astype(np.float64) / selected_screen_df.immobile.astype(np.float64))
            K = np.log10(selected_screen_df.K.astype(np.float64))
            popt, pcov = curve_fit(K_NMR, X , K)   # Perform the curve fit
            # Extract the optimized coefficients
            c = popt[0]
            # Print the optimized coefficients
            print('c:', c)
            # plot
            K_pred = K_NMR(X, c)
            # Calculate the K_diff_factor
            K_diff_factor = np.mean(np.maximum((10**K) / (10**K_pred), (10**K_pred) / (10**K)))  # Relative difference
            ax.scatter(10**K, 10**K_pred, color='r' if hue_dict is None else hue_dict[rck] , s=50, alpha=0.7, label=rck)
            ax.set(xscale='log')
            ax.set(yscale='log')
            ax.set_xlabel('K (m/s)', fontsize=13)
            ax.tick_params(axis='x', labelsize=11)
            ax.tick_params(axis='y', labelsize=11)
            ax.set_ylabel('K$_{NMR}$ (m/s)', fontsize=13)
            xlim = [0.5*min(min(10**K),min(10**K_pred)),1.5*max(max(10**K),max(10**K_pred))]
            ylim = xlim
            ax.set(ylim=ylim)
            ax.set(xlim=xlim)
            if len(rck_all)!=1:
                ax.set(title=rck)
            # Y=X and guidelines
            opacity = 0.75
            ax.plot(xlim,xlim, '--k', alpha=0.7, label='K = K$_{NMR}$')
            ax.plot(xlim,[10*xlim[0],10*xlim[1]], linestyle=':', color='gray', alpha=1, label='One OOM higher')
            ax.plot(xlim,[0.1*xlim[0],0.1*xlim[1]], linestyle=':', color='gray', alpha=1, label='One OOM lower')        
            # r2 score
            # from sklearn.metrics import r2_score
            # R2 = r2_score(K, K_pred)
            # print('R-squared score:', R2)
            TSS = np.sum((K - np.mean(K))**2)
            RSS = np.sum((K - K_pred)**2)
            R2 = 1 - RSS / TSS
            # Mean Squared Error
            MSE = np.mean((K - K_pred)**2)
            print('Mean Squared Error (MSE):', round(MSE,3) , '(m/s)^2')
            # Root Mean Squared Error
            RMSE = np.sqrt(np.mean((K - K_pred)**2))
            NRMSE = RMSE / (max(K)-min(K))
            print('Root Mean Squared Error (RMSE):', round(RMSE,3) , '(m/s)')
            TC_df_new_row = pd.DataFrame({"equation":"K = [(PHI/c)^2 * FFI/BVI]^2", "c":c, 'R2':R2, 'K_diff_factor': K_diff_factor, 
                                           'MSE': MSE, 'RMSE': RMSE, 'NRMSE': NRMSE, hue:rck},index=[0])
            TC_df = pd.concat([TC_df, TC_df_new_row], ignore_index=True)
            R2_round = str(round(R2,4))
            if R2_round == '0':
                R2_round = "{:.3e}".format(R2)
            print('R-squared score:' + R2_round)
            print('Mean K_diff_factor:', round(K_diff_factor,3), '\n')
            annot_text = str('$R^{2}$ = '+ str(R2_round)+'\nK$_{difference-factor}$ = '+ str(np.round(K_diff_factor,1)))
            ax.annotate(text=annot_text, xy=(xlim[0]*1.4,ylim[1]*0.3), color='k', fontsize=9)
        plt.tight_layout()
        plt.suptitle(MODEL, y=1.05)
        #Saving the figure in a file
        fig.set_facecolor("white")
        if output_dirc!='':
            plt.savefig(output_dirc+'\\'+'K_Calibration_Plot(TC Model)', dpi=600)
        plt.show()
        ######################################################
        #### Summation of Echoes (SOE) Model Calibration #####
        """ The model is: K_soe = c * SOE**d """
        MODEL = 'SOE Model'
        SOE_df = pd.DataFrame(columns=['equation','c','d','R2',hue])
        def K_NMR(SOE, c, d):   # Define the function to fit the data
            return d*np.log10(SOE) + np.log10(c)
        i = -1
        fig, axs = plt.subplots(1, len(rck_all), figsize=(4*len(rck_all),4), sharey=True, sharex=True)
        for rck in rck_all:
            i += 1
            if len(rck_all)==1:
                selected_screen_df = screen_df.copy()
                ax=axs
            else:
                selected_screen_df = screen_df[screen_df[hue]==rck].copy()
                ax = axs[i]
            print(MODEL)
            print('K = c * SOE**d')
            if hue == None:
                print('Curve fitting using all points')
            else:
                print('Curve fitting for: ', rck)
            print('Number of datapoints for curve fitting: ', len(selected_screen_df))
            SOE = selected_screen_df.soe.astype(np.float64)
            K = np.log10(selected_screen_df.K.astype(np.float64))
            param_bounds = ([0, 0], [np.inf, np.inf])
            popt, pcov = curve_fit(K_NMR, SOE, K, p0=[0,0], bounds=param_bounds)
            # Extract the optimized coefficients
            c = popt[0]
            d = popt[1]
            # Print the optimized coefficients
            print('c:', c)
            print('d:', d)
            # plot
            K_pred = K_NMR(SOE, c, d)
            # Calculate the K_diff_factor
            K_diff_factor = np.mean(np.maximum((10**K) / (10**K_pred), (10**K_pred) / (10**K)))  # Relative difference
            ax.scatter(10**K, 10**K_pred, color='c' if hue_dict is None else hue_dict[rck] , s=50, alpha=0.7, label=rck)
            ax.set(xscale='log')
            ax.set(yscale='log')
            ax.set_xlabel('K (m/s)', fontsize=13)
            ax.tick_params(axis='x', labelsize=11)
            ax.tick_params(axis='y', labelsize=11)
            ax.set_ylabel('K$_{NMR}$ (m/s)', fontsize=13)
            xlim = [0.5*min(min(10**K),min(10**K_pred)),1.5*max(max(10**K),max(10**K_pred))]
            ylim = xlim
            ax.set(ylim=ylim)
            ax.set(xlim=xlim)
            if len(rck_all)!=1:
                ax.set(title=rck)
            # Y=X and guidelines
            opacity = 0.75
            ax.plot(xlim,xlim, '--k', alpha=0.7, label='K = K$_{NMR}$')
            ax.plot(xlim,[10*xlim[0],10*xlim[1]], linestyle=':', color='gray', alpha=1, label='One OOM higher')
            ax.plot(xlim,[0.1*xlim[0],0.1*xlim[1]], linestyle=':', color='gray', alpha=1, label='One OOM lower')        
            # r2 score
            # from sklearn.metrics import r2_score
            # R2 = r2_score(K, K_pred)
            # print('R-squared score:', R2)
            TSS = np.sum((K - np.mean(K))**2)
            RSS = np.sum((K - K_pred)**2)
            R2 = 1 - RSS / TSS
            # Mean Squared Error
            MSE = np.mean((K - K_pred)**2)
            print('Mean Squared Error (MSE):', round(MSE,3) , '(m/s)^2')
            # Root Mean Squared Error
            RMSE = np.sqrt(np.mean((K - K_pred)**2))
            print('Root Mean Squared Error (RMSE):', round(RMSE,3) , '(m/s)')
            SOE_df_new_row = pd.DataFrame({"equation":"K = c * SOE**d", "c":c, "d":d,'R2':R2, 'K_diff_factor': K_diff_factor, 
                                           'MSE': MSE, 'RMSE': RMSE, 'NRMSE': NRMSE, hue:rck},index=[0])
            SOE_df = pd.concat([SOE_df, SOE_df_new_row], ignore_index=True)
            R2_round = str(round(R2,4))
            if R2_round == '0':
                R2_round = "{:.3e}".format(R2)
            print('R-squared score:' + R2_round)
            print('Mean K_diff_factor:', round(K_diff_factor,3), '\n')
            annot_text = str('$R^{2}$ = '+ str(R2_round)+'\nK$_{difference-factor}$ = '+ str(np.round(K_diff_factor,1)))
            ax.annotate(text=annot_text, xy=(xlim[0]*1.4,ylim[1]*0.3), color='k', fontsize=9)
        plt.tight_layout()
        plt.suptitle(MODEL, y=1.05)
        #Saving the figure in a file
        fig.set_facecolor("white")
        if output_dirc!='':
            plt.savefig(output_dirc+'\\'+'K_Calibration_Plot(SOE Model).png', dpi=600)
            with pd.ExcelWriter(output_dirc+'\\'+'K_Calibration_Results.xlsx') as writer:  # exporting screen_df and calibrated coefficient factors for various methods
                screen_df.to_excel(writer, sheet_name='BNMR_Screen_df', index=False)
                SDR_df.to_excel(writer, sheet_name='SDR_df', index=False)
                TC_df.to_excel(writer, sheet_name='TC_df', index=False)
                SOE_df.to_excel(writer, sheet_name='SOE_df', index=False)
        plt.show()

        if add_K_to_BNMR_data==True:
            methods = [None,None,None]
            if 'SDR' in best_methods:
                methods[0] = SDR_df
            if 'TC' in best_methods:
                methods[1] = TC_df
            if 'SOE' in best_methods:
                methods[2] = SOE_df
            max_noise_BNMR = data1.noise.max()    
            df_final = self.k_calculation_in_BNMR_dataframe (SDR_df=methods[0], TC_df=methods[1], SOE_df=methods[2], max_noise_BNMR=max_noise_BNMR, output_dirc=output_dirc)
        else: 
            df_final=[]
        return SDR_df, TC_df, SOE_df, df_final, screen_df

    def calibrate_K_estimation_models_2 (self, hue=None, hue_dict=None, add_K_to_BNMR_data=False , hue_2=None, hue2_dict=None, markersize=60, alpha=.7, output_dirc=''):
        """ Function for calibarting K estimation models that provides calibrated coefficient constants in separate dataframes for SDR, SOE, and TC models:
        - K_data_file: screen intervals (Borehole, TOP, BOTTOM, K, and if available ROCKSYMBOL and K_std). Note that TOP and BOTTOM values are in meters and K in m/s.
        - method= 'all', 'SDR', 'TC', 'SOE'. the default is all.
        - hue is a column in the dataframe which is meant to define classes to do the calibration for them separately. 
        - Hue2 is a column in the dataframe that is only for visualizing points with different columns in each plot.
        For instance, we can use hue for calibrating for each geology, and then use hue2 to color points in the plots separately for each borehole"""
        if hue == None:
            hue2 = 'ROCKSYMBOL'
        else:
            hue2 = hue
        screen_df = self.geolabled_database.copy(deep=True)
        # Total Porosity & T2ML plots versus K measurements (points after averaging for the screen interval)
        fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharex=True)
        axs = ax[0]  # subplot 1
        sns.scatterplot(data=screen_df, x='K', y='totalf', hue=hue, palette=hue_dict if hue_dict is not None else None, s=markersize, alpha=alpha, 
                        edgecolor='white', linewidth=1, ax=axs, legend=True)
        axs.set(xscale='log')
        axs.set(yscale='linear')
        # axs.set(ylim=(0,100))
        axs.set_xlabel('K (m/s)', fontsize=13)
        axs.set_ylabel('Total Porosity (%)', fontsize=13)
        axs.minorticks_on()
        axs.tick_params(axis='x', labelsize=11)
        axs.tick_params(axis='y', labelsize=11)
        # axs.grid(which='major', color='grey', linewidth=0.5)
        axs = ax[1] # subplot 2
        sns.scatterplot(data=screen_df, x='K', y='mlT2', hue=hue, palette=hue_dict if hue_dict is not None else None, s=markersize, alpha=alpha, 
                        edgecolor='white', linewidth=1, ax=axs, legend=True)
        axs.set(xscale='log')
        axs.set(yscale='log')
        axs.set_xlabel('K (m/s)', fontsize=13)
        axs.set_ylabel('$T_{2ML}$ (s)', fontsize=13)
        # axs.set(ylim=(1e-3,1))
        axs.minorticks_on()
        axs.tick_params(axis='x', labelsize=11)
        axs.tick_params(axis='y', labelsize=11)
        if hue != None:
            ax[0].legend().remove() # hide the legend of the first subplot
            ax[1].legend().remove()
            # create a legend for the second subplot outside the plot
            handles, labels = axs.get_legend_handles_labels()
            plt.legend(title='LEGEND', handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1, 0.5))
        # axs.grid(which='major', color='grey', linewidth=0.5)
        plt.tight_layout()
        #Saving the figure in a file
        fig.set_facecolor("white")
        if output_dirc!='':
            plt.savefig(output_dirc+'\\'+'K_Calibration_Plot(K_vs_NMR_parameters)'+str(hue)+'.jpg', dpi=600)
        plt.show()
        #################################
        ##### SDR Model Calibration #####
        """ The model is: K_sdr = b * PHI**m * T2ML**n """
        MODEL = 'SDR Model'
        SDR_df = pd.DataFrame(columns=['equation','b','m','n','R2',hue])
        i = -1
        if hue == None:
            rck_all = ['None']
        else:
            rck_all = screen_df[hue].unique()
        def K_NMR(x, b, m, n): # Define the function to fit the data
                PHI, T2ML = x
                return np.log10(b) + (PHI*m) + (T2ML* n)
        fig, axs = plt.subplots(1, len(rck_all), figsize=(5*len(rck_all),5), sharey=False, sharex=False)
        for rck in rck_all: 
            i += 1
            if len(rck_all)==1:
                selected_screen_df = screen_df.copy()
                ax=axs
            else:
                selected_screen_df = screen_df[screen_df[hue]==rck].copy()
                ax = axs[i]
            print(MODEL)
            print('K = b * PHI**m * T2ML**n')
            if hue == None:
                print('Curve fitting using all points')
            else:
                print('Curve fitting for: ', rck)
            print('Number of datapoints for curve fitting: ', len(selected_screen_df))
            PHI = np.log10(selected_screen_df.totalf.astype(np.float64)/100)
            T2ML = np.log10(selected_screen_df.mlT2.astype(np.float64))
            K = np.log10(selected_screen_df.K.astype(np.float64))
            param_bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
            popt, pcov = curve_fit(K_NMR, (PHI, T2ML), K, p0=[0, 0, 0], bounds=param_bounds) # Perform the curve fit
            # Extract the optimized coefficients
            b = popt[0]
            m = popt[1]
            n = popt[2]
            print('Estimated coefficients')
            print('b:', b)
            print('m:', m)
            print('n:', n)
            # plot
            K_pred = K_NMR((PHI, T2ML), b, m, n)
            # Calculate the K_diff_factor
            K_diff_factor = np.mean(np.maximum((10**K) / (10**K_pred), (10**K_pred) / (10**K)))  # Relative difference
            if hue_2:
                hue2_classes = selected_screen_df[hue_2].unique()
                if hue2_dict !=None:
                    color_map = hue2_dict
                else:
                    colors = plt.cm.tab10(range(len(hue2_classes)))
                    color_map = dict(zip(hue2_classes, colors))
                
                scatter_color = selected_screen_df[hue_2].map(color_map)
                handles = [mpatches.Patch(color=color_map[hue2_class], label=hue2_class) for hue2_class in hue2_classes]
                ax.legend(handles=handles, title=hue_2, fontsize=7, loc='lower right')
            else:
                scatter_color = 'b' if hue_dict is None else hue_dict[rck]
            ax.scatter(10**K, 10**K_pred, c=scatter_color, s=markersize, alpha=alpha, edgecolor='white', linewidth=1, label=rck)
            ax.set(xscale='log')
            ax.set(yscale='log')
            ax.set_xlabel('K (m/s)', fontsize=13)
            ax.tick_params(axis='x', labelsize=11)
            ax.tick_params(axis='y', labelsize=11)
            ax.set_ylabel('K$_{NMR}$ (m/s)', fontsize=13)
            xlim = [0.5*min(min(10**K),min(10**K_pred)),1.5*max(max(10**K),max(10**K_pred))]
            ylim = xlim
            ax.set(ylim=ylim)
            ax.set(xlim=xlim)
            # Y=X and guidelines
            opacity = 0.75
            ax.plot(xlim,xlim, '--k', alpha=0.7, label='K = K$_{NMR}$')
            ax.plot(xlim,[10*xlim[0],10*xlim[1]], linestyle=':', color='gray', alpha=1, label='One OOM higher')
            ax.plot(xlim,[0.1*xlim[0],0.1*xlim[1]], linestyle=':', color='gray', alpha=1, label='One OOM lower')
            # r2 score
            # from sklearn.metrics import r2_score
            # R2 = r2_score(K, K_pred)
            # print('R-squared score:', R2)
            TSS = np.sum((K - np.mean(K))**2)
            RSS = np.sum((K - K_pred)**2)
            R2 = 1 - RSS / TSS
            
            # Mean Squared Error
            MSE = np.mean((K - K_pred)**2)
            print('Mean Squared Error (MSE):', round(MSE,3) , '(m/s)^2')
            # Root Mean Squared Error
            RMSE = np.sqrt(np.mean((K - K_pred)**2))
            NRMSE = RMSE / (max(K)-min(K))
            print('Root Mean Squared Error (RMSE):', round(RMSE,3) , '(m/s)')
            R2_round = str(round(R2,4))
            SDR_df_new_row = pd.DataFrame({"equation":"K = b * PHI**m * T2ML**n", "b":b, "m":m,'n': n, 'R2':R2, 'K_diff_factor': K_diff_factor, 
                                           'MSE': MSE, 'RMSE': RMSE, 'NRMSE': NRMSE, hue:rck},index=[0])
            SDR_df = pd.concat([SDR_df, SDR_df_new_row], ignore_index=True)
            if R2_round == '0':
                R2_round = "{:.3e}".format(R2)
            print('R-squared score:' + R2_round)
            print('Mean K_diff_factor:', round(K_diff_factor,3), '\n')
            annot_text = str('$R^{2}$ = '+ str(R2_round)+'\nK$_{difference-factor}$ = '+ str(np.round(K_diff_factor,1)))
            ax.annotate(text=annot_text, xy=(xlim[0]*1.4,ylim[1]*0.3), color='k', fontsize=9)
            if len(rck_all)!=1:
                ax.set(title=rck)
            # fig.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.suptitle(MODEL, y=1.05)
        #Saving the figure in a file
        fig.set_facecolor("white")
        if output_dirc!='':
            plt.savefig(output_dirc+'\\'+'K_Calibration_Plot(SDR Model)'+str(hue)+'.jpg', dpi=600)
        plt.show()
        
        #################################
        ##### SDR Model (without porosity) Calibration #####
        """ The model is: K_sdr = b * T2ML**n """
        MODEL = 'SDR Model without porosity'
        SDR_df_m0 = pd.DataFrame(columns=['equation','b','n','R2',hue])
        i = -1
        if hue == None:
            rck_all = ['None']
        else:
            rck_all = screen_df[hue].unique()
        def K_NMR(x, b, n): # Define the function to fit the data
            T2ML = x
            return np.log10(b) + (T2ML* n)
        fig, axs = plt.subplots(1, len(rck_all), figsize=(5*len(rck_all),5), sharey=False, sharex=False)
        for rck in rck_all:
            i += 1
            if len(rck_all)==1:
                selected_screen_df = screen_df.copy()
                ax=axs
            else:
                selected_screen_df = screen_df[screen_df[hue]==rck].copy()
                ax = axs[i]
            print(MODEL)
            print('K = b * T2ML**n')
            if hue == None:
                print('Curve fitting using all points')
            else:
                print('Curve fitting for: ', rck)
            print('Number of datapoints for curve fitting: ', len(selected_screen_df))
            PHI = np.log10(selected_screen_df.totalf.astype(np.float64)/100)
            T2ML = np.log10(selected_screen_df.mlT2.astype(np.float64))
            K = np.log10(selected_screen_df.K.astype(np.float64))
            param_bounds = ([0,  0], [np.inf,  np.inf])
            popt, pcov = curve_fit(K_NMR, T2ML, K, p0=[0, 0], bounds=param_bounds) # Perform the curve fit
            # Extract the optimized coefficients
            b = popt[0]
            n = popt[1]

            print('Estimated coefficients')
            print('b:', b)
            print('n:', n)
            # plot
            K_pred = K_NMR(T2ML, b, n)
            # Calculate the K_diff_factor
            K_diff_factor = np.mean(np.maximum((10**K) / (10**K_pred), (10**K_pred) / (10**K)))  # Relative difference
            if hue_2:
                hue2_classes = selected_screen_df[hue_2].unique()
                if hue2_dict !=None:
                    color_map = hue2_dict
                else:
                    colors = plt.cm.tab10(range(len(hue2_classes)))
                    color_map = dict(zip(hue2_classes, colors))
                
                scatter_color = selected_screen_df[hue_2].map(color_map)
                handles = [mpatches.Patch(color=color_map[hue2_class], label=hue2_class) for hue2_class in hue2_classes]
                ax.legend(handles=handles, title=hue_2, fontsize=7, loc='lower right')
            else:
                scatter_color = 'b' if hue_dict is None else hue_dict[rck]
            ax.scatter(10**K, 10**K_pred, c=scatter_color, s=markersize, alpha=alpha, edgecolor='white', linewidth=1, label=rck)
            ax.set(xscale='log')
            ax.set(yscale='log')
            ax.set_xlabel('K (m/s)', fontsize=13)
            ax.tick_params(axis='x', labelsize=11)
            ax.tick_params(axis='y', labelsize=11)
            ax.set_ylabel('K$_{NMR}$ (m/s)', fontsize=13)
            xlim = [0.5*min(min(10**K),min(10**K_pred)),1.5*max(max(10**K),max(10**K_pred))]
            ylim = xlim
            ax.set(ylim=ylim)
            ax.set(xlim=xlim)
            # Y=X and guidelines
            opacity = 0.75
            ax.plot(xlim,xlim, '--k', alpha=0.7, label='K = K$_{NMR}$')
            ax.plot(xlim,[10*xlim[0],10*xlim[1]], linestyle=':', color='gray', alpha=1, label='One OOM higher')
            ax.plot(xlim,[0.1*xlim[0],0.1*xlim[1]], linestyle=':', color='gray', alpha=1, label='One OOM lower')
            # r2 score
            # from sklearn.metrics import r2_score
            # R2 = r2_score(K, K_pred)
            # print('R-squared score:', R2)
            TSS = np.sum((K - np.mean(K))**2)
            RSS = np.sum((K - K_pred)**2)
            R2 = 1 - RSS / TSS
            
            # Mean Squared Error
            MSE = np.mean((K - K_pred)**2)
            print('Mean Squared Error (MSE):', round(MSE,3) , '(m/s)^2')
            # Root Mean Squared Error
            RMSE = np.sqrt(np.mean((K - K_pred)**2))
            NRMSE = RMSE / (max(K)-min(K))
            print('Root Mean Squared Error (RMSE):', round(RMSE,3) , '(m/s)')
            R2_round = str(round(R2,4))
            SDR_df_m0_new_row = pd.DataFrame({"equation": "K = b * T2ML**n", "b": b, "n": n,'R2': R2,'K_diff_factor': 
                                              K_diff_factor,'MSE': MSE, 'RMSE': RMSE,'NRMSE': NRMSE, hue: rck}, index=[0])
            SDR_df_m0 = pd.concat([SDR_df_m0, SDR_df_m0_new_row], ignore_index=True)
            if R2_round == '0':
                R2_round = "{:.3e}".format(R2)
            print('R-squared score:' + R2_round)
            print('Mean K_diff_factor:', round(K_diff_factor,3), '\n')
            annot_text = str('$R^{2}$ = '+ str(R2_round)+'\nK$_{difference-factor}$ = '+ str(np.round(K_diff_factor,1)))
            ax.annotate(text=annot_text, xy=(xlim[0]*1.4,ylim[1]*0.3), color='k', fontsize=9)
            if len(rck_all)!=1:
                ax.set(title=rck)
            # fig.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.suptitle(MODEL, y=1.05)
        #Saving the figure in a file
        fig.set_facecolor("white")
        if output_dirc!='':
            plt.savefig(output_dirc+'\\'+'K_Calibration_Plot(SDR Model Revised)'+str(hue)+'.jpg', dpi=600)
        plt.show()
        
        ###############################################
        ##### Timur-Coates (TC) Model Calibration #####
        """ The model is: K_tc = [(PHI/c)^2 * FFI/BVI]^2  so 0.25 log K = log PHI*(FFI/BVI)**0.5 - log c """
        MODEL = 'TC Model'
        TC_df = pd.DataFrame(columns=['equation','c','R2',hue])
        def K_NMR(X, c):   # Define the function to fit the data
            return 4*np.log10(X/c)  # X=PHI*np.sqrt(FFI/BVI)
        i = -1
        fig, axs = plt.subplots(1,len(rck_all), figsize=(5*len(rck_all),5), sharey=False, sharex=False)
        for rck in rck_all:
            i += 1
            if len(rck_all)==1:
                selected_screen_df = screen_df.copy()
                ax=axs
            else:
                selected_screen_df = screen_df[screen_df[hue]==rck].copy()
                ax = axs[i]
            print(MODEL)
            print('K = [(PHI/c)^2 * FFI/BVI]^2')
            if hue == None:
                print('Curve fitting using all points')
            else:
                print('Curve fitting for: ', rck)
            
            # Filter out rows where immobile == 0 to avoid division to zero
            removed_rows = selected_screen_df[selected_screen_df.immobile == 0]
            selected_screen_df = selected_screen_df[selected_screen_df.immobile != 0]
            if not removed_rows.empty:
                print("Removed rows:")
                print(removed_rows)
                print('\n')
                        
            print('Number of datapoints for curve fitting: ', len(selected_screen_df))
            X = selected_screen_df.totalf.astype(np.float64) * np.sqrt(selected_screen_df.freef.astype(np.float64) / selected_screen_df.immobile.astype(np.float64))
            K = np.log10(selected_screen_df.K.astype(np.float64))
            popt, pcov = curve_fit(K_NMR, X , K)   # Perform the curve fit
            # Extract the optimized coefficients
            c = popt[0]
            # Print the optimized coefficients
            print('c:', c)
            # plot
            K_pred = K_NMR(X, c)
            # Calculate the K_diff_factor
            K_diff_factor = np.mean(np.maximum((10**K) / (10**K_pred), (10**K_pred) / (10**K)))  # Relative difference
            if hue_2:
                hue2_classes = selected_screen_df[hue_2].unique()
                if hue2_dict !=None:
                    color_map = hue2_dict
                else:
                    colors = plt.cm.tab10(range(len(hue2_classes)))
                    color_map = dict(zip(hue2_classes, colors))
                
                scatter_color = selected_screen_df[hue_2].map(color_map)
                handles = [mpatches.Patch(color=color_map[hue2_class], label=hue2_class) for hue2_class in hue2_classes]
                ax.legend(handles=handles, title=hue_2, fontsize=7, loc='lower right')
            else:
                scatter_color = 'b' if hue_dict is None else hue_dict[rck]
            ax.scatter(10**K, 10**K_pred, c=scatter_color, s=markersize, alpha=alpha, edgecolor='white', linewidth=1, label=rck)
            ax.set(xscale='log')
            ax.set(yscale='log')
            ax.set_xlabel('K (m/s)', fontsize=13)
            ax.tick_params(axis='x', labelsize=11)
            ax.tick_params(axis='y', labelsize=11)
            ax.set_ylabel('K$_{NMR}$ (m/s)', fontsize=13)
            xlim = [0.5*min(min(10**K),min(10**K_pred)),1.5*max(max(10**K),max(10**K_pred))]
            ylim = xlim
            ax.set(ylim=ylim)
            ax.set(xlim=xlim)
            # Y=X and guidelines
            opacity = 0.75
            ax.plot(xlim,xlim, '--k', alpha=0.7, label='K = K$_{NMR}$')
            ax.plot(xlim,[10*xlim[0],10*xlim[1]], linestyle=':', color='gray', alpha=1, label='One OOM higher')
            ax.plot(xlim,[0.1*xlim[0],0.1*xlim[1]], linestyle=':', color='gray', alpha=1, label='One OOM lower')        
            # r2 score
            # from sklearn.metrics import r2_score
            # R2 = r2_score(K, K_pred)
            # print('R-squared score:', R2)
            TSS = np.sum((K - np.mean(K))**2)
            RSS = np.sum((K - K_pred)**2)
            R2 = 1 - RSS / TSS
            # Mean Squared Error
            MSE = np.mean((K - K_pred)**2)
            print('Mean Squared Error (MSE):', round(MSE,3) , '(m/s)^2')
            # Root Mean Squared Error
            RMSE = np.sqrt(np.mean((K - K_pred)**2))
            NRMSE = RMSE / (max(K)-min(K))
            print('Root Mean Squared Error (RMSE):', round(RMSE,3) , '(m/s)')
            TC_df_new_row = pd.DataFrame({"equation":"K = [(PHI/c)^2 * FFI/BVI]^2", "c":c, 'R2':R2, 'K_diff_factor': K_diff_factor, 
                                           'MSE': MSE, 'RMSE': RMSE, 'NRMSE': NRMSE, hue:rck},index=[0])
            TC_df = pd.concat([TC_df, TC_df_new_row], ignore_index=True)
            R2_round = str(round(R2,4))
            if R2_round == '0':
                R2_round = "{:.3e}".format(R2)
            print('R-squared score:' + R2_round)
            print('Mean K_diff_factor:', round(K_diff_factor,3), '\n')
            annot_text = str('$R^{2}$ = '+ str(R2_round)+'\nK$_{difference-factor}$ = '+ str(np.round(K_diff_factor,1)))
            ax.annotate(text=annot_text, xy=(xlim[0]*1.4,ylim[1]*0.3), color='k', fontsize=9)
            if len(rck_all)!=1:
                ax.set(title=rck)
        plt.tight_layout()
        plt.suptitle(MODEL, y=1.05)
        #Saving the figure in a file
        fig.set_facecolor("white")
        if output_dirc!='':
            plt.savefig(output_dirc+'\\'+'K_Calibration_Plot(TC Model)'+str(hue)+'.jpg', dpi=600)
        plt.show()
        ######################################################
        #### Summation of Echoes (SOE) Model Calibration #####
        """ The model is: K_soe = c * SOE**d """
        MODEL = 'SOE Model'
        SOE_df = pd.DataFrame(columns=['equation','c','d','R2',hue])
        def K_NMR(SOE, c, d):   # Define the function to fit the data
            return d*np.log10(SOE) + np.log10(c)
        i = -1
        fig, axs = plt.subplots(1, len(rck_all), figsize=(5*len(rck_all),5), sharey=False, sharex=False)
        for rck in rck_all:
            i += 1
            if len(rck_all)==1:
                selected_screen_df = screen_df.copy()
                ax=axs
            else:
                selected_screen_df = screen_df[screen_df[hue]==rck].copy()
                ax = axs[i]
            print(MODEL)
            print('K = c * SOE**d')
            if hue == None:
                print('Curve fitting using all points')
            else:
                print('Curve fitting for: ', rck)
            print('Number of datapoints for curve fitting: ', len(selected_screen_df))
            SOE = selected_screen_df.soe.astype(np.float64)
            K = np.log10(selected_screen_df.K.astype(np.float64))
            param_bounds = ([0, 0], [np.inf, np.inf])
            popt, pcov = curve_fit(K_NMR, SOE, K, p0=[0,0], bounds=param_bounds)
            # Extract the optimized coefficients
            c = popt[0]
            d = popt[1]
            # Print the optimized coefficients
            print('c:', c)
            print('d:', d)
            # plot
            K_pred = K_NMR(SOE, c, d)
            # Calculate the K_diff_factor
            K_diff_factor = np.mean(np.maximum((10**K) / (10**K_pred), (10**K_pred) / (10**K)))  # Relative difference
            if hue_2:
                hue2_classes = selected_screen_df[hue_2].unique()
                if hue2_dict !=None:
                    color_map = hue2_dict
                else:
                    colors = plt.cm.tab10(range(len(hue2_classes)))
                    color_map = dict(zip(hue2_classes, colors))
                
                scatter_color = selected_screen_df[hue_2].map(color_map)
                handles = [mpatches.Patch(color=color_map[hue2_class], label=hue2_class) for hue2_class in hue2_classes]
                ax.legend(handles=handles, title=hue_2, fontsize=7, loc='lower right')
            else:
                scatter_color = 'b' if hue_dict is None else hue_dict[rck]
            ax.scatter(10**K, 10**K_pred, c=scatter_color, s=markersize, alpha=alpha, edgecolor='white', linewidth=1, label=rck)
            ax.set(xscale='log')
            ax.set(yscale='log')
            ax.set_xlabel('K (m/s)', fontsize=13)
            ax.tick_params(axis='x', labelsize=11)
            ax.tick_params(axis='y', labelsize=11)
            ax.set_ylabel('K$_{NMR}$ (m/s)', fontsize=13)
            xlim = [0.5*min(min(10**K),min(10**K_pred)),1.5*max(max(10**K),max(10**K_pred))]
            ylim = xlim
            ax.set(ylim=ylim)
            ax.set(xlim=xlim)
            # Y=X and guidelines
            opacity = 0.75
            ax.plot(xlim,xlim, '--k', alpha=0.7, label='K = K$_{NMR}$')
            ax.plot(xlim,[10*xlim[0],10*xlim[1]], linestyle=':', color='gray', alpha=1, label='One OOM higher')
            ax.plot(xlim,[0.1*xlim[0],0.1*xlim[1]], linestyle=':', color='gray', alpha=1, label='One OOM lower')        
            # r2 score
            # from sklearn.metrics import r2_score
            # R2 = r2_score(K, K_pred)
            # print('R-squared score:', R2)
            TSS = np.sum((K - np.mean(K))**2)
            RSS = np.sum((K - K_pred)**2)
            R2 = 1 - RSS / TSS
            # Mean Squared Error
            MSE = np.mean((K - K_pred)**2)
            print('Mean Squared Error (MSE):', round(MSE,3) , '(m/s)^2')
            # Root Mean Squared Error
            RMSE = np.sqrt(np.mean((K - K_pred)**2))
            print('Root Mean Squared Error (RMSE):', round(RMSE,3) , '(m/s)')
            SOE_df_new_row = pd.DataFrame({"equation":"K = c * SOE**d", "c":c, "d":d,'R2':R2, hue:rck},index=[0])
            SOE_df = pd.concat([SOE_df, SOE_df_new_row], ignore_index=True)
            R2_round = str(round(R2,4))
            if R2_round == '0':
                R2_round = "{:.3e}".format(R2)
            print('R-squared score:' + R2_round)
            print('Mean K_diff_factor:', round(K_diff_factor,3), '\n')
            annot_text = str('$R^{2}$ = '+ str(R2_round)+'\nK$_{difference-factor}$ = '+ str(np.round(K_diff_factor,1)))
            ax.annotate(text=annot_text, xy=(xlim[0]*1.4,ylim[1]*0.3), color='k', fontsize=9)
            if len(rck_all)!=1:
                ax.set(title=rck)
        plt.tight_layout()
        plt.suptitle(MODEL, y=1.05)
        #Saving the figure in a file
        fig.set_facecolor("white")
        if output_dirc!='':
            plt.savefig(output_dirc+'\\'+'K_Calibration_Plot(SOE Model)'+str(hue)+'.jpg', dpi=600)
            with pd.ExcelWriter(output_dirc+'\\'+'K_Calibration_Results.xlsx') as writer:  # exporting screen_df and calibrated coefficient factors for various methods
                screen_df.to_excel(writer, sheet_name='BNMR_Screen_df', index=False)
                SDR_df.to_excel(writer, sheet_name='SDR_df', index=False)
                SDR_df_m0.to_excel(writer, sheet_name='SDR_df_m0', index=False)
                TC_df.to_excel(writer, sheet_name='TC_df', index=False)
                SOE_df.to_excel(writer, sheet_name='SOE_df', index=False)
                
        plt.show()
        final_df = self.k_calculation_in_BNMR_dataframe(SDR_df=SDR_df, TC_df=TC_df, SOE_df=SOE_df, max_noise_BNMR=100, output_dirc='')
        return SDR_df, SDR_df_m0, TC_df, SOE_df, final_df
    
    def k_calculation_in_BNMR_dataframe (self, SDR_df=None, TC_df=None, SOE_df=None, max_noise_BNMR=20, output_dirc=''):
        """This is a function for calculating K values based on calibrated coefficient constants for SDR; SOE, and TC methods. Based on calibration results, we 
        can decide which method is better to use. It is also possible to get K based on coefficient factors from another resource or paper"""
        df = self.geolabled_database.copy(deep=True)
        columns_to_drop = ['unix_time','board_temp','magnet_temp','Ksdr','Ktc','Ksoe','Tsoe','Tsdr','Ttc']
        df.drop([col for col in columns_to_drop if col in df.columns], axis=1, inplace=True)
        if SDR_df is not None:
            df.loc[:,'Ksdr'] = SDR_df.loc[0,'b'] * ((df.loc[:,'totalf']/100)**SDR_df.loc[0,'m']) * (df.loc[:,'mlT2']**SDR_df.loc[0,'n'])
            df.loc[df.saturated=='no','Ksdr'] = np.nan
            df.loc[df.noise>max_noise_BNMR] = np.nan
        if TC_df is not None:
            df.loc[:,'immobile'] = df.loc[:,'capf']+df.loc[:,'clayf']
            df.loc[:,'Ktc'] = ((df.loc[:,'freef']/(df.loc[:,'immobile']))*(((df.loc[:,'totalf']/100)/TC_df.loc[0,'c'])**2))**2
            df.loc[df.saturated=='no','Ktc'] = np.nan
            df.loc[df.noise>max_noise_BNMR] = np.nan
        if SOE_df is not None:
            df.loc[:,'Ksoe'] = SOE_df.loc[0,'c'] * (df.loc[:,'soe']**SOE_df.loc[0,'d'])
            df.loc[df.saturated=='no','Ksoe'] = np.nan
            df.loc[df.noise>max_noise_BNMR] = np.nan
        if SDR_df is None and TC_df is None and SOE_df is None:
            print('None of the models are selected for K calculation.')
        if output_dirc !='':
            df.to_csv(output_dirc+'\\BNMR_Dataset_with_K.txt', sep='\t', index=False)
        return df





