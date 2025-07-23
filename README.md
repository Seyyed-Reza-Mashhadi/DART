# dart (Data Analysis and Representation Toolkit) for Borehole NMR Data
Comprehensive Borehole NMR data analysis tool for Vista Clara BNMR systems

## 🔍 Overview
DART is a powerful Python toolkit designed for end-to-end processing, visualization, and interpretation of borehole nuclear magnetic resonance (BNMR) data from Vista Clara systems. It integrates geospatial analysis, statistical modeling, and geological interpretation into a unified workflow for comprehensive subsurface characterization.

## 🌟 Key Features & Functionality

- Geospatial Analysis: Coordinate handling, profile azimuth calculation, KML/shapefile exports
- Statistical Modeling: Hydraulic conductivity (K) calibration, statistical distribution analysis
- Geological Integration: Lithology labeling, ternary plots of water fractions
- Automated Reporting: LAS file exports, Excel statistical reports, publication-ready figures

## 🧩 Architecture
### Data Class
- Handles borehole-based data management.
- Input: Receives borehole names (or identifiers) to load and manage related data.
### Statistics Class
- Performs dataset-based statistical modeling.
- Input: Takes a dataframe (e.g., pandas DataFrame) representing the dataset for analysis.
### Other Functions / Tools
- Utility functions and tools implemented outside of the classes to support various tasks.

## 🛠️ Simple Data Preprocessing & Integration
- Function Name: get_borehole_names
  Scans directories for all borehole data files in a survey/project
- Function Name: add_coordinates
  Merges XYZ coordinates with BNMR measurements
- Function Name: check_NonNumeric_columns
  Cleans invalid entries in geology/hydrology data
- Function Name: merge_txt_files
  Combines project datasets for unified analysis

### Geo-labeling & BNMR Data Integration of Individual Boreholes 
- Geo-labeling the BNMR data (if geology information is available)
- Merging all boreholes data into a unified dataframe
  
## 📈 Advanced Visualization Engine
Borehole Logs (logplot())
python
### Generate Professional Borehole Logs 
- Class Name: Data
- Function Name: logplot
- This feature combines:
  - Spin-echo decay heatmaps
  - Total porosity & water partitioning (clay/capillary/free porosity)
  - T₂ distribution curves
  - Geology log (if available)
  - Hydraulic conductivity (if available)

### Geological Sections (section_plot_all())
- Class Name: -
- Function Name: section_plot_all
- This feature provides:
  - Interactive profile selection (Tkinter GUI)
  - Topography-aware elevation scaling
  - Automatic azimuth labeling
  - Custom numerical or categorical variable sections (if present in the dataset)

### Statistical Graphics
- Class Name: ???
- Function Name: ???
- Various statistical graphs & illustrations including:
  - Histograms/KDEs/CDFs
  - Box plots of porosity by geology
  - Pair plots/grids 
  - Ternary diagrams 
  - And more...

### 💧 Hydraulic Conductivity Calibration
- Class Name: ???
- Function Name: ???
- Supported Models:
  - Schlumberger-DOll Research (SDR) Model: $K = b \cdot \phi^m \cdot T_{2ML}^n$
  - Timur-Coates (TC) Model: $K = \left[ \left( \frac{\phi}{c} \right)^2 \cdot \frac{FFI}{BVI} \right]^2$
  - Sum of Echoes (SOE) Model: $K = c \cdot \text{SOE}^d$
- Outputs: relevant plots (data biplots and validation plot), Calibrated coefficients, R², K difference factors, RMS and other common error metrics

## 🌐 Geospatial Outputs
### 🗂 Shapefile Creation 
- Class Name: ???
- Function Name: ???
- Exports averaged BNMR parameters per borehole/geology unit in individual shape files for spatial mapping purposes

### 🌍 KML Generator (with embedded PNG logplots)
- Class Name: ???
- Function Name: ???
- This feature provides:
  - Point location of boreholes with their names
  - The PNG pictures of borehole logs are embedded in the KML file, i.e., the borehole logs can be seen by hovering or selecting the borehole on Google Earth
  - A unique data product to share with basic BNMR data/outcome with stakeholders
     

## 📥 Installation
bash
pip install pandas numpy matplotlib seaborn geopandas shapely simplekml pyproj scipy
git clone https://github.com/yourusername/dart.git

📜 License
Distributed under the MIT License. See LICENSE for more information.

DART transforms raw BNMR data into actionable insights through integrated geospatial, statistical, and geological analysis - enabling researchers to quantify subsurface water properties with unprecedented efficiency.
# mentioning publications that used dart
# maybe example or sth






































1. Core Functionality
DART provides an end-to-end solution for processing, visualizing, and interpreting borehole nuclear magnetic resonance (BNMR) data. It integrates:

Geospatial Analysis: Coordinate handling, profile azimuth calculation, KML/shapefile exports.

Statistical Modeling: Hydraulic conductivity (K) calibration, porosity distribution analysis.

Geological Integration: Lithology labeling, ternary plots of water fractions.

Automated Reporting: LAS file exports, Excel statistical reports, publication-ready figures.

2. Key Modules
2.1 Data Preprocessing & Integration
get_borehole_names(): Scans directories for borehole data files.

add_coordinates(): Merges XYZ coordinates with BNMR measurements.

remove_duplicates(): Ensures unique depth measurements per borehole.

check_NonNumeric_columns(): Cleans invalid entries in geology/hydrology data.

merge_txt_files(): Combines project datasets for unified analysis.

2.2 Visualization Engine
Log Plots (logplot())

Combines:

Spin-echo decay heatmaps

Water partitioning (clay-bound, capillary-bound, free porosity)

T₂ distribution curves

Noise profiles

Hydraulic conductivity (K) from field/BNMR models

Customizable: Color palettes, noise filtering, K-plot overlays.

Section Plots (section_plot(), section_plot_geo(), etc.)

Projects data along user-defined transects:

Geology: Lithology-colored intervals from symbology files.

Porosity: Stacked clay/capillary/free water fractions.

Custom Variables: T₂ML, SOE, or user-defined parameters.

Features:

Interactive point selection (Tkinter GUI) for profile endpoints.

Topography-aware elevation scaling.

Automatic azimuth labeling (e.g., "N-S", "NE-SW").

Statistical Graphics

Histograms/KDEs/CDFs (hist_kde_cfd_plot).

Box plots of porosity by geology (boxplot_WC).

Pair plots/grids for parameter correlations (water_fractions_pairplot).

Ternary diagrams (ternary_plot).

2.3 Geological & Hydrological Tools
Geolabeling (geo_lable())

Assigns lithology symbols to BNMR measurements using TOP/BOTTOM intervals.

K Model Calibration (calibrate_K_estimation_models)

Fits field K data to NMR-derived models:

SDR: 
K
=
b
⋅
ϕ
m
⋅
T
2
M
L
n
K=b⋅ϕ 
m
 ⋅T 
2ML
n
​
 

Timur-Coates: 
K
=
[
(
ϕ
c
)
2
⋅
F
F
I
B
V
I
]
2
K=[( 
c
ϕ
​
 ) 
2
 ⋅ 
BVI
FFI
​
 ] 
2
 

SOE: 
K
=
c
⋅
SOE
d
K=c⋅SOE 
d
 

Outputs: Calibrated coefficients, R², K difference factors.

Water Partitioning: Classifies porosity into clay-bound, capillary-bound, and free fractions.

2.4 Geospatial Outputs
Shapefile Creation (create_avg_BNMR_shape_files)

Exports averaged BNMR parameters (min/mean/max) per borehole/geology.

KML Generator (export_kml)

Embeds borehole locations + PNG log plots into Google Earth.

2.5 Export Utilities
LAS Exporter (export_las)

Generates industry-standard LAS files for petrophysical software.

Statistical Reports (statistical_report)

Excel files with porosity/T₂ statistics grouped by geology.

3. Advanced Features
Interactive Profile Selection:
GUI for defining section start/end points with buffer distance.

python
section_plot(data, variable="mlT2", point_start=(x1,y1), point_end=(x2,y2))
Noise-Aware Processing:
Auto-exclusion of high-noise measurements (configurable threshold).

Multi-Project Support:
Merge datasets from different campaigns via merge_txt_files().

Automated Section Grids:
Plot geology, porosity, and custom variables in aligned profiles:

python
section_plot_all(data_geo, data_nmr, geo_symbology_file, "mlT2")
4. Class Architecture
Data Class
Purpose: Borehole-centric data management.

Methods:

import_data(): Loads Vista Clara exports (spin echo, T₂ bins, porosity).

logplot(): Generates multi-panel borehole logs.

geo_lable(): Tags BNMR data with geology from external files.

export_las()/export_kml(): Standardized output generation.

Statistics Class
Purpose: Dataset-wide statistical modeling.

Methods:

qc_dataset(): Noise distribution analysis.

calibrate_K_estimation_models(): Fits K models to field data.

boxplot_WC()/kde_comparison_plot(): Comparative visualizations.

statistical_report(): Exports descriptive statistics.

5. Technical Highlights
Dynamic Plot Styling:
Context-aware color scales (e.g., log scaling for T₂ML).

Topography Integration:
Elevation-based depth referencing in sections.

Geology-Averaged Analysis:
Shapefiles per lithology unit (e.g., Sandstone_Avg_BNMR.shp).

3D-Ready Workflow:
XYZ coordinates enable integration with modeling software (e.g., GOCAD).

6. Applications
Aquifer Characterization:
Map free vs. bound water in hydrogeological units.

Geotechnical Engineering:
Correlate T₂ML distributions with permeability.

Mineral Exploration:
Identify clay-bound water in alteration zones.

Academic Research:
Batch-process field campaigns into structured datasets.

7. Dependencies
Library	Role
pandas	Data manipulation
geopandas	Shapefile creation
matplotlib	Publication-quality plotting
seaborn	Statistical graphics
scipy	Curve fitting, interpolation
simplekml	Google Earth KML export
shapely	Geometric operations (buffers, points)
Summary
DART transforms raw Vista Clara BNMR exports into actionable insights through:

Unified Data Management: Merge multi-project data, add coordinates, remove noise.

Rich Visualization: Log plots, geological sections, statistical diagrams.

Model Calibration: Convert NMR parameters to hydraulic conductivity.

Geospatial Integration: KML/shapefile exports for GIS workflows.

Automation: Batch processing for large datasets.

Ideal for hydrogeologists, geophysicists, and researchers quantifying subsurface water properties.

