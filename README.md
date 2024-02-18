MultiPeak Ice (MPI) is a new processing approach designed specifically for complex ice surfaces, where the majority of ice mass imbalance occurs, and is able to reliably retrieve multiple elevation measurements from a single altimetry echo. MPI_retrackWaveform.py is the main code. Download relevant data (see below), revise the data location as per your directory, and add the code directory to your Python development environment (e.g., Spdyer).

1.	Download Sentinel-3A data from Copernicus Data Space Ecosystem at https://dataspace.copernicus.eu/. Change the directory of S3A to your own folder in MPI_retrackWaveform.py (Lines 1073-1074). One example of S3A data is below:

/cycle044/S3A_SR_2_LAN____20190505T140913_20190505T145942_20190530T202816_3029_044_210______LN3_O_NT_003.SEN3/enhanced_measurement.nc


2.	Download Zwally_GIS_basins_2km.nc from the data folder in the repository and change it to your own direction at Line 133 in mask.py
 

3.	Download DEMs
MPI now supports two DEMs, choose either one of them for your research and you can add your own DEMs by revising the dem.py
 
  (a) 1km DEM downloaded from: http://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/mosaic/v3.0/1km/arcticdem_mosaic_1km_v3.0.tif.
Download the dem and rename it as ‘arcticdem_mosaic_1km_v3.0.tif’. Make sure to change the data directory to your own folder at Line 42 in dem.py
 
  (b) 100m DEM downloaded from: https://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/mosaic/v3.0/100m/arcticdem_mosaic_100m_v3.0.tif
Download the DEM, apply a 9-year linear dh/dt to the DEM(optional), and rename it as ‘arcticdem_mosaic_100m_v3.0_greenland_9dhdt.tif’. Make sure to change the data directory to your own folder at Line 74 in dem.py

4. Run MPI_retrackWaveform.py

Contact: q.huang13@lancaster.ac.uk

Citation:
Huang, Q., McMillan, M., Muir, A., Phillips, J., & Slater, T. (2024). Multipeak retracking of radar altimetry waveforms over ice sheets. Remote Sensing of Environment, 303, 114020
