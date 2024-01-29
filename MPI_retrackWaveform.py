# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------
import time  # to time the code
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import numpy as np
from scipy import interpolate
from PIL import Image
from netCDF4 import Dataset
import pyproj as proj
import pandas as pd
from matplotlib import path
from skimage.transform import resize
import pickle
from scipy.signal import find_peaks
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
from areas import Area, landice_areas
# from masks import Masks, Mask
from dems import Dem
import sys
import matplotlib.pyplot as plt
import gc
from slopeCorrect3D import slopeCorrect3D
from extractDEM import extractDEM
from misc import getREMASegment, getLeadingEdge, getHeading
import os

from mission import Mission  # class to handle common altimetry mission functions
gc.collect()
"""
Class to process and store S3 track data
"""


class S3Track:

    """
    __init__
    """

    def __init__(self, trackID, outdir, maskName=None, outputProjection="epsg:4326", applyRetracking=True, applyCorrections=True, applySlopeCorrections=True, save=True, dem=None):

        self.trackID = trackID
        self.outdir = outdir
        self.maskName = maskName
        self.outputProjection = outputProjection
        self.applyRetracking = applyRetracking
        self.applyCorrections = applyCorrections
        self.applySlopeCorrections = applySlopeCorrections
        self.save = save
        self.dem = dem

        # ----------------------------------------------------------------------
        # Get mask
        # ----------------------------------------------------------------------

        data = Dataset(fname)
        lats_nadir, lons_nadir = thismission.get_20hz_lat_lon(data, nadir=True)
        lats_poca, lons_poca = thismission.get_20hz_lat_lon(data, nadir=False)

        nadirLat_1hz, nadirLon_1hz = thismission.get_1hz_lat_lon(data)

        bounded_lat, bounded_lon, bounded_indices, ninside = thisarea.inside_latlon_bounds(
            lats_nadir, lons_nadir)
        inmask, xb, yb = thismask.points_inside(
            bounded_lat, bounded_lon, thisarea.basin_numbers)  # returns 1s for inside, 0s outside
        inMask_20hz = bounded_indices[inmask]
        inMask_20hz_order = np.array(range(0, len(inMask_20hz)))

        bounded_lat_1hz, bounded_lon_1hz, bounded_indices_1hz, ninside_1hz = thisarea.inside_latlon_bounds(
            nadirLat_1hz, nadirLon_1hz)
        inmask_1hz, xb_1hz, yb_1hz = thismask.points_inside(
            bounded_lat_1hz, bounded_lon_1hz, thisarea.basin_numbers)  # returns 1s for inside, 0s outside
        inMask_1hz = bounded_indices_1hz[inmask_1hz]

        # ----------------------------------------------------------------------
        # Check output projection is epsg:3413 if slope correcting
        # ----------------------------------------------------------------------

        if self.applySlopeCorrections == True and self.outputProjection != "epsg:3413":
            self.outputProjection = "epsg:3413"  # epsg:3413
            print("WARNING: slope correction limited to  DEM! Output projection set to Arctic Polar Stereographic.")

        # ----------------------------------------------------------------------
        # Get, process, and store track data
        # ----------------------------------------------------------------------

        print("\n----------------------------------------------------------------------")
        print("Getting data for track " + str(trackID) + "...")
        print("----------------------------------------------------------------------")

        # 5775 5836 6147 5646,5696 allow for retracking a slice of the data (larger than 20 records), as opposed to the entire track
        self.data = self.getData(lons_nadir, lats_nadir, nadirLon_1hz,
                                 nadirLat_1hz, lons_poca, lats_poca, inMask_20hz, inMask_1hz)
        self.data['filename'] = fname
        self.data['data_index'] = inMask_20hz
        self.data['data_order'] = inMask_20hz_order

        if self.applyRetracking == True:
            print(
                "\n----------------------------------------------------------------------")
            print("Retracking data for track " + str(trackID) + "...")
            print(
                "----------------------------------------------------------------------")

            dem = Dem(self.dem)  # awi_grn_1km  arcticdem_1km
            POCA_z = dem.interp_dem(
                self.data['POCA Latitude'], self.data['POCA Longitude'], method='linear', xy_is_latlon=True)

            self.data['POCA z'] = POCA_z

            slopeCorrect_data = slopeCorrect3D(self.data["Waveform"], self.data["Nadir x"], self.data["Nadir y"], self.data["Tracker Range"],
                                               self.data["Altitude"], self.data["POCA x"], self.data["POCA y"], self.data["POCA z"], self.dem, dem_inFootprint=None)

            self.data = {**self.data, **slopeCorrect_data}

            self.data = {**self.data, **self.retrackData()}

            if self.applyCorrections == True:
                print(
                    "\n----------------------------------------------------------------------")
                print("Correcting data for track " + str(trackID) + "...")
                print(
                    "----------------------------------------------------------------------")
                self.data = {**self.data, **self.correctData()}

                if self.applySlopeCorrections == True:
                    print(
                        "\n----------------------------------------------------------------------")
                    print("Slope-correcting data for track " +
                          str(trackID) + "...")
                    print(
                        "----------------------------------------------------------------------")
                    # apply slope correction here
                    self.data = {**self.data, **self.applySlopeCorrect()}

        if self.save == True:

            outDirCycle = self.outdir + '/' + \
                'c'+str(self.trackID[-53:-51])+'/'
            isExist = os.path.exists(outDirCycle)
            if not isExist:

                # Create a new directory because it does not exist
                os.makedirs(outDirCycle)

            outputName = outDirCycle + self.trackID[-123:-24] + ".data"

            with open(outputName, "wb") as fp:
                # del self.data['Oversampled Waveform (Smoothed)']
                del self.data['Oversampled bin']
                del self.data['Oversampled Waveform']
                del self.data['Waveform (Smoothed)']
                pickle.dump(self.data, fp)


    """
    # Function to collect the L2 S3 raw track data
    """

    def getData(self, nadirLon, nadirLat, nadirLon_1hz, nadirLat_1hz, POCALon, POCALat, inMask_20hz, inMask_1hz):

        data = Dataset(self.trackID)

        # ----------------------------------------------------------------------
        # Convert coordinate variables to output projection (need to do this for masking to work correctly)
        # ----------------------------------------------------------------------

        p1 = proj.Proj(init='epsg:4326')  # WGS84
        p2 = proj.Proj(init=self.outputProjection)

        nadirX, nadirY = proj.transform(p1, p2, nadirLon, nadirLat)
        nadirX_1hz, nadirY_1hz = proj.transform(
            p1, p2, nadirLon_1hz, nadirLat_1hz)
        POCAX, POCAY = proj.transform(p1, p2, POCALon, POCALat)

        if np.size(inMask_20hz) == 0:  # check whether there are any measurements within the mask
            print(
                "WARNING: there are no measurements within the given mask (" + self.maskPath + ")!")
            return

        # ----------------------------------------------------------------------
        # Find track heading
        # ----------------------------------------------------------------------

        # compute change in latitude between successive records in middle of pass
        latDiff = nadirLat[(len(nadirLat)//2) + 1] - \
            nadirLat[(len(nadirLat)//2)]

        if latDiff > 0:  # identify track as ascending if first and last points are in S and N hemispheres respectively
            heading = "ascending"

        elif latDiff < 0:  # identify track as descending if first and last points are in N and S hemispheres respectively
            heading = "descending"

        else:
            heading = np.nan

        # ----------------------------------------------------------------------
        # Read in global attributes
        # ----------------------------------------------------------------------

        productName = data.product_name  # product filename
        cycle = data.cycle_number  # cycle number
        relativePassNumber = data.pass_number  # relative pass number
        firstTime = data.first_meas_time  # time of first measurement

        # ----------------------------------------------------------------------
        # Read in KU-band SAR variables and crop to mask
        # ----------------------------------------------------------------------

        POCALon = POCALon[inMask_20hz]
        POCALat = POCALat[inMask_20hz]
        POCAX = POCAX[inMask_20hz]
        POCAY = POCAY[inMask_20hz]
        nadirLon = nadirLon[inMask_20hz]  # nadir latitude
        nadirLat = nadirLat[inMask_20hz]  # nadir longitude
        nadirX = nadirX[inMask_20hz]  # nadir x
        nadirY = nadirY[inMask_20hz]  # nadir y
        # time in seconds since 2000-01-01 00:00:00.0
        t0000 = data.variables['time_20_ku'][inMask_20hz]

        # elevation including corrections for USO drift correction (uso_cor_20_ku), internal path correction (int_path_cor_20_ku), distance antenna-COG (cog_cor_01) and Doppler slope correction (dop_slope_cor_20_ku)
        icesheetElevation_L2 = data.variables["elevation_ice_sheet_20_ku"][inMask_20hz]
        retrackedRange_L2 = data.variables["range_ice_sheet_20_ku"][inMask_20hz]

        # ----------------------------------------------------------------------
        # Read in KU-band 20Hz L1b arrays and crop to mask
        # ----------------------------------------------------------------------

        waveform = data.variables['waveform_20_ku'][inMask_20hz]  # waveforms
        # Tracker Range corrected for USO frequency drift (uso_cor_20_ku) and internal path corrections (int_path_cor_20_ku)
        trackerRange = data.variables['tracker_range_20_ku'][inMask_20hz]
        # satellite altitude
        altitude = data.variables['alt_20_ku'][inMask_20hz]

        # ----------------------------------------------------------------------
        # Read in reference dem and ocog elevation
        # ----------------------------------------------------------------------

        # 'range_ice_sheet_20_ku'
        retrackedRange_L2_ocog = data.variables['range_ocog_20_ku'][inMask_20hz]
        ocogElevation_L2 = data.variables['elevation_ocog_20_ku'][inMask_20hz]
        # ----------------------------------------------------------------------
        # Read in KU-band 20Hz flags and crop to mask
        # ----------------------------------------------------------------------

        # surface type - flag 4 is ice
        surfaceClass = data.variables['surf_class_20_ku'][inMask_20hz]
        # waveform quality flag
        waveformQuality = data.variables['waveform_qual_ice_20_ku'][inMask_20hz]

        # ----------------------------------------------------------------------
        # Read in KU-band 1Hz vectors and crop to mask
        # ----------------------------------------------------------------------

        # time in seconds since 2000-01-01 00:00:00.0
        t0000_1hz = data.variables['time_01'][inMask_1hz]
        nadirLon_1hz = nadirLon_1hz[inMask_1hz]  # 1 Hz longitude
        nadirLat_1hz = nadirLat_1hz[inMask_1hz]  # 1 Hz latitude
        nadirX_1hz = nadirX_1hz[inMask_1hz]  # 1 Hz x
        nadirY_1hz = nadirY_1hz[inMask_1hz]  # 1 Hz y

        # ----------------------------------------------------------------------
        # Read in KU-band 1Hz corrections and crop to mask
        # ----------------------------------------------------------------------

        # 1 Hz distance antenna-COG correction on altimeter range
        COGCorrection_1hz = data.variables['cog_cor_01'][inMask_1hz]
        # 1 Hz Modeled instrumental correction on the altimeter range
        modeledInstrumentCorrection_1hz = data.variables['mod_instr_cor_range_01_ku'][inMask_1hz]
        # 1 Hz ECMWF dry tropospheric correction, in m, must be added (negative value) to the instrument range to correct for dry tropospheric range delays of the radar pulse
        dryTroposphericCorrection_1hz = data.variables['mod_dry_tropo_cor_meas_altitude_01'][inMask_1hz]
        # 1 Hz ECMWF wet tropospheric correction, in m, must be added (negative value) to the instrument range to correct for wet tropospheric range delays of the radar pulse
        wetTroposphericCorrection_1hz = data.variables['mod_wet_tropo_cor_meas_altitude_01'][inMask_1hz]
        # 1 Hz GIM ionospheric correction, in m, must be added (negative value) to the instrument range to correct for ionospheric range delays of the radar pulse
        ionosphericCorrection_1hz = data.variables['iono_cor_gim_01_ku'][inMask_1hz]
        # 1 Hz ECMWF low frequency IBE correction, in m, sea surface height correction due to air pressure at low frequency
        IBECorrection_1hz = data.variables['inv_bar_cor_01'][inMask_1hz]
        # 1 Hz high frequency IBE to be applied as correction to IBE, in m, sea surface height correction due to air pressure and wind at high frequency
        IBECorrection_HF_1hz = data.variables['hf_fluct_cor_01'][inMask_1hz]
        # 1 Hz tide from FES2004 model, in m, includes the corresponding loading tide (load_tide_sol2_01) and equilibrium long-period ocean tide height (ocean_tide_eq_01). The permanent tide (zero frequency) is not included in this parameter because it is included in the geoid and mean sea surface (geoid_01, mean_sea_surf_sol1_01)
        oceanTide_1hz = data.variables['ocean_tide_sol2_01'][inMask_1hz]
        # 1 Hz ocean load tide from FES2004 model, in m, note already applied in fes tide correction
        loadTide_1hz = data.variables['load_tide_sol2_01'][inMask_1hz]
        # 1 Hz solid earth tide in m
        solidEarthTide_1hz = data.variables['solid_earth_tide_01'][inMask_1hz]
        # 1 Hz pole tide, in m, accounting for deformation of the Earth induced by polar motion
        poleTide_1hz = data.variables['pole_tide_01'][inMask_1hz]

        data.close()  # close track nc connection

        # ----------------------------------------------------------------------
        # Form output dictionary
        # ----------------------------------------------------------------------

        output = {
            # write global attribute fields
            "Product Name": productName,
            "Heading": heading,
            "Relative Pass Number": relativePassNumber,
            "Cycle": cycle,
            "First Measured Time": firstTime,
            # "Data start": data_start,

            # write 20 hz ku-band arrays
            "Seconds Since 0000": t0000,
            "Nadir Longitude": nadirLon,
            "Nadir Latitude": nadirLat,
            "Nadir x": nadirX,
            "Nadir y": nadirY,
            "POCA Longitude": POCALon,
            "POCA Latitude": POCALat,
            "POCA x": POCAX,
            "POCA y": POCAY,
            "Ice Sheet Elevation (L2)": icesheetElevation_L2,
            "Retracked Range (L2)": retrackedRange_L2,
            "OCOG Elevation (L2)": ocogElevation_L2,
            "Retracked Range (L2) OCOG": retrackedRange_L2_ocog,
            "Tracker Range": trackerRange,
            "Waveform": waveform,
            "Altitude": altitude,
            "Surface Class": surfaceClass,
            "Waveform Quality": waveformQuality,

            # write 1Hz arrays
            "Seconds Since 0000 (1Hz)": t0000_1hz,
            "Nadir Longitude (1Hz)": nadirLon_1hz,
            "Nadir Latitude (1Hz)": nadirLat_1hz,
            "Nadir x (1Hz)": nadirX_1hz,
            "Nadir y (1Hz):": nadirY_1hz,

            # write 1hz instrumental corrections
            "COG Correction (1Hz)": COGCorrection_1hz,

            # write 1hz geophysical corrections
            "Modeled Instrument Correction (1Hz)": modeledInstrumentCorrection_1hz,
            "Dry Topospheric Correction (1Hz)": dryTroposphericCorrection_1hz,
            "Wet Topospheric Correction (1Hz)": wetTroposphericCorrection_1hz,
            "Ionspheric Correction (1Hz)": ionosphericCorrection_1hz,
            "IBE Correction (1Hz)": IBECorrection_1hz,
            "High Frequency IBE Correction (1Hz)": IBECorrection_HF_1hz,
            "Ocean Tide (1Hz)": oceanTide_1hz,
            "Ocean Load Tide (1Hz)": loadTide_1hz,
            "Solid Earth Tide (1Hz)": solidEarthTide_1hz,
            "Pole Tide (1Hz)": poleTide_1hz
        }

        # ----------------------------------------------------------------------
        # Return output
        # ----------------------------------------------------------------------

        return(output)

    """
    # Function to retrack the L2 S3 track data
    """

    def retrackData(self):

        # ----------------------------------------------------------------------
        # Define variables
        # ----------------------------------------------------------------------
        c = 299792458  # speed of light (m/s)
        B = 320000000  # chirp bandwidth used (Hz) from Donlon 2012 table 6
        # 43 for S3 ku-band - this is the bin number that the tracker range in a waveform should correspond to
        referenceBinIndex = 43

        rangeBinSize = c/(2*B)  # compute distance between each bin in meters

        # waveform oversampling factor (i.e. generate wfOversamplingFactor times more points when oversampling)
        wfOversamplingFactor = 100

        waveform = self.data["Waveform"]
        trackerRange = self.data["Tracker Range"]
        altitude = self.data["Altitude"]

        poca_lon = self.data["POCA Longitude"]
        poca_lat = self.data["POCA Latitude"]

        dem = Dem(self.dem)  # awi_grn_1km  arcticdem_1km
        reference_dem = dem.interp_dem(
            poca_lat, poca_lon, method='linear', xy_is_latlon=True)

        if np.isnan(reference_dem).all():
            print(
                "dem is not compatible with the mission data, multipeak retracker not performed!")
            sys.exit()

        numRecords = len(waveform)
        binSize = len(waveform[0])
        interpolatedBinSize = len(waveform[0])*wfOversamplingFactor

        retrackThreshold_tfmra = 0.5  # define threshold for leading edge maximum retracker

        # ----------------------------------------------------------------------
        # Define quality thresholds
        # ----------------------------------------------------------------------

        # if mean amplitude in noise bins exceeds threshold then reject waveform
        noiseThreshold = 0.3

        # power must be this much greater than thermal noise to be identified as leading edge
        leThreshold_id = 0.05

        # define threshold on normalised amplitude change which is required to be accepted as lead edge
        leThreshold_dp = 0.2

        # ----------------------------------------------------------------------
        # Initialise output arrays
        # ----------------------------------------------------------------------

        leStart = np.full((numRecords, 3), np.nan)
        leStop = np.full((numRecords, 3), np.nan)

        retrackPoint_tfmra = np.full((numRecords, 3), np.nan)
        retrackPoint_tfmra_LE2 = np.full((numRecords, 3), np.nan)
        retrackPoint_tfmra_LE3 = np.full((numRecords, 3), np.nan)

        num_peaks = np.full((numRecords), np.nan)
        retrackFlag = np.zeros((numRecords, 8))
        wfSmooth = np.full((numRecords, binSize), np.nan)
        wfNormInterpolated = np.full((numRecords, interpolatedBinSize), np.nan)
        wfNormSmoothInterpolated = np.full(
            (numRecords, interpolatedBinSize), np.nan)
        wfNoiseMean = np.full(numRecords, np.nan)
        wfNormInterpolated_d1 = np.full(
            (numRecords, interpolatedBinSize), np.nan)
        wfBinNumberIndices = np.full((numRecords, interpolatedBinSize), np.nan)

        # ----------------------------------------------------------------------
        # Retrack each waveform
        # ----------------------------------------------------------------------

        for record in range(numRecords):  # loop through the records

            # ----------------------------------------------------------------------
            # Load and normalise waveform
            # ----------------------------------------------------------------------

            wf = waveform[record]  # load waveform

            wfNorm = wf/max(wf)  # normalise waveform

            # ----------------------------------------------------------------------
            # Smooth waveform
            # ----------------------------------------------------------------------
            # pseudo-Gaussian smoothing (3 passes of sliding-average smoothing)
            wfNormSmooth = savgol_filter(wfNorm, 9, 3)

            np.insert(wfNormSmooth, 0, np.nan)  # set end values as nan
            # set end values as nan
            np.insert(wfNormSmooth, len(wfNorm), np.nan)
            wfSmooth[record] = wfNormSmooth*max(wf)  # store smoothed waveform

            # ----------------------------------------------------------------------
            # Compute thermal noise
            # ----------------------------------------------------------------------

            # sort power values of unsmoothed waveform in ascending order
            wfSorted = np.sort(wfNorm)

            # estimate noise based on lowest 6 samples
            wfNoiseMean[record] = np.mean(wfSorted[0:6])

            # ----------------------------------------------------------------------
            # Quality check 1 - check if mean noise is above the predefined threshold
            # ----------------------------------------------------------------------

            if wfNoiseMean[record] > noiseThreshold:
                print("WARNING: mean noise above threshold of " +
                      str(noiseThreshold) + " for waveform " + str(record) + ".")
                retrackFlag[record, 0] = 1
                continue

            # ----------------------------------------------------------------------
            # Oversample using spline
            # ----------------------------------------------------------------------

            # compute bin interval for oversampled waveform
            wfOversamplingInterval = 1/wfOversamplingFactor

            # create oversampled waveform bin indices
            wfBinNumberIndices[record] = np.arange(
                0,  len(wf), wfOversamplingInterval)

            wfNormSmoothInterpolated[record] = interpolate.splev(wfBinNumberIndices[record], interpolate.splrep(
                range(len(wf)), wfNormSmooth))  # compute spline and interpolated values of smoothed waveform at bin numbers

            # wfNormInterpolated[record] = interpolate.splev( wfBinNumberIndices[record], interpolate.splrep(range(len(wf)), wfNorm)) # compute spline and interpolated values of non smoothed waveform at bin numbers
            wfNormInterpolated[record] = interpolate.splev(wfBinNumberIndices[record], interpolate.splrep(range(
                len(wf)), wfNorm, k=1))  # compute spline and interpolated values of non smoothed waveform at bin numbers

            # ----------------------------------------------------------------------
            # Compute derivatives
            # ----------------------------------------------------------------------

            # compute first derivative of smoothed waveform
            wfNormSmoothInterpolated_d1 = np.gradient(
                wfNormSmoothInterpolated[record], wfOversamplingInterval)

            # compute first derivative of unsmoothed waveform
            wfNormInterpolated_d1[record] = np.gradient(
                wfNormInterpolated[record], wfOversamplingInterval)

            # ----------------------------------------------------------------------
            # Initiate parameters for iteratively finding leading edge with amplitude above predefined threshold
            # ----------------------------------------------------------------------

            leIndex_previous = 0  # previous leading edge index
            le_dp = 0  # normalised amplitude change

            # ----------------------------------------------------------------------
            # Loop through leading edges until one exceeds minimum amplitude threshold or end of waveform is reached
            # ----------------------------------------------------------------------
            while le_dp < leThreshold_dp:

                try:
                    # find next leading edge candidates (at least 1 original bin width from last) which are above the threshold
                    leIndices = np.where((wfNormSmoothInterpolated[record] > (wfNoiseMean[record] + leThreshold_id)) & (
                        wfBinNumberIndices[record] > wfBinNumberIndices[record][leIndex_previous + wfOversamplingFactor]))
                except:
                    print(
                        "WARNING: no leading edge found before end of waveform " + str(record) + " reached.")
                    retrackFlag[record, 3] = 1
                    break  # leIndex_previous 12700
                # ----------------------------------------------------------------------
                # Quality check 2 - check if no samples are sufficiently above the noise or large enough leading edge
                # ----------------------------------------------------------------------

                if np.size(leIndices) == 0:
                    print("WARNING: no samples in waveform " + str(record) +
                          " are sufficiently above noise and/or there is no leading edge of sufficient size.")
                    retrackFlag[record, 1] = 1
                    break

                else:  # else take the first index found
                    leIndex = leIndices[0][0]

                # ----------------------------------------------------------------------
                # If leading edge exists find position
                # ----------------------------------------------------------------------
                # find stationary points on leading edge where gradient first becomes negative again
                peakIndices = np.where((wfNormSmoothInterpolated_d1 <= 0) & (
                    wfBinNumberIndices[record] > wfBinNumberIndices[record][leIndex]))

                # ----------------------------------------------------------------------
                # Quality check 3 - Check if a waveform peak can be identified after the start of the leading edge
                # ----------------------------------------------------------------------

                if np.size(peakIndices) == 0:
                    print("WARNING: no peak in waveform " + str(record) +
                          " could be indentified after the start of the leading edge.")
                    retrackFlag[record, 2] = 1
                    break

                else:  # else take the first index found
                    firstPeakIndex = peakIndices[0][0]

                # ----------------------------------------------------------------------
                # Calculate amplitude of peak above the noise floor threshold
                # ----------------------------------------------------------------------

                le_dp = wfNormSmoothInterpolated[record][firstPeakIndex] - \
                    wfNormSmoothInterpolated[record][leIndex]

                # update previous leading edge to current one in case the amplitude change threshold is not met
                leIndex_previous = firstPeakIndex

                # ----------------------------------------------------------------------
                # Quality check 4 - check if end of waveform is reached
                # ----------------------------------------------------------------------

                if leIndex_previous > len(wfBinNumberIndices[record]) - wfOversamplingFactor:
                    print(
                        "WARNING: no leading edge found before end of waveform " + str(record) + " reached.")
                    retrackFlag[record, 3] = 1
                    break

            # ----------------------------------------------------------------------
            # Move to next record if leading edge could not be found
            # ----------------------------------------------------------------------

            if retrackFlag[record, 1] == 1 or retrackFlag[record, 2] == 1 or retrackFlag[record, 3] == 1:
                continue

            try:
                LEIndexStart, LEIndexEnd, rangeToLEStart, rangeToLEEnd = getLeadingEdge(
                    self.data["Waveform"][record], self.data["Tracker Range"][record])

                LEIndexStart = LEIndexStart.astype(
                    int)*100  # transform to interpolated bin

                peaks = LEIndexEnd.astype(int)*100

            except:
                print("WARNING: no peak in waveform " + str(record) +
                      " could be indentified after the start of the leading edge.")
                continue

            # find the number of peaks in the waveform
            num_peaks[record] = len(peaks)

            retrackIndex_tfmra = np.nan
            retrackIndex_tfmra_LE2 = np.nan
            retrackIndex_tfmra_LE3 = np.nan

            for j in range(len(LEIndexStart)):

                if j == 0:

                    amplitude = wfNormInterpolated[record][peaks[j]
                                                           ] - wfNoiseMean[record]

                    # compute retracking threshold as proportion of le above noise and then add back noise component to get back to wf power
                    wfRetrackThreshold_tfmra = wfNoiseMean[record] + \
                        retrackThreshold_tfmra*amplitude

                    try:

                        retrackIndex_tfmra = np.where((wfNormInterpolated[record] > wfRetrackThreshold_tfmra) & (wfBinNumberIndices[record] > wfBinNumberIndices[record][LEIndexStart[j]]) & (
                            wfBinNumberIndices[record] < wfBinNumberIndices[record][peaks[j]]))[0][0]  # find first leading edge value above the retracking threshold for waveform

                    except:
                        retrackIndex_tfmra = np.nan

                else:
                    # define le power above noise floor based on smoothed waveform to reduce impact of speckle on estimate of peak power i.e., amplitude above the noise
                    amplitude = wfNormInterpolated[record][peaks[j]
                                                           ] - wfNormInterpolated[record][LEIndexStart[j]]

                    # compute retracking threshold as proportion of le above noise and then add back noise component to get back to wf power
                    wfretrackThreshold_tfmra = wfNormInterpolated[record][LEIndexStart[j]
                                                                          ] + retrackThreshold_tfmra*amplitude

                    if j == 1:
                        wfRetrackThreshold_tfmra_LE2 = wfretrackThreshold_tfmra
                        try:
                            retrackIndex_tfmra_LE2 = np.where((wfNormInterpolated[record] > wfRetrackThreshold_tfmra_LE2) & (wfBinNumberIndices[record] > wfBinNumberIndices[record][LEIndexStart[j]]) & (
                                wfBinNumberIndices[record] < wfBinNumberIndices[record][peaks[j]]))[0][0]  # find first leading edge value above the retracking threshold for waveform
                        except:
                            retrackIndex_tfmra_LE2 = np.nan

                    if j == 2:
                        wfRetrackThreshold_tfmra_LE3 = wfretrackThreshold_tfmra
                        try:
                            retrackIndex_tfmra_LE3 = np.where((wfNormInterpolated[record] > wfRetrackThreshold_tfmra_LE3) & (wfBinNumberIndices[record] > wfBinNumberIndices[record][LEIndexStart[j]]) & (
                                wfBinNumberIndices[record] < wfBinNumberIndices[record][peaks[j]]))[0][0]  # find first leading edge value above the retracking threshold for waveform
                        except:
                            retrackIndex_tfmra_LE3 = np.nan

            # ----------------------------------------------------------------------
            # Store waveform retracking output parameters - columns give bin number, normalised amplitude value, original amplitude value
            # ----------------------------------------------------------------------

            leStart[record] = (wfBinNumberIndices[record][leIndex], wfNormSmoothInterpolated[record]
                               [leIndex], wfNormSmoothInterpolated[record][leIndex]*max(wf))

            leStop[record] = (wfBinNumberIndices[record][firstPeakIndex], wfNormSmoothInterpolated[record]
                              [firstPeakIndex], wfNormSmoothInterpolated[record][firstPeakIndex]*max(wf))

            if retrackFlag[record, 4] != 1:
                retrackPoint_tfmra[record] = (wfBinNumberIndices[record][retrackIndex_tfmra], wfNormInterpolated[record]
                                              [retrackIndex_tfmra], wfNormInterpolated[record][retrackIndex_tfmra]*max(wf))

            if retrackFlag[record, 7] != 1:

                if ~np.isnan(retrackIndex_tfmra_LE2):
                    retrackPoint_tfmra_LE2[record] = (wfBinNumberIndices[record][retrackIndex_tfmra_LE2], wfNormInterpolated[record]
                                                      [retrackIndex_tfmra_LE2], wfNormInterpolated[record][retrackIndex_tfmra_LE2]*max(wf))
                else:
                    retrackPoint_tfmra_LE2[record] = np.nan

                if ~np.isnan(retrackIndex_tfmra_LE3):
                    retrackPoint_tfmra_LE3[record] = (wfBinNumberIndices[record][retrackIndex_tfmra_LE3], wfNormInterpolated[record]
                                                      [retrackIndex_tfmra_LE3], wfNormInterpolated[record][retrackIndex_tfmra_LE3]*max(wf))
                else:
                    retrackPoint_tfmra_LE3[record] = np.nan

        # ----------------------------------------------------------------------
        # Compute range offsets from reference to retracked bins
        # ----------------------------------------------------------------------

        rangeOffset_tfmra = retrackPoint_tfmra[:, 0] - referenceBinIndex
        rangeOffset_tfmra_LE2 = retrackPoint_tfmra_LE2[:,
                                                       0] - referenceBinIndex
        rangeOffset_tfmra_LE3 = retrackPoint_tfmra_LE3[:,
                                                       0] - referenceBinIndex

        # ----------------------------------------------------------------------
        # Convert offsets to meters
        # ----------------------------------------------------------------------

        rangeOffsetMeters_tfmra = rangeOffset_tfmra * rangeBinSize
        rangeOffsetMeters_tfmra_LE2 = rangeOffset_tfmra_LE2 * rangeBinSize
        rangeOffsetMeters_tfmra_LE3 = rangeOffset_tfmra_LE3 * rangeBinSize

        # ----------------------------------------------------------------------
        # Compute ranges to surface by applying retracker offsets to tracker
        # ----------------------------------------------------------------------

        retrackedRange_tfmra = trackerRange + rangeOffsetMeters_tfmra
        retrackedRange_tfmra_LE2 = trackerRange + rangeOffsetMeters_tfmra_LE2
        retrackedRange_tfmra_LE3 = trackerRange + rangeOffsetMeters_tfmra_LE3

        # ----------------------------------------------------------------------
        # Compute elevations
        # ----------------------------------------------------------------------

        elevation_tfmra = altitude - retrackedRange_tfmra
        elevation_tfmra_LE2 = altitude - retrackedRange_tfmra_LE2
        elevation_tfmra_LE3 = altitude - retrackedRange_tfmra_LE3

        # ----------------------------------------------------------------------
        # Form output dictionary
        # ----------------------------------------------------------------------

        output = {
            # smoothed waveform
            "Waveform (Smoothed)": wfSmooth,
            "Oversampled Waveform": wfNormInterpolated,
            # "Oversampled Waveform (Smoothed)": wfNormSmoothInterpolated,

            # peak difference between L2 products and multipeak retracker
            "Number of peaks": num_peaks,

            # leading edge identifiers - columns give bin number, normalised amplitude value, original amplitude value
            "Leading Edge Start": leStart,
            "Leading Edge Stop": leStop,

            # Retrack point - columns give bin number, normalised amplitude value, original amplitude value
            "Retrack Point": {"tfmra": retrackPoint_tfmra, "tfmra_LE2": retrackPoint_tfmra_LE2, "tfmra_LE3": retrackPoint_tfmra_LE3, },

            # Range offset - columns give bin number and meters
            "Range Offset": {"tfmra": [rangeOffset_tfmra, rangeOffsetMeters_tfmra], "tfmra_LE2": [rangeOffset_tfmra_LE2, rangeOffsetMeters_tfmra_LE2], "tfmra_LE3": [rangeOffset_tfmra_LE3, rangeOffsetMeters_tfmra_LE3], },

            # Retracked range
            "Retracked Range": {"tfmra": retrackedRange_tfmra, "tfmra_LE2": retrackedRange_tfmra_LE2, "tfmra_LE3": retrackedRange_tfmra_LE3, },

            # uncorrected elevations
            "Uncorrected Elevation": {"tfmra": elevation_tfmra, "tfmra_LE2": elevation_tfmra_LE2, "tfmra_LE3": elevation_tfmra_LE3, },

            # retracker flags
            "Retracker Flags": retrackFlag,

            # Oversampled bin
            "Oversampled bin": wfBinNumberIndices,

        }

        # ----------------------------------------------------------------------
        # Return tracked data
        # ----------------------------------------------------------------------

        return(output)

    """
    # Function to apply instrumental and geophysical corrections to the L2 S3 track data
    """

    def correctData(self):

        # ----------------------------------------------------------------------
        # Extract vectors of 20 Hz fields
        # ----------------------------------------------------------------------

        # extract location fields
        t0000 = self.data["Seconds Since 0000"].data
        altitude = self.data["Altitude"]

        # get retracked ranges
        retrackedRanges = (self.data["Retracked Range"]["tfmra"], self.data["Retracked Range"]
                           ["tfmra_LE2"], self.data["Retracked Range"]["tfmra_LE3"],)

        # get uncorrected elevations
        uncorrectedElevations = (self.data["Uncorrected Elevation"]["tfmra"], self.data["Uncorrected Elevation"]
                                 ["tfmra_LE2"], self.data["Uncorrected Elevation"]["tfmra_LE3"],)

        # ----------------------------------------------------------------------
        # Extract vectors of 1 Hz fields
        # ----------------------------------------------------------------------

        # extract location fields
        t0000_1hz = self.data["Seconds Since 0000 (1Hz)"].data

        # extract instrument correction fields
        COGCorrection_1hz = self.data["COG Correction (1Hz)"]

        # extract geophysical correction fields
        dryTroposphericCorrection_1hz = self.data[
            "Dry Topospheric Correction (1Hz)"]
        wetTroposphericCorrection_1hz = self.data[
            "Wet Topospheric Correction (1Hz)"]
        ionosphericCorrection_1hz = self.data["Ionspheric Correction (1Hz)"]
        IBECorrection_1hz = self.data["IBE Correction (1Hz)"].data
        IBECorrection_HF_1hz = self.data["High Frequency IBE Correction (1Hz)"].data
        oceanTide_1hz = self.data["Ocean Tide (1Hz)"].data
        loadTide_1hz = self.data["Ocean Load Tide (1Hz)"]
        solidEarthTide_1hz = self.data["Solid Earth Tide (1Hz)"]
        poleTide_1hz = self.data["Pole Tide (1Hz)"]

        # ----------------------------------------------------------------------
        # Resample 1 Hz data to 20 Hz using linear interpolation
        # ----------------------------------------------------------------------

        if len(t0000_1hz) > 1:
            COGCorrection_20hz = interpolate.interp1d(
                t0000_1hz, COGCorrection_1hz, kind="linear", fill_value="extrapolate")(t0000)
            dryTroposphericCorrection_20hz = interpolate.interp1d(
                t0000_1hz, dryTroposphericCorrection_1hz, kind="linear", fill_value="extrapolate")(t0000)
            wetTroposphericCorrection_20hz = interpolate.interp1d(
                t0000_1hz, wetTroposphericCorrection_1hz, kind="linear", fill_value="extrapolate")(t0000)
            ionosphericCorrection_20hz = interpolate.interp1d(
                t0000_1hz, ionosphericCorrection_1hz, kind="linear", fill_value="extrapolate")(t0000)
            IBECorrection_20hz = interpolate.interp1d(
                t0000_1hz, IBECorrection_1hz, kind="linear", fill_value="extrapolate")(t0000)
            IBECorrection_HF_20hz = interpolate.interp1d(
                t0000_1hz, IBECorrection_HF_1hz, kind="linear", fill_value="extrapolate")(t0000)
            oceanTide_20hz = interpolate.interp1d(
                t0000_1hz, oceanTide_1hz, kind="linear", fill_value="extrapolate")(t0000)
            loadTide_20hz = interpolate.interp1d(
                t0000_1hz, loadTide_1hz, kind="linear", fill_value="extrapolate")(t0000)
            solidEarthTide_20hz = interpolate.interp1d(
                t0000_1hz, solidEarthTide_1hz, kind="linear", fill_value="extrapolate")(t0000)
            poleTide_20hz = interpolate.interp1d(
                t0000_1hz, poleTide_1hz, kind="linear", fill_value="extrapolate")(t0000)
        elif len(t0000_1hz) == 1:
            COGCorrection_20hz = np.asarray(
                [COGCorrection_1hz] * len(t0000)).flatten()
            dryTroposphericCorrection_20hz = np.asarray(
                [dryTroposphericCorrection_1hz] * len(t0000)).flatten()
            wetTroposphericCorrection_20hz = np.asarray(
                [wetTroposphericCorrection_1hz] * len(t0000)).flatten()
            ionosphericCorrection_20hz = np.asarray(
                [ionosphericCorrection_1hz] * len(t0000)).flatten()
            IBECorrection_20hz = np.asarray(
                [IBECorrection_1hz] * len(t0000)).flatten()
            IBECorrection_HF_20hz = np.asarray(
                [IBECorrection_HF_1hz] * len(t0000)).flatten()
            oceanTide_20hz = np.asarray([oceanTide_1hz] * len(t0000)).flatten()
            loadTide_20hz = np.asarray([loadTide_1hz] * len(t0000)).flatten()
            solidEarthTide_20hz = np.asarray(
                [solidEarthTide_1hz] * len(t0000)).flatten()
            poleTide_20hz = np.asarray([poleTide_1hz] * len(t0000)).flatten()
        else:
            COGCorrection_20hz = np.asarray([np.nan] * len(t0000)).flatten()
            dryTroposphericCorrection_20hz = np.asarray(
                [np.nan] * len(t0000)).flatten()
            wetTroposphericCorrection_20hz = np.asarray(
                [np.nan] * len(t0000)).flatten()
            ionosphericCorrection_20hz = np.asarray(
                [np.nan] * len(t0000)).flatten()
            IBECorrection_20hz = np.asarray([np.nan] * len(t0000)).flatten()
            IBECorrection_HF_20hz = np.asarray([np.nan] * len(t0000)).flatten()
            oceanTide_20hz = np.asarray([np.nan] * len(t0000)).flatten()
            loadTide_20hz = np.asarray([np.nan] * len(t0000)).flatten()
            solidEarthTide_20hz = np.asarray([np.nan] * len(t0000)).flatten()
            poleTide_20hz = np.asarray([np.nan] * len(t0000)).flatten()

        # ----------------------------------------------------------------------
        # Compute total instrument correction at 20 Hz
        # ----------------------------------------------------------------------

        # only apply distance cog based on communication from ESA EO Support (uso and internal path removed as these are already in the retracker range, and so would otherwise be applied twice)
        totalInstrumentCorrection_20hz = COGCorrection_20hz

        # ----------------------------------------------------------------------
        # Flag missing geophysical corrections
        # ----------------------------------------------------------------------

        # set to 0 if all corrections are present
        missingCorrections = np.zeros((len(dryTroposphericCorrection_20hz), 9))

        for i in range(len(missingCorrections)):
            missingCorrections[i, :] = np.isnan((oceanTide_20hz[i], IBECorrection_HF_20hz[i], IBECorrection_20hz[i], loadTide_20hz[i], poleTide_20hz[i],
                                                solidEarthTide_20hz[i], ionosphericCorrection_20hz[i], wetTroposphericCorrection_20hz[i], dryTroposphericCorrection_20hz[i]))

        # ----------------------------------------------------------------------
        # Compute total geophysical correction
        # ----------------------------------------------------------------------

        geophysicalCorrection_land_20hz = dryTroposphericCorrection_20hz + wetTroposphericCorrection_20hz + \
            ionosphericCorrection_20hz + loadTide_20hz + solidEarthTide_20hz + \
            poleTide_20hz  # default includes land ice corrections only
        geophysicalCorrection_float_20hz = geophysicalCorrection_land_20hz + oceanTide_20hz + \
            IBECorrection_20hz + \
            IBECorrection_HF_20hz  # compute version for floating ice including tide and ibe

        # ----------------------------------------------------------------------
        # Apply corrections
        # ----------------------------------------------------------------------

        retrackedRanges_corrected_land = np.full(
            (len(retrackedRanges), len(retrackedRanges[0])), np.nan)
        retrackedRanges_corrected_float = np.full(
            (len(retrackedRanges), len(retrackedRanges[0])), np.nan)
        elevations_corrected_land = np.full(
            (len(retrackedRanges), len(retrackedRanges[0])), np.nan)
        elevations_corrected_float = np.full(
            (len(retrackedRanges), len(retrackedRanges[0])), np.nan)

        # loop through the 4 retracker methods (maxgrad, tfmra, tcog, tcog_le)
        for i in range(len(retrackedRanges)):

            # ----------------------------------------------------------------------
            # Apply corrections to the retracked range
            # ----------------------------------------------------------------------

            retrackedRanges_corrected_land[i] = retrackedRanges[i] + totalInstrumentCorrection_20hz + \
                geophysicalCorrection_land_20hz  # add corrections to range for land ice

            # compute version for floating ice including tide and ibe corrections
            retrackedRanges_corrected_float[i] = retrackedRanges[i] + \
                totalInstrumentCorrection_20hz + geophysicalCorrection_float_20hz

            # ----------------------------------------------------------------------
            # Compute elevation
            # ----------------------------------------------------------------------

            # subtract retracked range from satellite altitude
            elevations_corrected_land[i] = altitude - \
                retrackedRanges_corrected_land[i]

            # compute floating version
            elevations_corrected_float[i] = altitude - \
                retrackedRanges_corrected_float[i]

        # ----------------------------------------------------------------------
        # Form output dictionary
        # ----------------------------------------------------------------------

        output = {
            # store total corrections
            "Total Instrument Correction (20Hz)": totalInstrumentCorrection_20hz,
            # columns are land and floating ice
            "Geophysical Correction (20Hz)": [geophysicalCorrection_land_20hz, geophysicalCorrection_float_20hz],
            "Missing Corrections Flag": missingCorrections,

            # store corrected ranges - columns are land and floating ice
            "Corrected Retracked Range": {
                "tfmra": [retrackedRanges_corrected_land[0], retrackedRanges_corrected_float[0]],
                "tfmra_LE2": [retrackedRanges_corrected_land[1], retrackedRanges_corrected_float[1]],
                "tfmra_LE3": [retrackedRanges_corrected_land[2], retrackedRanges_corrected_float[2]],


            },
            # store corrected elevations - columns are land and floating ice
            # store corrected elevations - columns are land and floating ice
            "Corrected Elevation": {
                "tfmra": [elevations_corrected_land[0], elevations_corrected_float[0]],
                "tfmra_LE2": [elevations_corrected_land[1], elevations_corrected_float[1]],
                "tfmra_LE3": [elevations_corrected_land[2], elevations_corrected_float[2]],

            }

        }

        # ----------------------------------------------------------------------
        # Return corrected data
        # ----------------------------------------------------------------------

        return(output)

    """
    # Function to apply slope-correction to the L2 S3 track data
    """

    def applySlopeCorrect(self):
        # store corrected elevations
        correctedElevations = self.data["Corrected Elevation"]

        slopeCorrection3D_LE_meanD_maxC = self.data["Slope Correction (LE_meanD_maxC)"]

        slopeCorrection3D_LE2_meanD_maxC = self.data["Slope Correction (LE2_meanD_maxC)"]

        slopeCorrection3D_LE3_meanD_maxC = self.data["Slope Correction (LE3_meanD_maxC)"]

        landElevations = [correctedElevations["tfmra"][0],
                          correctedElevations["tfmra_LE2"][0], correctedElevations["tfmra_LE3"][0], ]
        floatingElevations = [correctedElevations["tfmra"][1],
                              correctedElevations["tfmra_LE2"][1], correctedElevations["tfmra_LE3"][1], ]

        # get the number of records
        numRecords = len(slopeCorrection3D_LE_meanD_maxC)
        numRetracker = len(landElevations)

        # preallocate array for storing slope corrected land elevations at nadir for each retracker
        landElevations_corrected3D_LE_meanD_maxC = np.full(
            (numRetracker, numRecords), np.nan)
        # preallocate array for storing slope corrected floating ice elevations at nadir for each retracker
        floatingElevations_corrected3D_LE_meanD_maxC = np.full(
            (numRetracker, numRecords), np.nan)

        # preallocate array for storing slope corrected land elevations at nadir for each retracker
        landElevations_corrected3D_LE2_meanD_maxC = np.full(
            (numRetracker, numRecords), np.nan)
        # preallocate array for storing slope corrected floating ice elevations at nadir for each retracker
        floatingElevations_corrected3D_LE2_meanD_maxC = np.full(
            (numRetracker, numRecords), np.nan)

        # preallocate array for storing slope corrected land elevations at nadir for each retracker
        landElevations_corrected3D_LE3_meanD_maxC = np.full(
            (numRetracker, numRecords), np.nan)
        # preallocate array for storing slope corrected floating ice elevations at nadir for each retracker
        floatingElevations_corrected3D_LE3_meanD_maxC = np.full(
            (numRetracker, numRecords), np.nan)

        for i in range(numRecords):  # loop through records - consider parallelising

            for j in range(len(landElevations)):
                try:

                    landElevations_corrected3D_LE_meanD_maxC[j,
                                                             i] = landElevations[j][i] + slopeCorrection3D_LE_meanD_maxC[i]
                    floatingElevations_corrected3D_LE_meanD_maxC[j,
                                                                 i] = floatingElevations[j][i] + slopeCorrection3D_LE_meanD_maxC[i]

                    landElevations_corrected3D_LE2_meanD_maxC[j,
                                                              i] = landElevations[j][i] + slopeCorrection3D_LE2_meanD_maxC[i]
                    floatingElevations_corrected3D_LE2_meanD_maxC[j,
                                                                  i] = floatingElevations[j][i] + slopeCorrection3D_LE2_meanD_maxC[i]

                    landElevations_corrected3D_LE3_meanD_maxC[j,
                                                              i] = landElevations[j][i] + slopeCorrection3D_LE3_meanD_maxC[i]
                    floatingElevations_corrected3D_LE3_meanD_maxC[j,
                                                                  i] = floatingElevations[j][i] + slopeCorrection3D_LE3_meanD_maxC[i]
                except:
                    print('nan value for landElevations')

        result = {


            "Slope-Corrected3D_LE_meanD_maxC Elevation": {
                "tfmra": [landElevations_corrected3D_LE_meanD_maxC[0, :], floatingElevations_corrected3D_LE_meanD_maxC[0, :]],
            },


            "Slope-Corrected3D_LE2_meanD_maxC Elevation": {
                "tfmra_LE2": [landElevations_corrected3D_LE2_meanD_maxC[1, :], floatingElevations_corrected3D_LE2_meanD_maxC[1, :]],
            },


            "Slope-Corrected3D_LE3_meanD_maxC Elevation": {
                "tfmra_LE3": [landElevations_corrected3D_LE3_meanD_maxC[2, :], floatingElevations_corrected3D_LE3_meanD_maxC[2, :]],
            },
        }

        # ----------------------------------------------------------------------
        # Return corrected data
        # ----------------------------------------------------------------------

        return(result)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == '__main__':
    start_time = time.perf_counter()

# --------------------------------------------------------------------------------------------------------------
#  Get L2 files for time range
# --------------------------------------------------------------------------------------------------------------
    area = "greenland"
    thisarea = Area(area, load_mask=True)
    thismaskname = thisarea.maskname
    thismask = thisarea.mask
    thismission = Mission('S3A')
    
    outDir = 'E:/'
    files_found = 'E:/postdoc/cpom\cpdata\SATS\RA\S3A\L2\SR_2_LAN_NT\cycle044\S3A_SR_2_LAN____20190505T140913_20190505T145942_20190530T202816_3029_044_210______LN3_O_NT_003.SEN3/enhanced_measurement.nc'
    file_paths = files_found.split(';')

    dem = 'arcticdem_1km'  # arcticdem_1km or  arcticdem_100m_greenland_9_year_dhdt

    for i, fname in enumerate(file_paths):
        Track = S3Track(fname,
                        outDir,
                        maskName=thismaskname,
                        outputProjection="epsg:3413",
                        save=True, dem=dem)  # awi_grn_1km arcticdem_100m arcticdem_100m_greenland
    end_time = time.perf_counter()
    print('Total processing time : {:.2f} s'.format((end_time-start_time)))