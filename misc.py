# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------

from netCDF4 import Dataset
import pyproj as proj
import pandas as pd
import matplotlib as mpl
import numpy as np
import PIL
from scipy.stats import binned_statistic_2d
from scipy.signal import savgol_filter
from scipy import interpolate
import matplotlib.pyplot as plt
import shapefile
from scipy.io import loadmat
import os
import pathlib
from dems import Dem
from masks import Masks, Mask
import sys
from shapely.geometry.polygon import Polygon
from scipy.signal import find_peaks
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter

os.chdir(pathlib.Path(__file__).parent.resolve())


"""
Function to get S3 track headings (requires at least 2 records)
"""


def getHeading(nadirX, nadirY):

    numRecords = len(nadirX)

    heading = np.full(numRecords, np.nan)
    for record in range(numRecords):

        if record != numRecords-1:  # for all but the final record compute delta x and delta y to next record
            dx = nadirX[record+1] - nadirX[record]
            dy = nadirY[record+1] - nadirY[record]

        elif record == numRecords-1:  # else, copy the preceding dx and dy values
            dx = nadirX[record] - nadirX[record-1]
            dy = nadirY[record] - nadirY[record-1]

        # compute heading inner angle
        headingInner = np.rad2deg(np.arctan(dx/dy))

        if dy < 0:  # if heading vector orientated into Q2 or Q3 then add 180 degrees
            heading[record] = headingInner + 180

        elif dy > 0 and dx < 0:  # if heading vector orientated into Q4 then add 360 degrees to make positive
            heading[record] = headingInner + 360

        else:
            heading[record] = headingInner

    return heading


"""
Function to get REMA data about track 
    Bounds takes bottom left and top right coordinates in antartic polar stereographic [(minx,miny),(maxx,maxy)]
"""


def getREMASegment(bounds, dem, resolution=100, gridXY=True, flatten=False):

    # ----------------------------------------------------------------------
    # Load DEM
    # ----------------------------------------------------------------------
    # awi_ant_1km awi_grn_1km arcticdem_100m arcticdem_100m_greenland
    dem = Dem(dem)
    xdem = dem.xdem
    ydem = dem.ydem
    zdem = dem.zdem

    # ----------------------------------------------------------------------
    # Get coord bounds as index bounds
    # ----------------------------------------------------------------------

    minX_ind = (np.absolute(bounds[0][0]-xdem)).argmin()
    minY_ind = (np.absolute(bounds[0][1]-ydem)).argmin()
    maxX_ind = (np.absolute(bounds[1][0]-xdem)).argmin()
    maxY_ind = (np.absolute(bounds[1][1]-ydem)).argmin()

    # ----------------------------------------------------------------------
    # Crop x,y,z
    # ----------------------------------------------------------------------

    zdem = zdem[maxY_ind:minY_ind, minX_ind:maxX_ind]
    xdem = xdem[minX_ind:maxX_ind]
    ydem = ydem[maxY_ind:minY_ind]

    # ----------------------------------------------------------------------
    # Set void data to nan
    # ----------------------------------------------------------------------

    # void elevation data given as -9999 in REMA raster - set to nan
    voidData = np.where(zdem <= -9999)
    if np.any(voidData):
        zdem[voidData] = np.nan

    # ----------------------------------------------------------------------
    # Convert x,y vectors to grids if requested
    # ----------------------------------------------------------------------

    if gridXY == True:
        xdem, ydem = np.meshgrid(xdem, ydem)

        # ----------------------------------------------------------------------
        # Set x,y to nan where z is nan
        # ----------------------------------------------------------------------
        # create mask where true when zdem is nan, false otherwise
        nan_zdem = ~np.isnan(zdem)

        xdem[~nan_zdem] = np.nan
        ydem[~nan_zdem] = np.nan

    # ----------------------------------------------------------------------
    # Return, flattened if requested
    # ----------------------------------------------------------------------
    if flatten == False:
        return(xdem, ydem, zdem)
    else:
        return(xdem.flatten(), ydem.flatten(), zdem.flatten())


"""
Function to get the leading edges in an S3 waveform
"""


def getLeadingEdge(waveform, trackerRange, returnExtraVariables=False):

    # ----------------------------------------------------------------------
    # Define variables
    # ----------------------------------------------------------------------

    c = 299792458  # speed of light (m/s)
    B = 320000000  # chirp bandwidth used (Hz) from Donlon 2012 table 6
    rangeBinSize = c/(2*B)  # compute distance between each bin in meters
    # 43 for S3 ku-band - this is the bin number that the tracker range in a waveform should correspond to
    referenceBinIndex = 43
    wfOversamplingFactor = 100  # waveform oversampling factor
    # if mean amplitude in noise bins exceeds threshold then reject waveform
    noiseThreshold = 0.3
    # power must be this much greater than thermal noise to be identified as leading edge
    leThreshold_id = 0.05

    leIndex_end = []
    leIndex_start = []
    leRange_start = np.full(1, np.nan)
    leRange_end = np.full(1, np.nan)
    wfNormSmoothInterpolated = np.nan
    wfNormSmoothInterpolated_d1 = np.nan
    wfNoiseMean = np.nan
    wfNormInterpolated = np.nan

    # Infinite loop to allow for conditional breaking and avoid bloated return statements - maybe a dumb idea?
    while True:

        # ----------------------------------------------------------------------
        # Normalise waveform
        # ----------------------------------------------------------------------

        wfNorm = waveform/max(waveform)  # normalise waveform

        # ----------------------------------------------------------------------
        # Smooth waveform
        # ----------------------------------------------------------------------

        wfNormSmooth = savgol_filter(wfNorm, 9, 3)
        np.insert(wfNormSmooth, 0, np.nan)  # set end values as nan
        np.insert(wfNormSmooth, len(wfNorm), np.nan)

        # ----------------------------------------------------------------------
        # Compute thermal noise
        # ----------------------------------------------------------------------

        # sort power values of unsmoothed waveform in ascending order
        wfSorted = np.sort(wfNorm)
        # estimate noise based on lowest 6 samples
        wfNoiseMean = np.mean(wfSorted[0:6])

        # ----------------------------------------------------------------------
        # Quality check 1 - check if mean noise above predefined threshold
        # ----------------------------------------------------------------------

        if wfNoiseMean > noiseThreshold:
            break

        # ----------------------------------------------------------------------
        # Oversample using spline
        # ----------------------------------------------------------------------

        # compute bin interval for oversampled waveform
        wfOversamplingInterval = 1/wfOversamplingFactor
        # create oversampled waveform bin indices
        wfBinNumberIndices = np.arange(
            0,  len(waveform), wfOversamplingInterval)
        wfNormSmoothInterpolated = interpolate.splev(wfBinNumberIndices, interpolate.splrep(range(len(
            waveform)), wfNormSmooth))  # compute spline and interpolated values of smoothed waveform at bin numbers
        # compute spline and interpolated values of non smoothed waveform at bin numbers
        wfNormInterpolated = interpolate.splev(
            wfBinNumberIndices, interpolate.splrep(range(len(waveform)), wfNorm, k=1))

        # ----------------------------------------------------------------------
        # Compute derivatives
        # ----------------------------------------------------------------------

        # compute first derivative of smoothed waveform
        wfNormSmoothInterpolated_d1 = np.gradient(
            wfNormSmoothInterpolated, wfOversamplingInterval)

        # ----------------------------------------------------------------------
        # Loop through indices until no more peak candidates found or index a bin width away from end is reached
        # ----------------------------------------------------------------------

        # detect peaks after the first (100) gate(s)
        peaks, _ = find_peaks(
            wfNormInterpolated[wfOversamplingFactor:], height=0.2, distance=1000, prominence=0.2)
        peaks = peaks+wfOversamplingFactor
        if len(peaks) == 0:
            break

        # find next leading edge candidates (at least 1 original bin width from last) which are above the threshold
        leIndices = np.where((wfNormInterpolated > (wfNoiseMean + leThreshold_id))
                             & (wfBinNumberIndices > wfBinNumberIndices[wfOversamplingFactor]))

        # ----------------------------------------------------------------------
        # Quality check 2 - check if no samples are sufficiently above the noise or large enough leading edge
        # ----------------------------------------------------------------------

        if np.size(leIndices) == 0:
            break

        else:  # else take the first index found
            first_leIndex = leIndices[0][0]

        # select the minimum as opposed to the first gate
        first_min = np.argmin(wfNormInterpolated[first_leIndex:peaks[0]])

        wfAboveNoise = wfNormInterpolated - wfNoiseMean

        trough = argrelextrema(wfAboveNoise, np.less)

        for j in range(len(peaks)-1):
            leading_edge_start = np.where((trough > peaks[j]) & (
                trough < peaks[j+1]))  # find the trough between peaks
            # leading_edge_start[1] is the index for the peaks, e.g., the 2nd or 3rd
            trough_Power = wfNormInterpolated[trough[0][leading_edge_start[1]]]
            try:
                trough_index = leading_edge_start[1][np.argmin(trough_Power)]
            except:
                continue
            leIndex_start.append(trough[0][trough_index])

        # use the gate with minimum power since the first leIndex
        leIndex_start.insert(0, first_leIndex+first_min)
        leIndex_start = np.array(leIndex_start)/wfOversamplingFactor
        leIndex_end = peaks/wfOversamplingFactor
        # ----------------------------------------------------------------------
        # Get the range from the sat to the leading edge
        # ----------------------------------------------------------------------

        if len(leIndex_start) > 0:

            leRange_start = trackerRange - \
                (referenceBinIndex - leIndex_start)*rangeBinSize
            leRange_end = trackerRange + \
                (leIndex_end - referenceBinIndex)*rangeBinSize
        break

    if returnExtraVariables == False:
        return(leIndex_start, leIndex_end, leRange_start, leRange_end)
    else:
        return(leIndex_start, leIndex_end, leRange_start, leRange_end, wfNormSmoothInterpolated, wfNormSmoothInterpolated_d1, wfNoiseMean, wfNormInterpolated)


"""
Manual nearest-neighbour interpolation - (due to bad results with scipy etc)
"""


def getZ(x, y, xdem, ydem, zdem, fill):

    if np.shape(xdem) == np.shape(zdem):
        xdem = xdem[:, 0]
        ydem = ydem[0, :]

    z = np.full(len(x), fill)
    for i in range(len(z)):

        if np.isnan(x[i]) or np.isnan(y[i]):
            continue

        xind = np.nanargmin(np.abs(xdem - x[i]))
        yind = np.nanargmin(np.abs(ydem - y[i]))

        if np.isnan(zdem[yind, xind]):
            continue

        z[i] = zdem[yind, xind]

    return z
