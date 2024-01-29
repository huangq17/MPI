# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------

import numpy as np
import matplotlib as mpl
import os
import pathlib
import multiprocessing as mp

from misc import getREMASegment, getLeadingEdge, getHeading

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.chdir(pathlib.Path(__file__).parent.resolve())


"""
Function to extract Dem data within a single S3 altimetry footprint - used for parallel processing
"""


def extractDEM_record(xdem,  # xdem coords
                      ydem,  # ydem coords
                      zdem,  # zdem coords
                      nadirX,  # nadir x coords
                      nadirY,  # nadir y coords
                      waveform,  # waveform at record
                      trackerRange,  # range at record
                      altitude,  # altitude at record
                      heading,
                      ):  # heading of record

    # ----------------------------------------------------------------------
    # Define variables
    # ----------------------------------------------------------------------

    c = 299792458  # speed of light (m/s)
    B = 320000000  # chirp bandwidth used (Hz) from Donlon 2012 table 6
    # 43 for S3 ku-band - this is the bin number that the tracker range in a waveform should correspond to
    referenceBinIndex = 43
    rangeBinSize = c/(2*B)  # compute distance between each bin in meters
    numBins = 128  # the number of bins in each waveform
    acrossTrack_beam = 18000  # define beam radius in meters based on Donlon 2012
    # define alongtrack SAR beam footprint in meters based on S3 user handbook
    alongTrack_beam = 300

    # initialise outputs
    dem_inFootprint_xy = [np.full(1, np.nan), np.full(
        1, np.nan), np.full(1, np.nan)]
    dem_inFootprint_xyz = [np.full(1, np.nan), np.full(
        1, np.nan), np.full(1, np.nan)]
    dem_inFootprint_xyz_LE = [
        np.full(1, np.nan), np.full(1, np.nan), np.full(1, np.nan)]
    dem_inFootprint_xyz_LE2 = [
        np.full(1, np.nan), np.full(1, np.nan), np.full(1, np.nan)]
    dem_inFootprint_xyz_LE3 = [
        np.full(1, np.nan), np.full(1, np.nan), np.full(1, np.nan)]
    # ----------------------------------------------------------------------
    # Create xy beam footprint
    # ----------------------------------------------------------------------

    beamFootprint_xy = mpl.path.Path([(-alongTrack_beam/2, -acrossTrack_beam/2),  # Intialise beam footprint in xy with centre at 0,0
                                      (-alongTrack_beam/2, acrossTrack_beam/2),
                                      (alongTrack_beam/2, acrossTrack_beam/2),
                                      (alongTrack_beam/2, -acrossTrack_beam/2)])
    beamFootprint_xy = beamFootprint_xy.transformed(
        mpl.transforms.Affine2D().rotate_deg(90-heading))  # rotate
    beamFootprint_xy = beamFootprint_xy.transformed(
        mpl.transforms.Affine2D().translate(nadirX, nadirY))  # translate

    # ----------------------------------------------------------------------
    # Subset xdem and ydem to record area for faster running of contains_points()
    # ----------------------------------------------------------------------

    beamFootprint_xy_verts = beamFootprint_xy.vertices.copy()

    bfpX_max = np.max(beamFootprint_xy_verts[:, 0])
    bfpX_min = np.min(beamFootprint_xy_verts[:, 0])
    bfpY_max = np.max(beamFootprint_xy_verts[:, 1])
    bfpY_min = np.min(beamFootprint_xy_verts[:, 1])

    in_x_area = np.logical_and(xdem >= bfpX_min, xdem <= bfpX_max)
    in_y_area = np.logical_and(ydem >= bfpY_min, ydem <= bfpY_max)
    xydemMask = np.flatnonzero(in_x_area & in_y_area)

    xdem = xdem[xydemMask]
    ydem = ydem[xydemMask]
    zdem = zdem[xydemMask]

    # ----------------------------------------------------------------------
    # Identify indices of dem segment coords that are within xy beam footprint (top down)
    # ----------------------------------------------------------------------

    # Get boolean array checking whether dem coords are within the footprint
    beamFootprintMask_xy = beamFootprint_xy.contains_points(
        np.column_stack((xdem, ydem)))
    # get indices of coords that are within xy footprint
    beamFootprintIndices_xy = np.nonzero(beamFootprintMask_xy == 1)

    # ----------------------------------------------------------------------
    # Extract Dem within xy footprint bounds
    # ----------------------------------------------------------------------

    xdem_inFootprint_xy = xdem[beamFootprintIndices_xy]
    ydem_inFootprint_xy = ydem[beamFootprintIndices_xy]
    zdem_inFootprint_xy = zdem[beamFootprintIndices_xy]

    # ----------------------------------------------------------------------
    # Calculate range to window start and end, and range to leading edge start and end
    # ----------------------------------------------------------------------

    # calculate distance from sat to start of range window
    rangeToWindowStart = trackerRange - (referenceBinIndex)*rangeBinSize
    # calculate distance from sat to bottom of range window
    rangeToWindowEnd = trackerRange + (numBins-referenceBinIndex)*rangeBinSize

    _, _, rangeToLEStart, rangeToLEEnd = getLeadingEdge(
        waveform, trackerRange)  # get variables at leading edge start/end

    # ----------------------------------------------------------------------
    # Check whether there are any points in the xy footprint / leading edge variables are nan
    # ----------------------------------------------------------------------

    if len(zdem_inFootprint_xy) == 0:
        beamFootprintIndices_xy = np.nan,
        beamFootprintIndices_xyz = np.nan,
        beamFootprintIndices_xyz_LE = np.nan,
        beamFootprintIndices_xyz_LE2 = np.nan,
        beamFootprintIndices_xyz_LE3 = np.nan,
        pointToSat_dist = np.nan
        return([dem_inFootprint_xy, dem_inFootprint_xyz, dem_inFootprint_xyz_LE, dem_inFootprint_xyz_LE2,
                dem_inFootprint_xyz_LE3, beamFootprintIndices_xy, beamFootprintIndices_xyz, beamFootprintIndices_xyz_LE,
                beamFootprintIndices_xyz_LE2, beamFootprintIndices_xyz_LE3, pointToSat_dist
                ])
    else:
        dem_inFootprint_xy = (xdem_inFootprint_xy,
                              ydem_inFootprint_xy, zdem_inFootprint_xy)

    # ----------------------------------------------------------------------
    # Find distance of points to nadir
    # ----------------------------------------------------------------------

    xyzdem = np.column_stack(
        [xdem_inFootprint_xy, ydem_inFootprint_xy, zdem_inFootprint_xy])  # zip dem coords
    # zip sat locatation coords
    satLoc = np.column_stack([nadirX, nadirY, altitude])
    pointToSat_vec = xyzdem - satLoc  # get vector from sat to each dem point
    # convert to distances
    pointToSat_dist = [np.linalg.norm(vec) for vec in pointToSat_vec]

    # ----------------------------------------------------------------------
    # Identify indices of dem segment coords that are within the full window and leading edge xyz footprints
    # ----------------------------------------------------------------------

    beamFootprintIndices_xyz = np.where(np.logical_and(
        pointToSat_dist >= rangeToWindowStart, pointToSat_dist <= rangeToWindowEnd))

    if len(beamFootprintIndices_xyz[0]) == 0:  # beamFootprintIndices_xyz=[]
        beamFootprintIndices_xyz = np.nan

    if np.isnan(rangeToLEStart).all():
        beamFootprintIndices_xyz_LE = np.nan
        beamFootprintIndices_xyz_LE2 = np.nan
        beamFootprintIndices_xyz_LE3 = np.nan
    else:
        beamFootprintIndices_xyz_LE = np.where(np.logical_and(
            pointToSat_dist >= rangeToLEStart[0], pointToSat_dist <= rangeToLEEnd[0]))
        if len(rangeToLEStart) > 1:
            try:
                beamFootprintIndices_xyz_LE2 = np.where(np.logical_and(
                    pointToSat_dist >= rangeToLEStart[1], pointToSat_dist <= rangeToLEEnd[1]))

            except:
                beamFootprintIndices_xyz_LE2 = np.nan

        if len(rangeToLEStart) > 2:
            try:
                beamFootprintIndices_xyz_LE3 = np.where(np.logical_and(
                    pointToSat_dist >= rangeToLEStart[2], pointToSat_dist <= rangeToLEEnd[2]))

            except:
                beamFootprintIndices_xyz_LE3 = np.nan

    # ----------------------------------------------------------------------
    # Extract Dem within full window and leading edge xyz footprints
    # ----------------------------------------------------------------------

    if np.size(beamFootprintIndices_xyz) > 0 and ~np.isnan(beamFootprintIndices_xyz).all():
        xdem_inFootprint_xyz = np.asarray(
            xdem_inFootprint_xy[beamFootprintIndices_xyz])
        ydem_inFootprint_xyz = np.asarray(
            ydem_inFootprint_xy[beamFootprintIndices_xyz])
        zdem_inFootprint_xyz = np.asarray(
            zdem_inFootprint_xy[beamFootprintIndices_xyz])

        dem_inFootprint_xyz = (xdem_inFootprint_xyz,
                               ydem_inFootprint_xyz, zdem_inFootprint_xyz)

    if np.size(beamFootprintIndices_xyz_LE) > 0 and ~np.isnan(beamFootprintIndices_xyz_LE).all():
        xdem_inFootprint_xyz_LE = np.asarray(
            xdem_inFootprint_xy[beamFootprintIndices_xyz_LE])
        ydem_inFootprint_xyz_LE = np.asarray(
            ydem_inFootprint_xy[beamFootprintIndices_xyz_LE])
        zdem_inFootprint_xyz_LE = np.asarray(
            zdem_inFootprint_xy[beamFootprintIndices_xyz_LE])

        dem_inFootprint_xyz_LE = (
            xdem_inFootprint_xyz_LE, ydem_inFootprint_xyz_LE, zdem_inFootprint_xyz_LE)

    if len(rangeToLEStart) > 1 and ~np.isnan(beamFootprintIndices_xyz_LE2).all():
        xdem_inFootprint_xyz_LE2 = np.asarray(
            xdem_inFootprint_xy[beamFootprintIndices_xyz_LE2])
        ydem_inFootprint_xyz_LE2 = np.asarray(
            ydem_inFootprint_xy[beamFootprintIndices_xyz_LE2])
        zdem_inFootprint_xyz_LE2 = np.asarray(
            zdem_inFootprint_xy[beamFootprintIndices_xyz_LE2])
        dem_inFootprint_xyz_LE2 = (
            xdem_inFootprint_xyz_LE2, ydem_inFootprint_xyz_LE2, zdem_inFootprint_xyz_LE2)
    else:
        xdem_inFootprint_xyz_LE2 = np.nan
        ydem_inFootprint_xyz_LE2 = np.nan
        zdem_inFootprint_xyz_LE2 = np.nan
        beamFootprintIndices_xyz_LE2 = np.nan
        dem_inFootprint_xyz_LE2 = (
            xdem_inFootprint_xyz_LE2, ydem_inFootprint_xyz_LE2, zdem_inFootprint_xyz_LE2)

    if len(rangeToLEStart) > 2 and ~np.isnan(beamFootprintIndices_xyz_LE3).all():
        xdem_inFootprint_xyz_LE3 = np.asarray(
            xdem_inFootprint_xy[beamFootprintIndices_xyz_LE3])
        ydem_inFootprint_xyz_LE3 = np.asarray(
            ydem_inFootprint_xy[beamFootprintIndices_xyz_LE3])
        zdem_inFootprint_xyz_LE3 = np.asarray(
            zdem_inFootprint_xy[beamFootprintIndices_xyz_LE3])
        dem_inFootprint_xyz_LE3 = (
            xdem_inFootprint_xyz_LE3, ydem_inFootprint_xyz_LE3, zdem_inFootprint_xyz_LE3)
    else:
        xdem_inFootprint_xyz_LE3 = np.nan
        ydem_inFootprint_xyz_LE3 = np.nan
        zdem_inFootprint_xyz_LE3 = np.nan
        beamFootprintIndices_xyz_LE3 = np.nan
        dem_inFootprint_xyz_LE3 = (
            xdem_inFootprint_xyz_LE3, ydem_inFootprint_xyz_LE3, zdem_inFootprint_xyz_LE3)

    # ----------------------------------------------------------------------
    # Return results
    # ----------------------------------------------------------------------

    return([dem_inFootprint_xy, dem_inFootprint_xyz, dem_inFootprint_xyz_LE, dem_inFootprint_xyz_LE2, dem_inFootprint_xyz_LE3, beamFootprintIndices_xy, beamFootprintIndices_xyz, beamFootprintIndices_xyz_LE, beamFootprintIndices_xyz_LE2, beamFootprintIndices_xyz_LE3, pointToSat_dist])


"""
Function to extract Dem data within footprints in S3 track
"""


def extractDEM(waveform, nadirX, nadirY, trackerRange, altitude, dem):

    # ----------------------------------------------------------------------
    # Define variables
    # ----------------------------------------------------------------------

    acrossTrack_beam = 18000  # define beam radius in meters based on Donlon 2012
    demPosting = 100  # define dem posting
    numRecords = len(nadirX)  # obtain the number of records

    # ----------------------------------------------------------------------
    # Extract Dem data about the track
    # ----------------------------------------------------------------------

    # get the bounds about the track, adjusted for across track beam width and the dem posting
    x_min = min(nadirX) - (acrossTrack_beam/2 + demPosting)
    x_max = max(nadirX) + (acrossTrack_beam/2 + demPosting)
    y_min = min(nadirY) - (acrossTrack_beam/2 + demPosting)
    y_max = max(nadirY) + (acrossTrack_beam/2 + demPosting)

    # extract the REMA data within the bounds
    xdem, ydem, zdem = getREMASegment(
        [(x_min, y_min), (x_max, y_max)], dem, gridXY=True, flatten=True)

    # ----------------------------------------------------------------------
    # Get headings for each record
    # ----------------------------------------------------------------------

    heading = getHeading(nadirX, nadirY)

    # ----------------------------------------------------------------------
    # Compute beam footprint for each record using parallel computing
    # ----------------------------------------------------------------------

    output = []

    # """
    pool = mp.Pool()
    output = pool.starmap(extractDEM_record, [(xdem, ydem, zdem, nadirX[record], nadirY[record], waveform[record],
                          trackerRange[record], altitude[record], heading[record]) for record in range(numRecords)])
    pool.close
    # """

    dem_inFootprint_xy = [i[0] for i in output]
    dem_inFootprint_xyz = [i[1] for i in output]
    dem_inFootprint_xyz_LE = [i[2] for i in output]
    dem_inFootprint_xyz_LE2 = [i[3] for i in output]
    dem_inFootprint_xyz_LE3 = [i[4] for i in output]

    beamFootprintIndices_xy = [i[5] for i in output]
    beamFootprintIndices_xyz = [i[6] for i in output]
    beamFootprintIndices_xyz_LE = [i[7] for i in output]
    beamFootprintIndices_xyz_LE2 = [i[8] for i in output]
    beamFootprintIndices_xyz_LE3 = [i[9] for i in output]

    pointToSat_dist = [i[10] for i in output]

    # ----------------------------------------------------------------------
    # Form output dictionary
    # ----------------------------------------------------------------------

    outputDict = {"Dem in Footprint (X,Y)": dem_inFootprint_xy,
                  "Dem in Footprint (X,Y,Z)": dem_inFootprint_xyz,
                  "Dem in Footprint (X,Y,Z) (Leading Edge)": dem_inFootprint_xyz_LE,
                  "Dem in Footprint (X,Y,Z) (Leading Edge2)": dem_inFootprint_xyz_LE2,
                  "Dem in Footprint (X,Y,Z) (Leading Edge3)": dem_inFootprint_xyz_LE3,

                  "beamFootprintIndices_xy": beamFootprintIndices_xy,
                  "beamFootprintIndices_xyz": beamFootprintIndices_xyz,
                  "beamFootprintIndices_xyz_LE": beamFootprintIndices_xyz_LE,
                  "beamFootprintIndices_xyz_LE2": beamFootprintIndices_xyz_LE2,
                  "beamFootprintIndices_xyz_LE3": beamFootprintIndices_xyz_LE3,

                  "pointToSat_dist": pointToSat_dist


                  }
    return outputDict
