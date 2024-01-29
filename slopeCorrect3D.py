import numpy as np
from extractDEM import extractDEM
from sklearn.cluster import AgglomerativeClustering


def slopeCorrect3D(waveform, nadirX, nadirY, z, altitude, ESA_pocaX, ESA_pocaY, ESA_pocaZ, dem, dem_inFootprint=None):

    # ----------------------------------------------------------------------
    # Get REMA data within footprints if not provided
    # ----------------------------------------------------------------------

    if dem_inFootprint == None:
        REMADict = extractDEM(waveform, nadirX, nadirY, z, altitude, dem)

    dem_inFootprint = REMADict["Dem in Footprint (X,Y)"]
    dem_inFootprint_xyz = REMADict["Dem in Footprint (X,Y,Z)"]
    dem_inFootprint_xyz_LE = REMADict["Dem in Footprint (X,Y,Z) (Leading Edge)"]
    dem_inFootprint_xyz_LE2 = REMADict["Dem in Footprint (X,Y,Z) (Leading Edge2)"]
    dem_inFootprint_xyz_LE3 = REMADict["Dem in Footprint (X,Y,Z) (Leading Edge3)"]

    beamFootprintIndices_xy = REMADict["beamFootprintIndices_xy"]
    beamFootprintIndices_xyz = REMADict["beamFootprintIndices_xyz"]
    beamFootprintIndices_xyz_LE = REMADict["beamFootprintIndices_xyz_LE"]
    beamFootprintIndices_xyz_LE2 = REMADict["beamFootprintIndices_xyz_LE2"]
    beamFootprintIndices_xyz_LE3 = REMADict["beamFootprintIndices_xyz_LE3"]

    pointToSat_dist = REMADict["pointToSat_dist"]

    # ----------------------------------------------------------------------
    # Slope correct records
    # ----------------------------------------------------------------------

    numRecords = len(nadirX)

    xdem_poca_xyz_LE_meanD_maxC = np.full(numRecords, np.nan)
    ydem_poca_xyz_LE_meanD_maxC = np.full(numRecords, np.nan)
    zdem_poca_xyz_LE_meanD_maxC = np.full(numRecords, np.nan)
    pocaRange_xyz_LE_meanD_maxC = np.full(numRecords, np.nan)
    slopeCorrection_xyz_LE_meanD_maxC = np.full(numRecords, np.nan)

    xdem_poca_xyz_LE2_meanD_maxC = np.full(numRecords, np.nan)
    ydem_poca_xyz_LE2_meanD_maxC = np.full(numRecords, np.nan)
    zdem_poca_xyz_LE2_meanD_maxC = np.full(numRecords, np.nan)
    pocaRange_xyz_LE2_meanD_maxC = np.full(numRecords, np.nan)
    slopeCorrection_xyz_LE2_meanD_maxC = np.full(numRecords, np.nan)

    xdem_poca_xyz_LE3_meanD_maxC = np.full(numRecords, np.nan)
    ydem_poca_xyz_LE3_meanD_maxC = np.full(numRecords, np.nan)
    zdem_poca_xyz_LE3_meanD_maxC = np.full(numRecords, np.nan)
    pocaRange_xyz_LE3_meanD_maxC = np.full(numRecords, np.nan)
    slopeCorrection_xyz_LE3_meanD_maxC = np.full(numRecords, np.nan)

    dem_inFootprint_xyz_LE_c0 = [[np.nan] *
                                 3 for i in range(len(dem_inFootprint_xyz_LE))]
    dem_inFootprint_xyz_LE_c1 = [[np.nan] *
                                 3 for i in range(len(dem_inFootprint_xyz_LE))]
    dem_inFootprint_xyz_LE_c2 = [[np.nan] *
                                 3 for i in range(len(dem_inFootprint_xyz_LE))]
    dem_inFootprint_xyz_LE_c3 = [[np.nan] *
                                 3 for i in range(len(dem_inFootprint_xyz_LE))]

    dem_inFootprint_xyz_LE2_c0 = [
        [np.nan] * 3 for i in range(len(dem_inFootprint_xyz_LE2))]
    dem_inFootprint_xyz_LE2_c1 = [
        [np.nan] * 3 for i in range(len(dem_inFootprint_xyz_LE2))]
    dem_inFootprint_xyz_LE2_c2 = [
        [np.nan] * 3 for i in range(len(dem_inFootprint_xyz_LE2))]
    dem_inFootprint_xyz_LE2_c3 = [
        [np.nan] * 3 for i in range(len(dem_inFootprint_xyz_LE2))]

    dem_inFootprint_xyz_LE3_c0 = [
        [np.nan] * 3 for i in range(len(dem_inFootprint_xyz_LE3))]
    dem_inFootprint_xyz_LE3_c1 = [
        [np.nan] * 3 for i in range(len(dem_inFootprint_xyz_LE3))]
    dem_inFootprint_xyz_LE3_c2 = [
        [np.nan] * 3 for i in range(len(dem_inFootprint_xyz_LE3))]
    dem_inFootprint_xyz_LE3_c3 = [
        [np.nan] * 3 for i in range(len(dem_inFootprint_xyz_LE3))]

    ESApocaRange = np.full(numRecords, np.nan)
    xyzdem_ESA = np.column_stack(
        [ESA_pocaX, ESA_pocaY, ESA_pocaZ])  # zip dem coords
    # zip sat locatation coords
    satLoc = np.column_stack([nadirX, nadirY, altitude])
    # get vector from sat to each dem point
    pointToSat_vec_ESA = xyzdem_ESA - satLoc
    # convert to distances
    ESApocaRange = [np.linalg.norm(vec) for vec in pointToSat_vec_ESA]

    cluster_ind1 = np.full(numRecords, np.nan)
    cluster_ind2 = np.full(numRecords, np.nan)
    cluster_ind3 = np.full(numRecords, np.nan)

    for i in range(numRecords):  # loop through records
        # ----------------------------------------------------------------------
        # Find POCA
        # ----------------------------------------------------------------------

        # find range to, and indices of, dem pixel closest to satellite
        dem_Range = np.asarray(pointToSat_dist[i])

        # ----------------------------------------------------------------------
        # Find POCA for LEPTA
        # ----------------------------------------------------------------------
        if np.isnan(beamFootprintIndices_xyz_LE[i]).all():

            xdem_poca_xyz_LE_meanD_maxC[i] = np.nan
            ydem_poca_xyz_LE_meanD_maxC[i] = np.nan
            zdem_poca_xyz_LE_meanD_maxC[i] = np.nan
            pocaRange_xyz_LE_meanD_maxC[i] = np.nan
            slopeCorrection_xyz_LE_meanD_maxC[i] = np.nan
        else:

            # ----------------------------------------------------------------------
            # Refined LEPTA for 1st leading edge
            # ----------------------------------------------------------------------

            cluster_indices1 = np.asarray(
                beamFootprintIndices_xyz_LE[i]).reshape(-1, 1)

            if len(cluster_indices1) > 1:
                clustering = AgglomerativeClustering(
                    n_clusters=None, distance_threshold=10, linkage='single').fit(cluster_indices1)
                # ----------------------------------------------------------------------
                # find the cluster with the most points \
                # ----------------------------------------------------------------------
                counts = np.bincount(clustering.labels_)  # find the mode
                # store the cluster number
                cluster_ind1[i] = len(np.unique(clustering.labels_))

                beamFootprintIndices_xyz_LE_maxC = np.where(
                    clustering.labels_ == np.argmax(counts))
                # ----------------------------------------------------------------------
                # find the cluster closest to satellite
                # ----------------------------------------------------------------------
                cluster_class_range = np.full(
                    len(np.unique(clustering.labels_)), np.nan)
                for j, k in enumerate(np.unique(clustering.labels_)):
                    cluster_class = np.where(clustering.labels_ == k)
                    # cluster_class_range[i]=np.nanmean(dem_Range(beamFootprintIndices_xyz_LE2[i][0][cluster_class]))
                    cluster_class_range[j] = np.nanmean(
                        dem_Range[cluster_indices1[cluster_class]])
                    if j == 0:
                        c0 = beamFootprintIndices_xyz_LE[i][0][np.where(
                            clustering.labels_ == k)]
                        dem_inFootprint_xyz_LE_c0[i] = [
                            dem_inFootprint[i][0][c0], dem_inFootprint[i][1][c0], dem_inFootprint[i][2][c0]]
                    if j == 1:
                        c1 = beamFootprintIndices_xyz_LE[i][0][np.where(
                            clustering.labels_ == k)]
                        dem_inFootprint_xyz_LE_c1[i] = [
                            dem_inFootprint[i][0][c1], dem_inFootprint[i][1][c1], dem_inFootprint[i][2][c1]]
                    if j == 2:
                        c2 = beamFootprintIndices_xyz_LE[i][0][np.where(
                            clustering.labels_ == k)]
                        dem_inFootprint_xyz_LE_c2[i] = [
                            dem_inFootprint[i][0][c2], dem_inFootprint[i][1][c2], dem_inFootprint[i][2][c2]]
                    if j == 3:
                        c3 = beamFootprintIndices_xyz_LE[i][0][np.where(
                            clustering.labels_ == k)]
                        dem_inFootprint_xyz_LE_c3[i] = [
                            dem_inFootprint[i][0][c3], dem_inFootprint[i][1][c3], dem_inFootprint[i][2][c3]]

            else:
                beamFootprintIndices_xyz_LE_maxC = 0  # select the only indice

                cluster_ind1[i] = 1

            # ----------------------------------------------------------------------
            # choose the cluster with the most points and then average them as the poca
            # ----------------------------------------------------------------------
            xdem_poca_xyz_LE_meanD_maxC[i] = np.nanmean(
                dem_inFootprint_xyz_LE[i][0][beamFootprintIndices_xyz_LE_maxC])
            ydem_poca_xyz_LE_meanD_maxC[i] = np.nanmean(
                dem_inFootprint_xyz_LE[i][1][beamFootprintIndices_xyz_LE_maxC])
            zdem_poca_xyz_LE_meanD_maxC[i] = np.nanmean(
                dem_inFootprint_xyz_LE[i][2][beamFootprintIndices_xyz_LE_maxC])
            POCALoc_LE_meanD_maxC = np.column_stack(
                [xdem_poca_xyz_LE_meanD_maxC[i], ydem_poca_xyz_LE_meanD_maxC[i], zdem_poca_xyz_LE_meanD_maxC[i]])  # zip poca coords
            POCAToSat_meanD_maxC = POCALoc_LE_meanD_maxC - \
                satLoc[i]  # get vector from sat to poca
            pocaRange_xyz_LE_meanD_maxC[i] = np.linalg.norm(
                POCAToSat_meanD_maxC)  # convert to range
            # compute slope correction to apply to assumed nadir altimeter elevation to relocate to poca
            slopeCorrection_xyz_LE_meanD_maxC[i] = pocaRange_xyz_LE_meanD_maxC[i] + \
                zdem_poca_xyz_LE_meanD_maxC[i] - altitude[i]
            #
        # ----------------------------------------------------------------------
        # LEPTA for LE2
        # ----------------------------------------------------------------------
        if np.isnan(beamFootprintIndices_xyz_LE2[i]).all():

            xdem_poca_xyz_LE2_meanD_maxC[i] = np.nan
            ydem_poca_xyz_LE2_meanD_maxC[i] = np.nan
            zdem_poca_xyz_LE2_meanD_maxC[i] = np.nan
            pocaRange_xyz_LE2_meanD_maxC[i] = np.nan
            slopeCorrection_xyz_LE2_meanD_maxC[i] = np.nan
        else:

            # ----------------------------------------------------------------------
            # Refined LEPTA for LE2, two versions
            # ----------------------------------------------------------------------

            cluster_indices2 = np.asarray(
                beamFootprintIndices_xyz_LE2[i]).reshape(-1, 1)

            if len(cluster_indices2) > 1:
                clustering = AgglomerativeClustering(
                    n_clusters=None, distance_threshold=10, linkage='single').fit(cluster_indices2)
                # ----------------------------------------------------------------------
                # find the cluster with the most points \
                # ----------------------------------------------------------------------
                counts = np.bincount(clustering.labels_)  # find the mode
                cluster_ind2[i] = len(np.unique(clustering.labels_))
                beamFootprintIndices_xyz_LE2_maxC = np.where(
                    clustering.labels_ == np.argmax(counts))
                # ----------------------------------------------------------------------
                # find the cluster closest to satellite
                # ----------------------------------------------------------------------
                cluster_class_range = np.full(
                    len(np.unique(clustering.labels_)), np.nan)
                for j, k in enumerate(np.unique(clustering.labels_)):
                    cluster_class = np.where(clustering.labels_ == k)
                    # cluster_class_range[i]=np.nanmean(dem_Range(beamFootprintIndices_xyz_LE2[i][0][cluster_class]))
                    cluster_class_range[j] = np.nanmean(
                        dem_Range[cluster_indices2[cluster_class]])
                    if j == 0:
                        # beamFootprintIndices_xyz_LE2_c0[i]=np.where(clustering.labels_==k)
                        c0 = beamFootprintIndices_xyz_LE2[i][0][np.where(
                            clustering.labels_ == k)]
                        dem_inFootprint_xyz_LE2_c0[i] = [
                            dem_inFootprint[i][0][c0], dem_inFootprint[i][1][c0], dem_inFootprint[i][2][c0]]
                    if j == 1:
                        c1 = beamFootprintIndices_xyz_LE2[i][0][np.where(
                            clustering.labels_ == k)]
                        dem_inFootprint_xyz_LE2_c1[i] = [
                            dem_inFootprint[i][0][c1], dem_inFootprint[i][1][c1], dem_inFootprint[i][2][c1]]
                    if j == 2:
                        c2 = beamFootprintIndices_xyz_LE2[i][0][np.where(
                            clustering.labels_ == k)]
                        dem_inFootprint_xyz_LE2_c2[i] = [
                            dem_inFootprint[i][0][c2], dem_inFootprint[i][1][c2], dem_inFootprint[i][2][c2]]
                    if j == 3:
                        c3 = beamFootprintIndices_xyz_LE2[i][0][np.where(
                            clustering.labels_ == k)]
                        dem_inFootprint_xyz_LE2_c3[i] = [
                            dem_inFootprint[i][0][c3], dem_inFootprint[i][1][c3], dem_inFootprint[i][2][c3]]

            else:
                beamFootprintIndices_xyz_LE2_maxC = 0  # select the only indice
                cluster_ind2[i] = 1

           # ----------------------------------------------------------------------
           # choose the cluster with the most points and then average them as the poca
           # ----------------------------------------------------------------------
            xdem_poca_xyz_LE2_meanD_maxC[i] = np.nanmean(
                dem_inFootprint_xyz_LE2[i][0][beamFootprintIndices_xyz_LE2_maxC])
            ydem_poca_xyz_LE2_meanD_maxC[i] = np.nanmean(
                dem_inFootprint_xyz_LE2[i][1][beamFootprintIndices_xyz_LE2_maxC])
            zdem_poca_xyz_LE2_meanD_maxC[i] = np.nanmean(
                dem_inFootprint_xyz_LE2[i][2][beamFootprintIndices_xyz_LE2_maxC])
            POCALoc_LE2_meanD_maxC = np.column_stack(
                [xdem_poca_xyz_LE2_meanD_maxC[i], ydem_poca_xyz_LE2_meanD_maxC[i], zdem_poca_xyz_LE2_meanD_maxC[i]])  # zip poca coords
            POCAToSat2_meanD_maxC = POCALoc_LE2_meanD_maxC - \
                satLoc[i]  # get vector from sat to poca
            pocaRange_xyz_LE2_meanD_maxC[i] = np.linalg.norm(
                POCAToSat2_meanD_maxC)  # convert to range
            # compute slope correction to apply to assumed nadir altimeter elevation to relocate to poca
            slopeCorrection_xyz_LE2_meanD_maxC[i] = pocaRange_xyz_LE2_meanD_maxC[i] + \
                zdem_poca_xyz_LE2_meanD_maxC[i] - altitude[i]

        # ----------------------------------------------------------------------
        # Find POCA for LE3 LEPTA
        # ----------------------------------------------------------------------
        if np.isnan(beamFootprintIndices_xyz_LE3[i]).all():

            xdem_poca_xyz_LE3_meanD_maxC[i] = np.nan
            ydem_poca_xyz_LE3_meanD_maxC[i] = np.nan
            zdem_poca_xyz_LE3_meanD_maxC[i] = np.nan
            pocaRange_xyz_LE3_meanD_maxC[i] = np.nan
            slopeCorrection_xyz_LE3_meanD_maxC[i] = np.nan
        else:

            # ----------------------------------------------------------------------
            # Refined LEPTA for LE3
            # ----------------------------------------------------------------------

            cluster_indices3 = np.asarray(
                beamFootprintIndices_xyz_LE3[i]).reshape(-1, 1)

            if len(cluster_indices3) > 1:
                clustering = AgglomerativeClustering(
                    n_clusters=None, distance_threshold=10, linkage='single').fit(cluster_indices3)
                # ----------------------------------------------------------------------
                # find the cluster with the most points \
                # ----------------------------------------------------------------------
                counts = np.bincount(clustering.labels_)  # find the mode
                cluster_ind3[i] = len(np.unique(clustering.labels_))
                beamFootprintIndices_xyz_LE3_maxC = np.where(
                    clustering.labels_ == np.argmax(counts))

                # ----------------------------------------------------------------------
                # find the cluster closest to satellite
                # ----------------------------------------------------------------------
                cluster_class_range = np.full(
                    len(np.unique(clustering.labels_)), np.nan)
                for j, k in enumerate(np.unique(clustering.labels_)):
                    cluster_class = np.where(clustering.labels_ == k)
                    cluster_class_range[j] = np.nanmean(
                        dem_Range[cluster_indices3[cluster_class]])
                    if j == 0:
                        c0 = beamFootprintIndices_xyz_LE3[i][0][np.where(
                            clustering.labels_ == k)]
                        dem_inFootprint_xyz_LE3_c0[i] = [
                            dem_inFootprint[i][0][c0], dem_inFootprint[i][1][c0], dem_inFootprint[i][2][c0]]

                    if j == 1:
                        c1 = beamFootprintIndices_xyz_LE3[i][0][np.where(
                            clustering.labels_ == k)]
                        dem_inFootprint_xyz_LE3_c1[i] = [
                            dem_inFootprint[i][0][c1], dem_inFootprint[i][1][c1], dem_inFootprint[i][2][c1]]

                    if j == 2:
                        c2 = beamFootprintIndices_xyz_LE3[i][0][np.where(
                            clustering.labels_ == k)]
                        dem_inFootprint_xyz_LE3_c2[i] = [
                            dem_inFootprint[i][0][c2], dem_inFootprint[i][1][c2], dem_inFootprint[i][2][c2]]

                    if j == 3:
                        c3 = beamFootprintIndices_xyz_LE3[i][0][np.where(
                            clustering.labels_ == k)]
                        dem_inFootprint_xyz_LE3_c3[i] = [
                            dem_inFootprint[i][0][c3], dem_inFootprint[i][1][c3], dem_inFootprint[i][2][c3]]

            else:
                beamFootprintIndices_xyz_LE3_maxC = 0  # select the only indice
                cluster_ind3[i] = 1

            xdem_poca_xyz_LE3_meanD_maxC[i] = np.nanmean(
                dem_inFootprint_xyz_LE3[i][0][beamFootprintIndices_xyz_LE3_maxC])
            ydem_poca_xyz_LE3_meanD_maxC[i] = np.nanmean(
                dem_inFootprint_xyz_LE3[i][1][beamFootprintIndices_xyz_LE3_maxC])
            zdem_poca_xyz_LE3_meanD_maxC[i] = np.nanmean(
                dem_inFootprint_xyz_LE3[i][2][beamFootprintIndices_xyz_LE3_maxC])
            POCALoc_LE3_meanD_maxC = np.column_stack(
                [xdem_poca_xyz_LE3_meanD_maxC[i], ydem_poca_xyz_LE3_meanD_maxC[i], zdem_poca_xyz_LE3_meanD_maxC[i]])  # zip poca coords
            POCAToSat_meanD_maxC = POCALoc_LE3_meanD_maxC - \
                satLoc[i]  # get vector from sat to poca
            pocaRange_xyz_LE3_meanD_maxC[i] = np.linalg.norm(
                POCAToSat_meanD_maxC)  # convert to range
            # compute slope correction to apply to assumed nadir altimeter elevation to relocate to poca
            slopeCorrection_xyz_LE3_meanD_maxC[i] = pocaRange_xyz_LE3_meanD_maxC[i] + \
                zdem_poca_xyz_LE3_meanD_maxC[i] - altitude[i]
            #

    REMADict['dem_inFootprint_xyz_LE_c0'] = dem_inFootprint_xyz_LE_c0
    REMADict['dem_inFootprint_xyz_LE_c1'] = dem_inFootprint_xyz_LE_c1
    REMADict['dem_inFootprint_xyz_LE_c2'] = dem_inFootprint_xyz_LE_c2
    REMADict['dem_inFootprint_xyz_LE_c3'] = dem_inFootprint_xyz_LE_c3

    REMADict['dem_inFootprint_xyz_LE2_c0'] = dem_inFootprint_xyz_LE2_c0
    REMADict['dem_inFootprint_xyz_LE2_c1'] = dem_inFootprint_xyz_LE2_c1
    REMADict['dem_inFootprint_xyz_LE2_c2'] = dem_inFootprint_xyz_LE2_c2
    REMADict['dem_inFootprint_xyz_LE2_c3'] = dem_inFootprint_xyz_LE2_c3

    REMADict['dem_inFootprint_xyz_LE3_c0'] = dem_inFootprint_xyz_LE3_c0
    REMADict['dem_inFootprint_xyz_LE3_c1'] = dem_inFootprint_xyz_LE3_c1
    REMADict['dem_inFootprint_xyz_LE3_c2'] = dem_inFootprint_xyz_LE3_c2
    REMADict['dem_inFootprint_xyz_LE3_c3'] = dem_inFootprint_xyz_LE3_c3
    # ----------------------------------------------------------------------
    # Form Output Dictionary
    # ----------------------------------------------------------------------

    output = {


        "POCA x (LE_meanD_maxC)": xdem_poca_xyz_LE_meanD_maxC,
        "POCA y (LE_meanD_maxC)": ydem_poca_xyz_LE_meanD_maxC,
        "POCA z (LE_meanD_maxC)": zdem_poca_xyz_LE_meanD_maxC,
        "Slope Correction (LE_meanD_maxC)": slopeCorrection_xyz_LE_meanD_maxC,
        "pocaRange (LE_meanD_maxC)": pocaRange_xyz_LE_meanD_maxC,


        "POCA x (LE2_meanD_maxC)": xdem_poca_xyz_LE2_meanD_maxC,
        "POCA y (LE2_meanD_maxC)": ydem_poca_xyz_LE2_meanD_maxC,
        "POCA z (LE2_meanD_maxC)": zdem_poca_xyz_LE2_meanD_maxC,
        "Slope Correction (LE2_meanD_maxC)": slopeCorrection_xyz_LE2_meanD_maxC,
        "pocaRange (LE2_meanD_maxC)": pocaRange_xyz_LE2_meanD_maxC,


        "POCA x (LE3_meanD_maxC)": xdem_poca_xyz_LE3_meanD_maxC,
        "POCA y (LE3_meanD_maxC)": ydem_poca_xyz_LE3_meanD_maxC,
        "POCA z (LE3_meanD_maxC)": zdem_poca_xyz_LE3_meanD_maxC,
        "Slope Correction (LE3_meanD_maxC)": slopeCorrection_xyz_LE3_meanD_maxC,
        "pocaRange (LE3_meanD_maxC)": pocaRange_xyz_LE3_meanD_maxC,

        "ESApocaRange": ESApocaRange,
        "REMADict": REMADict,

        "cluster_indices1": cluster_ind1,
        "cluster_indices2": cluster_ind2,
        "cluster_indices3": cluster_ind3
    }

    return output
