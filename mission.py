# ------------------------------------------------------------------------------------------
# Import required python modules
# ------------------------------------------------------------------------------------------

import glob  # pathname pattern expansion functions
import time  # time functions
from datetime import datetime  # date & time functions
from datetime import timedelta  # time difference functions
from calendar import isleap
from math import modf
import os  # for Operating System functions like directory reading, creation
from netCDF4 import Dataset  # for NetCDF reading functionality
import numpy as np
from math import floor as floor
import logging
import h5py

log = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------------
# Module global variables
# ------------------------------------------------------------------------------------------

all_mission_list = [
    "S3A",
]  # List of supported altimeter missions (short names)
all_latencies = [
    "NRT",
    "NTC",
    "STC",
]  # NRT=near real time (within 3 hrs), STC=Slow Time Critical (within 48-hrs), NTC=not time critical (final quality),

# ------------------------------------------------------------------------------------------
# Classes
# ------------------------------------------------------------------------------------------


class Mission:
    """Class containing RA mission specific settings for S3A, S3B, EV, CS2, IS2, E1, E2

    functions to return a directory and file list for a mission L2 cycle
    """

    def __init__(self, name="S3A", dataset=None):
        """
        Set default values for each mission
        Parameters:
        name	:  string, mission identifier, S3A, S3B, EV, CS2, IS2, E1, E2 (upper or lower case)
        dataset :  default=None, which uses standard ESA L2(i) products.
                        :  If set, use alternative L2 mission dataset identifier, options are 'cs2_cryotempo_b'
                        :
        """

        self.dataset = dataset

        # Epoch for mission time functions
        self.epoch = datetime(1991, 1, 1, 0)

        self.name = name.upper()  # allows 's3a' as well as 'S3A' as input to name

        # Check the mission name is in the allowed mission list
        if self.name not in all_mission_list:
            print("Supported missions are ", all_mission_list)
            raise Exception("ERROR: mission not supported ", self.name)

        # Setup S3A mission defaults
        if self.name == "S3A":
            self.long_name = "Sentinel-3A"  # Long mission name

            # Pass (half-orbit) numbering
            self.reference_ground_track_low = (
                None  # lowest Ref. Ground Track in a cycle (used for ICESat-2)
            )
            self.reference_ground_track_high = (
                # highest Ref. Ground Track in a cycle (used for ICESat-2)
                None
            )

            # Mission L2 Latitude and Longitude parameter names
            self.lat_20_name = "lat_20_ku"  # 20Hz nadir latitude parameter name
            self.lon_20_name = "lon_20_ku"  # 20Hz nadir longitude parameter name
            self.lat_20_cor_name = "lat_cor_20_ku"  # 20Hz poca latitude parameter name
            self.lon_20_cor_name = "lon_cor_20_ku"  # 20Hz poca longitude parameter name
            self.lat_20_name_cband = "lat_20_c"  # 20Hz C-band latitude parameter name
            self.lon_20_name_cband = "lon_20_c"  # 20Hz C-band longitude parameter name
            self.lat_20_cor_name_cband = (
                "lat_cor_20_c"  # 20Hz C-band POCA latitude parameter name
            )
            self.lon_20_cor_name_cband = (
                "lon_cor_20_c"  # 20Hz C-band POCA longitude parameter name
            )
            self.lat_01_name = "lat_01"  # 1Hz nadir latitude parameter name
            self.lon_01_name = "lon_01"  # 1Hz nadir longitude parameter name

        else:
            raise Exception("ERROR: mission not supported ", self.name)

    def get_1hz_lat_lon(self, nc):
        """

        :param nc: NetCDF Dataset()

        returns 1Hz latitudes, longitudes East (0..360)
        """

        if self.name == "S3A":

            return nc.variables[self.lat_01_name][:], np.mod(
                nc.variables[self.lon_01_name][:], 360
            )

    def get_20hz_lat_lon(
        self, nc, nadir=False, replace_badslp_with_nadir_locations=False, l1b=False
    ):
        """

        :param nc: NetCDF Dataset()
        :param nadir: if true return the lat, lon at nadir, if false return poca lat,lon

        returns latitudes, longitudes East (0..360)
        """

        if self.name == 'S3A':

            if nadir or l1b:
                return nc.variables[self.lat_20_name][:], np.mod(
                    nc.variables[self.lon_20_name][:], 360
                )
            else:
                if replace_badslp_with_nadir_locations:
                    lat_20_cor = nc.variables[self.lat_20_cor_name][:]
                    lon_20_cor = np.mod(
                        nc.variables[self.lon_20_cor_name][:], 360)
                    bad = np.flatnonzero(lat_20_cor.mask == True)
                    if bad.size > 0:
                        try:  # some datasets like CryoTEMPO don't have nadir locations, in which case skip this
                            lat_20_nadir, lon_20_nadir = self.get_20hz_lat_lon(
                                nc, nadir=True
                            )
                            lat_20_cor[bad] = lat_20_nadir[bad]
                            lon_20_cor[bad] = lon_20_nadir[bad]
                        except:
                            pass
                    return lat_20_cor, lon_20_cor

                else:
                    return nc.variables[self.lat_20_cor_name][:], np.mod(
                        nc.variables[self.lon_20_cor_name][:], 360
                    )
