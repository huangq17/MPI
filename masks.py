
# ----------------------------------------------------------------------
#  Import python modules
# ----------------------------------------------------------------------

import sys
from os import environ  # operating system functions

import numpy as np  # Numpy maths and array functions
import pyproj as proj  # cartographic projections and coordinate transformations library
import shapefile  # shapefile reader
from netCDF4 import Dataset  # NetCDF functions
from pyproj import (CRS,  # transform between projections, and CRS definitions
                    Transformer)
from shapely.geometry import Point  # point functions
from shapely.geometry.polygon import Polygon  # polygon functions

# List of all current Cryosphere masks
#    - if adding to list, remember to add a \ after the comment section
cryosphere_mask_list = [
    'greenland_icesheet_2km_grid_mask',  # Greenland ice sheet grounded ice mask, from 2km grid, source: Zwally 2012. Can select basins \

]

# ----------------------------------------------------------------------
#  Classes
# ----------------------------------------------------------------------


class Masks:
    """
    class to keep track of which Cryosphere area masks have been loaded (read in to memory).
    Useful if processing multiple areas, where you may want to use area masks multiple times
    (but not reload them- loading is slow for some masks (ie 2km grids)).

    Example:
    --------
    masks=Masks()
    thismask=masks.load('some_mask')
    ...
    thismask=masks.load('some_other_mask')
    ...
    thismask=masks.load('some_mask')  #  in this case mask will already be available
                                      #  and does not need to be read in from disk


    """

    def __init__(self):

        self.greenland_icesheet_2km_grid_mask = None

    def load(self, name, basin_number=None):

        if not name:
            return None
        if name == 'none':
            return None

        print('Loading ', name)

        # only current allows 1 basin number to be passed in. If a list use first one
        if isinstance(basin_number, list):
            basin_number = basin_number[0]

        if name not in cryosphere_mask_list:
            sys.exit(name+' not in allowed Cryosphere mask list')

        # latitude limits mask

        elif name == 'greenland_icesheet_2km_grid_mask':
            if not self.greenland_icesheet_2km_grid_mask:
                self.greenland_icesheet_2km_grid_mask = Mask(
                    'greenland_icesheet_2km_grid_mask')
            return self.greenland_icesheet_2km_grid_mask

        else:
            sys.exit('Mask not found', name)


class Mask:

    crs_wgs = CRS('epsg:4326')  # assuming you're using WGS84 geographic

    def __init__(self, mask_name, basin_numbers=None):
        """
        mask_name : string containing mask name. Example : 'antarctic_icesheet_2km_grid_mask'
        basin_numbers : an optional list of basin numbers
        """

        print('Loading Mask ', mask_name)

        self.mask_name = mask_name
        self.nomask = False
        self.mask_type = None  # 'xylimits', 'polygon', 'grid','latlimits'

        if not mask_name:
            print('mask_name is None, nothing to initialise')
            self.nomask = True
            return

        if mask_name not in cryosphere_mask_list:
            sys.exit(mask_name+' not in allowed Cryosphere mask list')

        self.mask_name = mask_name

        self.grid_value_names = None
        self.grid_colors = None
        self.mask_grid_possible_values = None

        if basin_numbers:  # check if basin_numbers is a scalar. If so turn in to a single value list
            if not isinstance(basin_numbers, list):
                basin_numbers = [basin_numbers]
        self.basin_numbers = basin_numbers

        self.polygons = None
        self.polygons_lon = np.array([])
        self.polygons_lat = np.array([])
        self.polygon = None
        self.polygon_lon = np.array([])
        self.polygon_lat = np.array([])

        # --------------------------------------------------------------------------------------------------------------
        # Initialise Latitude Limits Masks
        # --------------------------------------------------------------------------------------------------------------

        if mask_name == 'greenland_icesheet_2km_grid_mask':
            self.mask_type = 'grid'  # 'xylimits', 'polygon', 'grid','latlimits'

            print('Setting up greenland_icesheet_2km_grid_mask..')

            # read netcdf file
            nc = Dataset(environ['CPOM_SOFTWARE_DIR'] +
                         '/cpom/resources/drainage_basins/greenland/zwally_2012_grn_icesheet_basins/basins/Zwally_GIS_basins_2km.nc')

            self.nx = nc.dimensions['gre_basin_nx'].size
            self.ny = nc.dimensions['gre_basin_ny'].size

            self.minxm = nc.variables['gre_basin_minxm'][:]
            self.minym = nc.variables['gre_basin_minym'][:]
            self.binsize = nc.variables['gre_basin_binsize'][:]
            self.mask_grid = np.array(
                nc.variables['gre_basin_mask'][:]).astype(int)
            nc.close()
            # Polar Stereo - North -latitude of origin 70N, 45
            self.crs_bng = CRS('epsg:3413')
            self.grid_value_names = ['None', '1.1', '1.2', '1.3', '1.4', '2.1', '2.2', '3.1',
                                     '3.2', '3.3', '4.1', '4.2', '4.3', '5.0', '6.1', '6.2', '7.1', '7.2', '8.1', '8.2']
            self.mask_grid_possible_values = [
                i for i in range(20)]  # values in the mask_grid
            self.grid_colors = ['blue', 'bisque', 'darkorange', 'moccasin', 'gold', 'greenyellow', 'yellowgreen', 'gray', 'lightgray', 'silver', 'purple',
                                'sandybrown', 'peachpuff', 'coral', 'tomato', 'navy', 'lavender', 'olivedrab', 'lightyellow', 'sienna']

        # Setup the Transforms
        self.xy_to_lonlat_transformer = Transformer.from_proj(
            self.crs_bng, self.crs_wgs, always_xy=True)
        self.lonlat_to_xy_transformer = Transformer.from_proj(
            self.crs_wgs, self.crs_bng, always_xy=True)

        print("mask setup completed")

    def points_inside(self, lats, lons, basin_numbers=None, inputs_are_xy=False):
        """
        lats: list or array of latitude values
        lons: list or array of longitude values (0..360E)
        basin_numbers : list of basin numbers to include in mask (ie [1,2,3]).
                                        If None include entire mask area.
                        For polygon masks, only 1 basin number is allowed
                        For grid masks, multiple basin numbers can be specified
                        basin_numbers overrides self.basin_numbers in Mask initialisation
        inputs_are_xy : True if input lats, lons are instead polar stereo x,y (ie transform already done)
        returns [inmask,x,y]
        inmask= [array  of 0s (not in mask area) or 1s (in mask area)]
        x=[array of transformed x locations of all input lats,lons]
        y=[array of transformed y locations of all input lats,lons]
        """

        if not self.mask_name:
            inmask = np.zeros(lats.size, np.bool_)
            return inmask, None, None

        if not isinstance(lats, np.ndarray):
            if isinstance(lats, list):
                lats = np.array(lats)
            else:
                sys.exit('lats  must be an array or list of values')
        if not isinstance(lons, np.ndarray):
            if isinstance(lons, list):
                lons = np.array(lons)
            else:
                sys.exit('lons  must be an array or list of values')

        if basin_numbers:  # turn in to a list if a scalar
            if not isinstance(basin_numbers, (list, np.ndarray)):
                basin_numbers = [basin_numbers]

        if self.basin_numbers:
            if not basin_numbers:
                basin_numbers = self.basin_numbers

        # --------------------------------------------------------------------------------------------
        # Find points inside a lat/lon limit mask
        # --------------------------------------------------------------------------------------------
        if self.mask_type == 'latlimits':

            if inputs_are_xy:
                x, y = lats, lons
            else:
                x, y = self.latlon_to_xy(lats, lons)

            inmask = np.logical_and(lats >= np.min(
                self.latlimits), lats <= np.max(self.latlimits))

            return inmask, x, y
        # --------------------------------------------------------------------------------------------
        # Find points inside a x,y limit mask
        # --------------------------------------------------------------------------------------------

        if self.mask_type == 'xylimits':
            inmask = np.zeros(lats.size, np.bool_)

            # then cast your geographic coordinate pair to the projected system
            if inputs_are_xy:
                x, y = lats, lons
            else:
                x, y = self.latlon_to_xy(lats, lons)

            for i in range(x.size):
                if (x[i] >= self.xlimits[0] and x[i] <= self.xlimits[1]) and (y[i] >= self.ylimits[0] and y[i] <= self.ylimits[1]):
                    inmask[i] = True
            return inmask, x, y

        # Zwally Grid Masks
        if (self.mask_name == 'greenland_icesheet_2km_grid_mask'):

            if inputs_are_xy:
                x, y = lats, lons
            else:
                x, y = self.lonlat_to_xy_transformer.transform(lons, lats)

            inmask = np.zeros(lats.size, np.bool_)

            for i in range(0, lats.size):
                # calculate equivalent (ii,jj) in mask array
                ii = int(np.around((x[i] - self.minxm)/self.binsize))
                jj = int(np.around((y[i] - self.minym)/self.binsize))

                # Check bounds of Basin Mask array
                if ii < 0 or ii >= self.nx:
                    continue
                if jj < 0 or jj >= self.ny:
                    continue

                if self.mask_grid[jj, ii] > 0:
                    inmask[i] = True
            return inmask, x, y

    def latlon_to_xy(self, lats, lons):
        """
        :param lats: latitude points in degs
        :param lons: longitude points in degrees E
        :return: x,y in polar stereo projection of mask
        """
        return self.lonlat_to_xy_transformer.transform(lons, lats)
