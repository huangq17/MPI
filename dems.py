
import logging
from pyproj import Transformer, CRS   # transforms and coord definitions
# import pyproj as proj  # cartographic projections and coordinate transformations library
# from scipy.ndimage.filters import gaussian_filter
import sys
import os
from tifffile import imread  # to support large TIFF files
# from netCDF4 import Dataset  # http://code.google.com/p/netcdf4-python/
import numpy as np
from scipy.interpolate import interpn
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


log = logging.getLogger(__name__)

dem_list = ['arcticdem_1km',

            'arcticdem_100m_greenland_9_year_dhdt',

            ]

# Class contains DEM handling


class Dem:
    def __init__(self, name, filled=True):
        self.name = name
        self.crs_wgs = CRS('epsg:4326')  # we are using WGS84 for all DEMs

        if name not in dem_list:
            log.error('DEM {} not in allowed list'.format(name))
            sys.exit(1)
        # -------------------------------------------------------------------------------------------------------------------
        #  PGC ArcticDEM 1km Mosaic
        # -------------------------------------------------------------------------------------------------------------------

        if name == 'arcticdem_1km':
            # Source: http://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/mosaic/v3.0/1km/arcticdem_mosaic_1km_v3.0.tif.
            # EPSG:3413,  Polar Stereographic North projection
            demfile = os.environ['CPDATA_DIR'] + \
                '/SATS/RA/DEMS/arctic_dem_1km/arcticdem_mosaic_1km_v3.0.tif'
            print('Loading ArcticDEM 1km Greenland DEM..')
            print(demfile)
            im = Image.open(demfile)

            ncols, nrows = im.size
            self.zdem = np.array(im.getdata()).reshape((nrows, ncols))

            # Set void data to Nan
            void_data = np.where(self.zdem == -9999)
            if np.any(void_data):
                self.zdem[void_data] = np.nan

            self.xdem = np.linspace(-4000000.000,
                                    3400000.000, ncols, endpoint=True)
            self.ydem = np.linspace(-3400000.000,
                                    4100000.000, nrows, endpoint=True)
            self.ydem = np.flip(self.ydem)
            self.mindemx = self.xdem.min()
            self.mindemy = self.ydem.min()
            self.binsize = 1e3  # 1km grid resolution in m
            self.src_institute = 'PGC'
            self.name = 'ArcticDEM 1km'
            # Polar Stereo - North -latitude of origin 70N, 45
            self.crs_bng = CRS('epsg:3413')
            self.southern_hemisphere = False

        if name == 'arcticdem_100m_greenland_9_year_dhdt':
            # Source: https://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/mosaic/v3.0/100m/arcticdem_mosaic_100m_v3.0.tif
            # EPSG:3413,  Polar Stereographic North projection
            demfile = os.environ['CPDATA_DIR'] + \
                '/SATS/RA/DEMS/arctic_dem_100m/arcticdem_mosaic_100m_v3.0_greenland_9dhdt.tif'
            print('Loading ArcticDEM 100m DEM: for Greenland_9ydhdt..')
            print(demfile)

            im = imread(demfile)

            nrows, ncols = im.shape
            self.zdem = im

            print('DEM nrows x ncols = {} x {}'.format(nrows, ncols))

            # Set void data to Nan
            if 0:
                void_data = np.where(self.zdem == -9999)
                if np.any(void_data):
                    self.zdem[void_data] = np.nan

            self.xdem = np.linspace(-687900.000,
                                    962100.000, ncols, endpoint=True)
            self.ydem = np.linspace(-3410600.000,  -
                                    510600.000, nrows, endpoint=True)
            self.ydem = np.flip(self.ydem)
            self.mindemx = self.xdem.min()
            self.mindemy = self.ydem.min()
            self.binsize = 100  # 100m grid resolution in m
            self.src_institute = 'PGC'
            self.name = 'ArcticDEM 100m: Greenland_9ydhdt'
            # Polar Stereo - North -latitude of origin 70N, 45
            self.crs_bng = CRS('epsg:3413')
            self.southern_hemisphere = False

        # Setup the Transforms
        self.xy_to_lonlat_transformer = Transformer.from_proj(
            self.crs_bng, self.crs_wgs, always_xy=True)
        self.lonlat_to_xy_transformer = Transformer.from_proj(
            self.crs_wgs, self.crs_bng, always_xy=True)

    # ----------------------------------------------------------------------------------------------
    # Interpolate DEM, input x,y can be arrays or single, units m, in projection (epsg:3031")
    # returns the interpolated elevation(s) at x,y
    # x,y : x,y cartesian coordinates in the DEM's projection in m
    # OR, when xy_is_latlon is True:
    # x,y : latitude, longitude values in degs N and E (note the order, not longitude, latitude!)
    #
    # method: string containing the interpolation method. Default is 'linear'. Options are
    # “linear” and “nearest”, and “splinef2d” (see scipy.interpolate.interpn docs).
    #
    # Where your input points are outside the DEM area, then np.nan values will be returned
    # ----------------------------------------------------------------------------------------------

    def interp_dem(self, x, y, method='linear', xy_is_latlon=False):
        # Transform to x,y if inputs are lat,lon
        if xy_is_latlon:
            x, y = self.lonlat_to_xy_transformer.transform(
                y, x)  # transform lon,lat -> x,y
        myydem = np.flip(self.ydem.copy())
        myzdem = np.flip(self.zdem.copy(), 0)
        return interpn((myydem, self.xdem), myzdem, (y, x), method=method, bounds_error=False, fill_value=np.nan)
