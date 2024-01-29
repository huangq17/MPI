from typing import Tuple, List
from socketserver import ThreadingUDPServer
import numpy as np
import pyproj as proj  # cartographic projections and coordinate transformations library
from pyproj import (
    Transformer,
    CRS,
)  # Transformer transforms between projections, CRS is an alternative to pyproj.Proj

from masks import Mask  # cryosphere area grid and polygon mask definitions

# Class contains settings and defaults for each defined ice area

# List of all defined land ice areas
landice_areas = [
    "greenland",  # Greenland area. Data mask: GIS: all Zwally 2012 GIS basin polygons combined   \
]


class Area:
    def __init__(self, area=None, load_mask=True, existing_mask=None):
        """
        :param area: string containing area specifier name such as 'arctic'. Must be in landice_areas, or seaice_areas list
        :param load_mask: if True load the data mask. This can be slow, so you may wish to manage when this is done externally
        :param existing_mask: provide a Mask() object if you have an already loaded mask elsewhere. This is to avoid reloading the mask (which can be slow)
        """
        self.area = area

        # Set area defaults: overide in each area section

        # For lat/lon -> x,y transforms for each area
        # assuming you're using WGS84 geographic
        self.crs_wgs = CRS("epsg:4326")
        # Default is Polar Stereo south. Changed within each area init for Northern areas
        self.crs_bng = CRS("epsg:3031")  # Polar Stereo - South -71S
        self.crs_number = None  # for non-polar EPSG numbers
        self.projection_extent = None  # [lon0,lon1,lat0,lat1]

        self.lon_0 = None  # projection y-axis longitude

        # Data masking
        self.mask = None if not existing_mask else existing_mask  # Mask() class
        self.masktype = (
            None if not existing_mask else existing_mask.mask_type
        )  # data mask type : 'grid', 'polygon', 'limit',or None
        self.maskname = None if not existing_mask else existing_mask.mask_name
        self.basin_numbers = (
            None if not existing_mask else existing_mask.basin_numbers
        )  # basin number of Antarctic Zwally basins (1..27,29-31) or GIS basins (1..19,20-26) to include in mask
        # Mask from data's surface_type parameter, 0 or None=no_masking, 1=ocean,2=grounded ice, 3=floating ice, 4=ground+floating
        self.data_surface_type_mask = None

        if area == None:
            pass

        # ------------------------------------------------------------
        #  GREENLAND
        # ------------------------------------------------------------
        elif area == "greenland":
            self.area_name = "Greenland Ice Sheet"

            # recommended gridding bin size for this area
            self.gridsize_km = 5

            # Projection for x,y <-> lat,lon transforms
            self.crs_bng = CRS(
                "epsg:3413"
            )  # Polar Stereo - North -latitude of origin 70N, 45

            # Area rectangle or circle
            self.specify_by_centre = (
                True  # specify plot area by centre lat/lon, width, height (km)
            )

            self.hemisphere = "north"

            # Area min/max lat/lon for initial data filtering
            # minimum longitude to initially filter records for area (0..360E)
            self.minlon = 285.0
            # maximum longitude to initially filter records for area (0..360E)
            self.maxlon = 350.0
            self.minlat = 60.0  # minimum latitude to initially filter records for area
            self.maxlat = 83.0  # maximum latitude to initially filter records for area

            # Data mask
            self.masktype = "grid"
            self.maskname = "greenland_icesheet_2km_grid_mask"

        else:
            self.area_name = "unknown"
            print(
                "Area ",
                area,
                " not found in Areas class\nFor a full list of allowed names, run: plot_param_map_multi.py --listareas",
            )
            exit(1)

        # ---------------------------------------------------------------------------------------------
        # load_mask = False, existing_mask = None
        # ---------------------------------------------------------------------------------------------
        if load_mask:
            if self.mask is None:
                if self.masktype is not None:
                    self.mask = Mask(self.maskname, self.basin_numbers)

        if self.hemisphere == "north":
            self.arctic_dem_available = True
        if self.hemisphere == "south":
            self.antarctic_dem_available = True

        # Setup the Transforms
        self.xy_to_lonlat_transformer = Transformer.from_proj(
            self.crs_bng, self.crs_wgs, always_xy=True
        )
        self.lonlat_to_xy_transformer = Transformer.from_proj(
            self.crs_wgs, self.crs_bng, always_xy=True
        )

        if self.minlon < 0.0:
            self.minlon = 360 + self.minlon
        if self.maxlon < 0.0:
            self.maxlon = 360 + self.maxlon

    def inside_latlon_bounds(self, lats, lons):
        """
        find the locations inside the area's bounding box (note this is not the area mask, but a quick filter
        to reduce the number of points that require masking (which can be slow))

        returns: [latitude values in bounding box] [longitude values in bounding box] [indices in box] number_in_box

        """

        in_lat_area = np.logical_and(lats >= self.minlat, lats <= self.maxlat)
        in_lon_area = np.logical_and(lons >= self.minlon, lons <= self.maxlon)
        bounded_indices = np.flatnonzero(in_lat_area & in_lon_area)
        if bounded_indices.size > 0:
            bounded_lats = lats[bounded_indices]
            bounded_lons = lons[bounded_indices]
            return bounded_lats, bounded_lons, bounded_indices, bounded_indices.size
        else:
            return None, None, None, 0
