
from cis.data_io.products.AProduct import AProduct
from cis.data_io.Coord import Coord, CoordList
from cis.data_io.ungridded_data import UngriddedData
from cis.data_io.ungridded_data import Metadata
import cis.data_io.netcdf as netcdf
import numpy as np
import h5py
import re
import os
import logging
from datetime import datetime

class SEVIRI_L2_CLAAS_HDF(AProduct):
    """
    Reads cloud top temperature variable from a CMSAF (CTX version 001) HDF file containing non CF compliant metadata.
    Applies scale and offset to data.
    Masks fill values.
    Creates variable metadata.
    Reads latitude and longitude values from a separate HDF file.
    Returns data, metadata and coordinates.
    """

    def get_file_signature(self):
        #"Returns a list of regular expressions, which are used to decided which files require this plugin."
        
        return [r'.*/np/.*/CPPin.*\.nc$']
    
    @staticmethod
    def _find_dataset(h5obj, target_name):
        """Find a dataset by full path or by leaf name anywhere in the file."""
        # Direct path
        if target_name in h5obj and isinstance(h5obj[target_name], h5py.Dataset):
            return h5obj[target_name]

        matches = []
        def _visitor(name, obj):
            if isinstance(obj, h5py.Dataset) and name.split("/")[-1] == target_name:
                matches.append(obj)
        h5obj.visititems(_visitor)

        if not matches:
            raise KeyError(f"Dataset '{target_name}' not found in HDF5 file.")
        if len(matches) > 1:
            logging.warning(f"Multiple datasets named '{target_name}' found; using the first at {matches[0].name}")
        return matches[0]

    def create_data_object(self, filenames, variable):
        logging.debug("Creating data object (h5py, all files) for variable " + variable)

        # Ensure we have a list of files
        if isinstance(filenames, (str, bytes)):
            filenames = [filenames]

        # Let your existing logic build coordinates for ALL files
        coords = self.create_coords(filenames, variable)

        # Read and concatenate data from every file
        arrays = []
        for fn in filenames:
            with h5py.File(fn, "r") as f:
                ds = self._find_dataset(f, variable)
                data = ds[()]  # numpy array

                # Keep dtype; just ensure it's numeric before masking comparison
                if not np.issubdtype(data.dtype, np.number):
                    data = data.astype(np.float32)

                # DO NOT apply scale/offset (matches your original commented-out behavior)

                # Mask fill (original behavior): <= 0.0
                data = np.ma.masked_less_equal(data, 0.0)

                arrays.append(data)

        # Try to concatenate along the first axis if shapes align; otherwise flatten+stack
        try:
            var_data = np.ma.concatenate(arrays, axis=0)
        except Exception:
            logging.warning("Could not concatenate along axis=0; falling back to 1D stacking.")
            arrays_flat = [a.ravel() for a in arrays]
            var_data = np.ma.concatenate(arrays_flat, axis=0)

        # Keep your original metadata exactly
        metadata = Metadata(name='Cloud_Top_Phase', units='')

        logging.info(type(var_data))
        logging.info(type(metadata))
        logging.info(type(coords))

        return UngriddedData(var_data, metadata, coords)

    def create_coords(self, filenames, variable=None):
        # "Read in the relevant coordinates for the specified file and return them as a list."

        # list of coordinates to be read in
        coordinates = ['lat', 'lon']
        logging.info("Listing coordinates: " + str(coordinates))

        # in this case, the coord information is in a separate file
        aux_filenames = ['/wolke_scratch/dnikolo/CLAAS_Data/np/CM_SAF_CLAAS3_L2_AUX.nc']

        aux_data = netcdf.read(aux_filenames[0], coordinates)
        lat = aux_data['lat']
        lon = aux_data['lon']

        # mask out fill values
        lat_data = netcdf.get_data(lat)[0]
        lon_data = netcdf.get_data(lon)[0]

        ny, nx = lat_data.shape
        nt = len(filenames)

        # ---------- Parse time from filenames ----------
        # Filenames look like: CPPinYYYYMMDDHHMMSS... (e.g., CPPin20080201001500...)
        # Fallback also uses the /YYYY/MM/DD/ path parts if needed.
        times_dt = []
        for fn in filenames:
            base = os.path.basename(fn)
            m = re.search(r'CPPin(\d{8})(\d{6})', base)
            if not m:
                # Fallback: /YYYY/MM/DD/...HHMMSS...
                m = re.search(r'/(\d{4})/(\d{2})/(\d{2})/.*?(\d{6})', fn)
                if not m:
                    raise ValueError(f"Could not parse time from filename: {fn}")
                ymd = m.group(1) + m.group(2) + m.group(3)
                hms = m.group(4)
            else:
                ymd, hms = m.group(1), m.group(2)

            dt_obj = datetime.strptime(ymd + hms, "%Y%m%d%H%M%S")
            times_dt.append(dt_obj)

        # Convert to numeric CF-style time: seconds since 1970-01-01 00:00:00
        epoch = datetime(1970, 1, 1)
        times_sec = np.array([(t - epoch).total_seconds() for t in times_dt], dtype=np.float64)

        # ---------- Broadcast coords across time and flatten ----------
        # Repeat across time dimension
        lat_3d = np.broadcast_to(lat_data, (nt, ny, nx))
        lon_3d = np.broadcast_to(lon_data, (nt, ny, nx))
        time_3d = np.broadcast_to(times_sec.reshape(nt, 1, 1), (nt, ny, nx))

        # Flatten to 1-D for Ungridded coordinates
        lat_1d = lat_3d.ravel()
        lon_1d = lon_3d.ravel()
        time_1d = time_3d.ravel()

        # ---------- Create metadata ----------
        logging.info("gathering coordinates metadata")
        lat_metadata = Metadata(
            name='lat', standard_name='latitude', long_name='Latitude',
            shape=None, units='degrees_north', range=[-81, 81], missing_value=-9999.0
        )
        lon_metadata = Metadata(
            name='lon', standard_name='longitude', long_name='Longitude',
            shape=None, units='degrees_east', range=[-81, 81], missing_value=-9999.0
        )
        time_metadata = Metadata(
            name='time', standard_name='time', long_name='Time',
            shape=None, units='seconds since 1970-01-01 00:00:00',
            range=None, missing_value=-9999.0
        )

        # ---------- Build Coord objects ----------
        # Use 'T' for time, 'Y' for latitude, 'X' for longitude
        time_coord = Coord(time_1d, time_metadata, 'T')
        lat_coord = Coord(lat_1d, lat_metadata, 'Y')
        lon_coord = Coord(lon_1d, lon_metadata, 'X')

        return CoordList([time_coord, lat_coord, lon_coord])
