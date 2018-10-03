# coding=utf-8
from itertools import cycle

import numpy as np
import xarray as xr

from geotechnicalprofile.gef_helpers import hydraulic_conductance
from geotechnicalprofile.gef_helpers import hydraulic_resistance
from geotechnicalprofile.gef_helpers import lithology
from geotechnicalprofile.gef_helpers import fijnedeeltjes

supported_version = [1, 1, 0]
multiple_list = [
    'COLUMNINFO', 'COLUMNVOID', 'MEASUREMENTTEXT', 'MEASUREMENTVAR'
]
dtyped = {
    'default': [str],
    'GEFID': [int],
    'COLUMN': [int],
    'LASTSCAN': [int],
    'COLUMNINFO': [int, str, str, int],
    'COLUMNVOID': [int, float],
    'MEASUREMENTVAR': [int, float, str, str],
    'XYID': [int, float, float, float, float],
    'ZID': [int, float, float]
}


class DataStore(xr.Dataset):
    """The data class that stores the measurements. The user should never initiate this class
    directly, but use read_xml_dir or open_datastore functions instead.

        Parameters
        ----------
        data_vars : dict-like, optional
            A mapping from variable names to :py:class:`~xarray.DataArray`
            objects, :py:class:`~xarray.Variable` objects or tuples of the
            form ``(dims, data[, attrs])`` which can be used as arguments to
            create a new ``Variable``. Each dimension must have the same length
            in all variables in which it appears.
        coords : dict-like, optional
            Another mapping in the same form as the `variables` argument,
            except the each item is saved on the datastore as a "coordinate".
            These variables have an associated meaning: they describe
            constant/fixed/independent quantities, unlike the
            varying/measured/dependent quantities that belong in `variables`.
            Coordinates values may be given by 1-dimensional arrays or scalars,
            in which case `dims` do not need to be supplied: 1D arrays will be
            assumed to give index values along the dimension with the same
            name.
        attrs : dict-like, optional
            Global attributes to save on this datastore.
        sections : dict, optional
            Sections for calibration. The dictionary should contain key-var couples
            in which the key is the name of the calibration temp time series. And
            the var is a list of slice objects as 'slice(start, stop)'; start and
            stop in meter (float).
        compat : {'broadcast_equals', 'equals', 'identical'}, optional
            String indicating how to compare variables of the same name for
            potential conflicts when initializing this datastore:

            - 'broadcast_equals': all values must be equal when variables are
              broadcast against each other to ensure common dimensions.
            - 'equals': all values and dimensions must be the same.
            - 'identical': all values, dimensions and attributes must be the
              same.

        See Also
        --------
        dtscalibration.read_xml_dir : Load measurements stored in XML-files
        dtscalibration.open_datastore : Load (calibrated) measurements from netCDF-like file
        """

    def __init__(self, *args, **kwargs):
        super(DataStore, self).__init__(*args, **kwargs)


def read_gef(fp):
    attrs = {}

    with open(fp, 'rb') as f:
        s = str(f.readline(), 'ASCII')
        key_val(s, k='GEFID', v=supported_version, dtyped=dtyped, out=attrs)

        while True:
            s = str(f.readline(), 'ASCII', 'ignore')
            if s[1:4] == 'EOH':  # End of header
                break

            key_val(s, dtyped=dtyped, out=attrs)

        X = np.loadtxt(f)

    assert X.shape == (attrs['LASTSCAN'], attrs['COLUMN'])

    for col_info in attrs['COLUMNVOID']:
        mask = np.isclose(X[:, col_info[0] - 1], col_info[1])
        X[mask, col_info[0] - 1] = np.nan

    attrs['x (m)'] = attrs['XYID'][1]
    attrs['y (m)'] = attrs['XYID'][2]
    attrs['maaiveld (m+NAP)'] = attrs['ZID'][1]

    for item in attrs['MEASUREMENTVAR']:
        key = item[3] + ' (' + item[2] + ')'
        attrs[key] = item[1]

    labels = [item[2] + ' (' + item[1] + ')' for item in attrs['COLUMNINFO']]

    hellingsmeter_aanwezig = 'gecorrigeerde diepte (m)' in labels

    if hellingsmeter_aanwezig:
        diepte = 'gecorrigeerde diepte (m)'

    else:
        diepte = 'sondeerlengte (m)'

    dim_col = labels.index(diepte)
    dim_dat = attrs['ZID'][1] - X[:, dim_col]
    dim_label = r'z (m+NAP)'

    data = {}
    for label, x_item in zip(labels, X.T):
        data[label] = (r'z (m+NAP)', x_item)

    ds = DataStore(data_vars=data, coords={dim_label: dim_dat}, attrs=attrs)

    hydraulic_conductance(ds)
    hydraulic_resistance(ds)
    lithology(ds)
    fijnedeeltjes(ds)

    return ds


def key_val(s, k=None, v=None, dtyped=None, out=None):
    s2 = s.split(sep='= ')
    s2_k = s2[0][1:]

    if s2_k in dtyped:
        dtype = dtyped[s2_k]

    else:
        dtype = dtyped['default']

    if len(s2) == 1:
        s2_k = s2_k[:-3]
        s2_v = ''

    else:
        i = zip(cycle(dtype), s2[1][:-2].split(sep=', '))

        try:
            s2_v = [dt(s2i) for dt, s2i in i]

        except:
            print(s2_k,
                  f". Interpreting {s2[1][:-2].split(sep=', ')} with {dtype}")

        if s2_k not in multiple_list and len(s2_v) == 1:
            s2_v = s2_v[0]

    # checks
    if k:
        assert k.upper() == s2[0][1:], \
            f'Was expecting a {k.upper()} entry in the file.\nGot {s2[0][1:]} instead'

    if v:
        assert v == s2_v, \
            f'Was expecting to read {v} for {k} in the file.\nGot {s2_v} instead.'

    # save result
    if out is not None:
        if s2_k in out:
            out[s2_k].append(s2_v)

        elif s2_k in multiple_list:
            out[s2_k] = [s2_v]

        else:
            out[s2_k] = s2_v

    else:
        return s2_k, s2_v
