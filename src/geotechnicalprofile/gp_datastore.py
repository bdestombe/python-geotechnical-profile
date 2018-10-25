# coding=utf-8
import inspect
import tempfile
from datetime import datetime as dt
from itertools import cycle

import numpy as np
import pandas as pd
import xarray as xr

from geotechnicalprofile.gef_helpers import fijnedeeltjes
from geotechnicalprofile.gef_helpers import hydraulic_conductance
from geotechnicalprofile.gef_helpers import hydraulic_resistance
from geotechnicalprofile.gef_helpers import lithology

supported_version = [1, 1, 0]
multiple_list = [
    'COLUMNINFO', 'COLUMNVOID', 'MEASUREMENTTEXT', 'MEASUREMENTVAR'
    ]
dtyped = {
    'default':        [str],
    'GEFID':          [int],
    'COLUMN':         [int],
    'LASTSCAN':       [int],
    'STARTDATE':      [int],
    'STARTTIME':      [int, int, float],
    'COLUMNINFO':     [int, str, str, int],
    'COLUMNVOID':     [int, float],
    'MEASUREMENTVAR': [int, float, str, str],
    'XYID':           [int, float, float, float, float],
    'ZID':            [int, float, float]
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

    def __init__(self, *args, geffile=None, **kwargs):
        super(DataStore, self).__init__(*args, **kwargs)

        if '_geffile' not in self.attrs:
            self.attrs['_geffile'] = ''

        self.geffile = geffile

    # noinspection PyIncorrectDocstring
    @property
    def geffile(self):
        """
        returns the orional gef file as a string
        """
        assert hasattr(self, '_geffile'), 'first set the geffile'
        return self.attrs['_geffile']

    @geffile.setter
    def geffile(self, geffile):
        """geffile can be a filepath or a filehandle or None"""

        if not hasattr(geffile, 'read') and geffile:
            # geffile is a filepath
            with open(geffile, 'rb') as infile:
                s = str(infile.read(), 'ASCII', 'ignore')

        elif not geffile:
            s = ''

        else:
            # GEF file is already a open filehandle
            s = str(geffile.read(), 'ASCII', 'ignore')

        self.attrs['_geffile'] = s
        pass

    @geffile.deleter
    def geffile(self):
        self.geffile = ''
        pass

    # noinspection PyIncorrectDocstring
    @property
    def geffilehandle(self):
        """Returns a filehandle of the geffile"""
        assert hasattr(self, '_geffile'), 'first set the geffile'
        s = self.attrs['_geffile']
        fh = tempfile.TemporaryFile(mode='r+b')
        fh.write(s.encode())
        fh.seek(0)
        return fh

    @geffilehandle.setter
    def geffilehandle(self, geffile):
        self.geffile = geffile

    @geffile.deleter
    def geffile(self):
        del self.geffile

    def add_dts(self, dts, x_top, x_bot, dts_tmp_label, gef_tmp_label, time_ref=None,
                depth_coords='depth_dts'):
        if x_top < x_bot:
            flip_flag = False
        else:
            flip_flag = True

        # Fill GEF datastore with x and time coords
        # Align DTS x
        if flip_flag:
            x_slice = slice(x_top, x_bot, -1)
        else:
            x_slice = slice(x_top, x_bot)

        dss = dts.sel(x=x_slice)

        if flip_flag:
            y = dss.x.data - x_slice.start
        else:
            y = x_slice.start - dss.x.data

        self.coords[depth_coords] = (depth_coords, y, self.depth.attrs)

        if 'time' in dss[dts_tmp_label].dims:
            assert time_ref
            if isinstance(time_ref, str):
                time_ref = np.datetime64(time_ref)

            time_rel = (dss.time.data - time_ref) / np.timedelta64(1, 'D')
            time_rel_attrs = {
                'units':       'days',
                'description': 'time after start experiment. Timestamp is halfway the '
                               'aqcuisition'}
            self.coords['duration'] = ('duration', time_rel, time_rel_attrs)
            self.coords['time'] = ('duration', dss.time.data, dss.time.attrs)

        # Construct data array for gef datastore
        btmp_data = dss[dts_tmp_label].data

        if dss[dts_tmp_label].dims == ('x', 'time'):
            dims = (depth_coords, 'duration')
        elif dss[dts_tmp_label].dims == ('x',):
            dims = (depth_coords,)
        elif dss[dts_tmp_label].dims == ('duration',):
            dims = ('time',)
        else:
            dims = tuple()

        self[gef_tmp_label] = (dims, btmp_data, dss[dts_tmp_label].attrs)

    def plot(self, labels=None, xlims=None, ylim=None, ylabel=None, xlabels=None,
             q_label=None, q_conf_bound=False, q_xlim=None, title=None, temp_label=None,
             templim=None):

        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator
        import numpy as np

        zlim = [-50, 2]
        axn = len(xlims) + bool(q_label) + bool(temp_label)
        axi_q = axn - 1
        axi_temp = axn - 2

        f, axs = plt.subplots(1, axn, figsize=(16.53, 11.69), dpi=100)

        if title:
            f.suptitle(title + ': ' + ' - '.join(self.PROJECTID + [self.TESTID, self.dts_t0_stamp]))

        else:
            f.suptitle(' - '.join(self.PROJECTID + [self.TESTID, self.dts_t0_stamp]))

        mv = self.attrs['maaiveld (m+NAP)']

        for plot_lab, ax in zip(labels, axs):
            if plot_lab not in self:
                ax.set_xlabel(plot_lab)
                ax.set_xlim(xlims[plot_lab])
                ax.set_ylim(zlim)
                ax.yaxis.set_ticks(np.arange(zlim[0], zlim[1], 5.))
                ax.minorticks_on()
                ax.tick_params(axis='y', which='minor', direction='out')
                ax.xaxis.set_minor_locator(MultipleLocator(1))
                ax.grid(which='minor', linewidth=0.3, c='lightgrey')
                ax.grid(which='major', linewidth=0.4, c='grey')
                continue

            print('about to print', plot_lab, 'to', ax)
            y = self.depth.data + mv
            x = self[plot_lab].data
            ax.plot(x, y, linewidth=0.5, c='black')

            # swap_axes(ax)
            ax.yaxis.set_ticks(np.arange(zlim[0], zlim[1], 5.))
            ax.minorticks_on()
            ax.tick_params(axis='y', which='minor', direction='out')
            ax.tick_params(axis='x', which='minor', bottom='off')
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.grid(which='minor', linewidth=0.3, c='lightgrey', axis='y')
            ax.grid(which='major', linewidth=0.4, c='grey')
            ax.set_xlim(xlims[plot_lab])
            ax.set_ylim(zlim)
            ax.set_ylabel('')
            xlabel = plot_lab + ' (' + self[plot_lab].units + ')'
            ax.set_xlabel(xlabel)
            ax.axhline(y=mv, linestyle='--', linewidth=0.6, c='black')

        axs[0].set_ylabel(r'$z$ (m+NAP)')

        if q_label:
            axs[axi_q].grid(which='minor', linewidth=0.3, c='lightgrey')
            axs[axi_q].grid(which='major', linewidth=0.4, c='grey')
            axs[axi_q].axhline(y=mv, linestyle='--', linewidth=0.6, c='black')

            y = self[q_label].depth_dts.data + mv
            q50 = self[q_label].data

            if q_conf_bound:
                qll = q_label + q_conf_bound[0]
                qlu = q_label + q_conf_bound[1]

                ql = self[qll].data
                qu = self[qlu].data

                axs[axi_q].fill_betweenx(y, ql, qu, label='95% confidence interval',
                                         facecolor='cyan',
                                         linestyle='None')
            axs[axi_q].plot(q50, y, linewidth=0.5, label=q_label, c='black')

            axs[axi_q].yaxis.set_ticks(np.arange(zlim[0], zlim[1], 5))
            axs[axi_q].minorticks_on()
            axs[axi_q].tick_params(axis='y', which='minor', direction='out')
            axs[axi_q].tick_params(axis='x', which='minor', bottom='off')
            axs[axi_q].xaxis.set_minor_locator(MultipleLocator(1))
            axs[axi_q].set_ylim(zlim)
            axs[axi_q].set_xlim(q_xlim)
            axs[axi_q].set_xlabel('Specific discharge (m/day)')
            axs[axi_q].legend(fontsize='small')

        if temp_label:
            axs[axi_temp].grid(which='minor', linewidth=0.3, c='lightgrey')
            axs[axi_temp].grid(which='major', linewidth=0.4, c='grey')
            axs[axi_temp].axhline(y=mv, linestyle='--', linewidth=0.6, c='black')

            y = self[temp_label].depth_dts.data + mv
            btmp = self[temp_label].data
            axs[axi_temp].plot(btmp, y, linewidth=0.5, label=temp_label, c='black')

            axs[axi_temp].yaxis.set_ticks(np.arange(zlim[0], zlim[1], 5))
            axs[axi_temp].minorticks_on()
            axs[axi_temp].tick_params(axis='y', which='minor', direction='out')
            axs[axi_temp].tick_params(axis='x', which='minor', bottom='off')
            axs[axi_temp].xaxis.set_minor_locator(MultipleLocator(1))
            axs[axi_temp].set_ylim(zlim)
            axs[axi_temp].set_xlim(templim)
            axs[axi_temp].set_xlabel('Background temperature ($^\circ$C)')

        f.tight_layout()
        return f


def open_datastore(filename_or_obj, group=None, decode_cf=True,
                   mask_and_scale=None, decode_times=True, autoclose=False,
                   concat_characters=True, decode_coords=True, engine=None,
                   chunks=None, lock=None, cache=None, drop_variables=None,
                   backend_kwargs=None, **kwargs):
    """Load and decode a datastore from a file or file-like object.
    Parameters
    ----------
    filename_or_obj : str, Path, file or xarray.backends.*DataStore
        Strings and Path objects are interpreted as a path to a netCDF file
        or an OpenDAP URL and opened with python-netCDF4, unless the filename
        ends with .gz, in which case the file is gunzipped and opened with
        scipy.io.netcdf (only netCDF3 supported). File-like objects are opened
        with scipy.io.netcdf (only netCDF3 supported).
    group : str, optional
        Path to the netCDF4 group in the given file to open (only works for
        netCDF4 files).
    decode_cf : bool, optional
        Whether to decode these variables, assuming they were saved according
        to CF conventions.
    mask_and_scale : bool, optional
        If True, replace array values equal to `_FillValue` with NA and scale
        values according to the formula `original_values * scale_factor +
        add_offset`, where `_FillValue`, `scale_factor` and `add_offset` are
        taken from variable attributes (if they exist).  If the `_FillValue` or
        `missing_value` attribute contains multiple values a warning will be
        issued and all array values matching one of the multiple values will
        be replaced by NA. mask_and_scale defaults to True except for the
        pseudonetcdf backend.
    decode_times : bool, optional
        If True, decode times encoded in the standard NetCDF datetime format
        into datetime objects. Otherwise, leave them encoded as numbers.
    autoclose : bool, optional
        If True, automatically close files to avoid OS Error of too many files
        being open.  However, this option doesn't work with streams, e.g.,
        BytesIO.
    concat_characters : bool, optional
        If True, concatenate along the last dimension of character arrays to
        form string arrays. Dimensions will only be concatenated over (and
        removed) if they have no corresponding variable and if they are only
        used as the last dimension of character arrays.
    decode_coords : bool, optional
        If True, decode the 'coordinates' attribute to identify coordinates in
        the resulting dataset.
    engine : {'netcdf4', 'scipy', 'pydap', 'h5netcdf', 'pynio', 'pseudonetcdf'}, optional
        Engine to use when reading files. If not provided, the default engine
        is chosen based on available dependencies, with a preference for
        'netcdf4'.
    chunks : int or dict, optional
        If chunks is provided, it used to load the new dataset into dask
        arrays. ``chunks={}`` loads the dataset with dask using a single
        chunk for all arrays.
    lock : False, True or threading.Lock, optional
        If chunks is provided, this argument is passed on to
        :py:func:`dask.array.from_array`. By default, a global lock is
        used when reading data from netCDF files with the netcdf4 and h5netcdf
        engines to avoid issues with concurrent access when using dask's
        multithreaded backend.
    cache : bool, optional
        If True, cache data loaded from the underlying datastore in memory as
        NumPy arrays when accessed to avoid reading from the underlying data-
        store multiple times. Defaults to True unless you specify the `chunks`
        argument to use dask, in which case it defaults to False. Does not
        change the behavior of coordinates corresponding to dimensions, which
        always load their data from disk into a ``pandas.Index``.
    drop_variables: string or iterable, optional
        A variable or list of variables to exclude from being parsed from the
        dataset. This may be useful to drop variables with problems or
        inconsistent values.
    backend_kwargs: dictionary, optional
        A dictionary of keyword arguments to pass on to the backend. This
        may be useful when backend options would improve performance or
        allow user control of dataset processing.
    Returns
    -------
    dataset : Dataset
        The newly created dataset.
    See Also
    --------
    read_xml_dir
    """

    xr_kws = inspect.signature(xr.open_dataset).parameters.keys()

    ds_kwargs = {k: v for k, v in kwargs.items() if k not in xr_kws}

    ds_xr = xr.open_dataset(
        filename_or_obj, group=group, decode_cf=decode_cf,
        mask_and_scale=mask_and_scale, decode_times=decode_times, autoclose=autoclose,
        concat_characters=concat_characters, decode_coords=decode_coords, engine=engine,
        chunks=chunks, lock=lock, cache=cache, drop_variables=drop_variables,
        backend_kwargs=backend_kwargs)

    ds = DataStore(data_vars=ds_xr.data_vars,
                   coords=ds_xr.coords,
                   attrs=ds_xr.attrs,
                   **ds_kwargs)
    return ds


def read_gef(fp):
    """
    Read a GEF file
    :param fp: Provide the file path
    :return: geotechnicalprofile.DataStore

    TODO: put read_gef in the init. reading self.geffilehandle
    """
    def read_raw_file(attrs, f):
        """Internal function to read the raw content of the GEF file. attrs is updated in the
        while loop"""

        s = str(f.readline(), 'ASCII')
        key_val(s, k='GEFID', v=supported_version, dtyped=dtyped, out=attrs)
        while True:
            s = str(f.readline(), 'ASCII', 'ignore')
            if s[1:4] == 'EOH':  # End of header
                break

            key_val(s, dtyped=dtyped, out=attrs)
        X = np.loadtxt(f)
        return X

    attrs = {}

    if not hasattr(fp, 'read'):
        with open(fp, 'rb') as f:
            X = read_raw_file(attrs, f)

    else:
        X = read_raw_file(attrs, fp)

    del attrs['DATAFORMAT']
    del attrs['OS']

    assert X.shape == (attrs['LASTSCAN'], attrs['COLUMN'])
    del attrs['LASTSCAN']
    del attrs['COLUMN']

    for col_info in attrs['COLUMNVOID']:
        mask = np.isclose(X[:, col_info[0] - 1], col_info[1])
        X[mask, col_info[0] - 1] = np.nan
    del attrs['COLUMNVOID']

    attrs['x (m)'] = attrs['XYID'][1]
    attrs['y (m)'] = attrs['XYID'][2]
    assert attrs['XYID'][0] == 31000  # ensure it is RDS
    del attrs['XYID']

    attrs['maaiveld (m+NAP)'] = attrs['ZID'][1]
    assert attrs['ZID'][0] == 31000  # ensure it is NAP
    del attrs['ZID']

    for item in attrs['MEASUREMENTVAR']:
        key = item[3] + ' (' + item[2] + ')'
        attrs[key] = item[1]

    del attrs['MEASUREMENTVAR']

    for item in attrs['MEASUREMENTTEXT']:
        key = item[2] + ' (' + item[0] + ')'
        attrs[key] = item[1]

    del attrs['MEASUREMENTTEXT']

    labels = [item[2] for item in attrs['COLUMNINFO']]
    units = [item[1] for item in attrs['COLUMNINFO']]
    del attrs['COLUMNINFO']

    hellingsmeter_aanwezig = 'gecorrigeerde diepte' in labels

    if hellingsmeter_aanwezig:
        diepte = 'gecorrigeerde diepte'

    else:
        diepte = 'sondeerlengte'

    icol_diepte = labels.index(diepte)
    dim_coord = attrs['maaiveld (m+NAP)'] - X[:, icol_diepte]
    coords = {
        'z':     ('depth', dim_coord, {
            'units':       'm+NAP',
            'description': 'Depth w.r.t. reference level',
            'methode':     diepte}),
        'depth': ('depth', - X[:, icol_diepte], {
            'units':       'm+maaiveld',
            'description': 'Depth w.r.t. surface level',
            'methode':     diepte})}

    data = {}
    for label, unit, x_item in zip(labels, units, X.T):
        data[label.lower()] = (r'depth', x_item, {
            'units': unit})

    _date = attrs['STARTDATE'] + attrs['STARTTIME']
    date = [int(item) for item in _date]
    attrs['date'] = str(pd.to_datetime(dt(*date), utc=True))
    del attrs['STARTDATE']
    del attrs['STARTTIME']

    ds = DataStore(geffile=fp, data_vars=data, coords=coords, attrs=attrs)

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

        except ValueError:
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
