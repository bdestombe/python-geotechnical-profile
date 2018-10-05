# coding=utf-8
import numpy as np
import xarray as xr
from scipy.optimize import fsolve


def check_unit(ds, label, unit):
    assert ds[label].attrs['units'] == unit, 'not the correct unit'
    pass


def hydraulic_conductance(ds):
    # Horizontal
    check_unit(ds, 'puntdruk', 'MPa')
    check_unit(ds, 'wrijvingsgetal', '%')

    key = 'Kh'
    f = 0.825
    ds[key] = (('depth',),
               ds['puntdruk'] / (f * np.exp(ds['wrijvingsgetal']) ** 3),
               {'units': 'm per dag'})  # m/d

    mask = ds['wrijvingsgetal'] > 2.
    ds[key][mask] = ds['puntdruk'][mask] / (
        f * np.exp(ds['wrijvingsgetal'][mask]))  # m/d
    ds[key][mask][ds[key][mask] < 1e-4] = 1e-4


def hydraulic_resistance(ds):
    key = 'Hydraulische weerstand1'

    z = ds.depth.data
    dz = np.hstack((z[1:] - z[:-1], z[-1] - z[-2]))

    dz_formula = 0.02

    colum_A = ds['Kh']
    colum_K = np.zeros(ds.depth.shape)
    mask = -xr.ufuncs.log10(colum_A) > 0
    colum_K[mask] = xr.ufuncs.log10(colum_A[mask] ** 1.6)
    colum_K[colum_K < -7.5] = -7.5
    colum_K[colum_K < -5.] = 0.7 * colum_K[colum_K < -5.]

    mask = np.logical_not(np.isclose(0., colum_K))
    colum_K[mask] = 1. / np.exp(colum_K[mask]) / 1.6

    colum_K *= dz / dz_formula  # correct for measurement interval

    ds[key] = (('depth',), colum_K, {'units': 'day/2cm'})

    key2 = 'Rv weerstandslaag2'
    ds[key2] = (('depth',), 50 * colum_K, {'units': 'day/m'})


def lithology(ds):
    lithologie_attrs = ((1, 'Water voerend'), (2, 'Water remmend'), (3, 'Organic'))
    ds.attrs.update(lithologie_attrs=lithologie_attrs)
    key = 'Lithologie'

    ds[key] = (('depth',), np.ones(ds.depth.shape, dtype=int))

    mask = ds['Hydraulische weerstand1'] > 6.
    ds[key][mask] = 3

    mask = xr.ufuncs.logical_and(ds['Hydraulische weerstand1'] > 0.,
                                 ds['Hydraulische weerstand1'] < 6.)
    ds[key][mask] = 2


def veenindex(ds):
    key = 'Veenindex'
    ds[key] = (('depth',), np.zeros(ds.depth.shape), {'units': '-'})

    mask = ds['wrijvingsgetal'] > 4.
    ds[key][mask] = ds['wrijvingsgetal'][mask] ** 2


def fijnedeeltjes(ds):
    key = 'Fijne deeltjes < 75 um'
    ds[key] = (('depth',), np.zeros(ds.depth.shape), {'units': '%'})

    Ic = ((3.47 - np.log10(ds['puntdruk'])) ** 2 +
          (np.log10(ds['wrijvingsgetal']) + 1.22) ** 2) ** 0.5

    mask = np.logical_and(Ic > 1.31, Ic <= 2.5)
    ds[key][mask] = 42.0 * Ic[mask] - 55.0 + 10 * np.sin(
        ((Ic[mask] - 2.5) / 1.19) * np.pi)

    mask = np.logical_and(Ic > 2.5, Ic <= 3.1)
    ds[key][mask] = 83.3 * Ic[mask] - 158.3

    mask = Ic > 3.1
    ds[key][mask] = 100.

    mask = np.logical_and(
        np.logical_and(Ic > 1.31, Ic <= 2.36), ds['wrijvingsgetal'] < 0.6)
    ds[key][mask] = 5 * ds['wrijvingsgetal'][mask]


def estimate_wl_between(elevation, waterpressure, zlim):
    zmask = np.logical_and(elevation > zlim[0], elevation < zlim[1],
                           np.logical_not(np.isnan(waterpressure)).data)

    wl = fsolve(
        lambda wlf: (-0.009804 * (elevation[zmask] - wlf) - waterpressure[zmask]) ** 2,
        0)[0]
    return wl
