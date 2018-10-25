
1. Load your first measurement files
====================================

This notebook is located in
https://github.com/bdestombe/python-dts-calibration/tree/master/examples/notebooks

.. code:: ipython3

    import os
    import glob
    
    from geotechnicalprofile.gp_datastore import read_gef

The data files are located in ``./python-dts-calibration/tests/data``

.. code:: ipython3

    try:
        # this file is excecuted as script
        wd = os.path.dirname(os.path.realpath(__file__))
        
    except:
        # Excecuted from console. pwd = ./docs
        wd = os.getcwd()
    
    folderpath = os.path.join(wd, '..', '..', 'tests', 'data', 'GEF')
    print(folderpath)


.. parsed-literal::

    /Users/bfdestombe/PycharmProjects/GeotechnicalProfile/python-geotechnical-profile/examples/notebooks/../../tests/data/GEF


.. code:: ipython3

    # Just to show which files are in the folder
    file_ext = '*.GEF'
    filepathlist = sorted(glob.glob(os.path.join(folderpath, file_ext)))
    filenamelist = [os.path.basename(path) for path in filepathlist]
    
    for fn in filenamelist:
        print(fn)
        
    filepath = filepathlist[0]


.. parsed-literal::

    67059_DKM002.GEF
    67059_DKM005.GEF
    67059_DKM006.GEF
    67059_DKM011.GEF
    67059_DKM022.GEF
    67059_DKP001.GEF


.. code:: ipython3

    filepath




.. parsed-literal::

    '/Users/bfdestombe/PycharmProjects/GeotechnicalProfile/python-geotechnical-profile/examples/notebooks/../../tests/data/GEF/67059_DKM002.GEF'



.. code:: ipython3

    ds = read_gef(filepath)

.. code:: ipython3

    print(ds)


.. parsed-literal::

    <xarray.DataStore>
    Dimensions:                  (depth: 2330)
    Coordinates:
        z                        (depth) float64 -1.76 -1.78 -1.8 ... -48.16 -48.18
      * depth                    (depth) float64 -0.0 -0.02 ... -46.4 -46.42
    Data variables:
        sondeerlengte            (depth) float64 0.0 0.02 0.04 ... 46.54 46.56 46.58
        puntdruk                 (depth) float64 nan 0.56 0.79 ... 59.8 60.2 57.19
        lokale wrijving          (depth) float64 nan 0.019 0.023 ... nan nan nan
        helling                  (depth) float64 nan 1.6 1.633 1.667 ... 7.1 7.1 7.1
        helling x                (depth) float64 nan -0.9 -0.6 ... -6.4 -6.4 -6.4
        helling y                (depth) float64 nan -1.3 -0.8 ... -3.2 -3.2 -3.2
        snelheid                 (depth) float64 nan 0.0 1.4 1.4 ... 0.7 0.6 0.6 0.6
        wrijvingsgetal           (depth) float64 nan 2.554 3.006 ... nan nan nan
        tijd                     (depth) float64 nan 74.0 ... 3.978e+03 3.981e+03
        gecorrigeerde diepte     (depth) float64 0.0 0.02 0.03999 ... 46.4 46.42
        Kh                       (depth) float64 nan 0.0528 0.04737 ... nan nan nan
        Hydraulische weerstand1  (depth) float64 -0.0 -4.823 -5.201 ... -0.0 -0.0
        Rv weerstandslaag2       (depth) float64 -0.0 -241.1 -260.0 ... -0.0 -0.0
        Lithologie               (depth) int64 1 1 1 1 1 1 1 1 1 ... 1 1 1 1 1 1 1 1
        Fijne deeltjes < 75 um   (depth) float64 0.0 100.0 100.0 ... 0.0 0.0 0.0
    Attributes:
        GEFID:                                         [1, 1, 0]
        FILEOWNER:                                     Wiertsema & Partners
        FILEDATE:                                      ['2017', '9', '8']
        PROJECTID:                                     ['CPT', '67059', '1']
        COMPANYID:                                     ['Wiertsema & Partners', '...
        TESTID:                                        DKM002
        REPORTCODE:                                    ['GEF-CPT-Report', '1', '1...
        x (m):                                         134992.0
        y (m):                                         474008.0
        maaiveld (m+NAP):                              -1.76
        nom. opp. conuspunt (mm2):                     1500.0
        nom. opp. wrijvingsmantel (mm2):               22530.0
        Net. opp. quotint (-):                         0.67
        afstand conus tot midden kleefmantel (mm):     100.0
        wrijvings-meter aanwezig (-):                  1.0
        waterdruk-meter u1 aanwezig (-):               0.0
        waterdruk-meter u2 aanwezig (-):               0.0
        waterdruk-meter u3 aanwezig (-):               0.0
        helling-meter aanwezig (-):                    1.0
        sondeermethode (-):                            4.0
        einddiepte (m):                                46.42281
        stopcriterium (-):                             1.0
        offset conus voor de meting (MPa):             14.734782
        offset conus na de meting (MPa):               14.732331
        offset wrijving voor de meting (MPa):          0.0
        offset wrijving na de meting (MPa):            0.0
        offset helling voor de meting (graden):        0.0
        offset helling na de meting (graden):          0.0
        offset helling NZ voor de meting (graden):     24.0
        offset helling NZ na de meting (graden):       23.4375
        offset helling OW voor de meting (graden):     27.1875
        offset helling OW na de meting (graden):       26.8125
        opdrachtgever (1):                             0
        projectnaam (2):                               Meetnetwerk met glasvezelk...
        projectplaats (3):                             Weesp
        conustype (4):                                 SUB-15/080801
        sondeerapparaat (5):                           Hyson
        norm waaraan deze sondering moet voldoen (6):  Norm: NEN-EN-ISO 22476-1; ...
        vast horizontaal referentievlak (9):           maaiveld
        methode verticale positiebepaling (42):        MDGZ
        methode locatiebepaling (43):                  LDGZ
        date:                                          2017-09-04 09:54:00+00:00
    \n#FILEOWN...
        lithologie_attrs:                              ((1, 'Water voerend'), (2,...

