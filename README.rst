========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis|
        |
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|

.. |docs| image:: https://readthedocs.org/projects/python-geotechnical-profile/badge/?style=flat
    :target: https://readthedocs.org/projects/python-geotechnical-profile
    :alt: Documentation Status


.. |travis| image:: https://travis-ci.org/bdestombe/python-geotechnical-profile.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/bdestombe/python-geotechnical-profile

.. |version| image:: https://img.shields.io/pypi/v/geotechnicalprofile.svg
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/geotechnicalprofile

.. |commits-since| image:: https://img.shields.io/github/commits-since/bdestombe/python-geotechnical-profile/v0.1.0.svg
    :alt: Commits since latest release
    :target: https://github.com/bdestombe/python-geotechnical-profile/compare/v0.1.0...master

.. |wheel| image:: https://img.shields.io/pypi/wheel/geotechnicalprofile.svg
    :alt: PyPI Wheel
    :target: https://pypi.python.org/pypi/geotechnicalprofile

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/geotechnicalprofile.svg
    :alt: Supported versions
    :target: https://pypi.python.org/pypi/geotechnicalprofile

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/geotechnicalprofile.svg
    :alt: Supported implementations
    :target: https://pypi.python.org/pypi/geotechnicalprofile


.. end-badges

Load, manage, and plot geotechnical data, including GEF and DTS measurements

* Free software: BSD 3-Clause License

Installation
============

::

    pip install geotechnicalprofile

Documentation
=============


https://python-geotechnical-profile.readthedocs.io/


Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
