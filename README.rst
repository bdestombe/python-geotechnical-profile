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
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
    * - Example notebooks
      - |example-notebooks|

.. |docs| image:: https://readthedocs.org/projects/python-geotechnical-profile/badge/?style=flat
    :target: https://readthedocs.org/projects/python-geotechnical-profile
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/bdestombe/python-geotechnical-profile.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/bdestombe/python-geotechnical-profile

.. |codecov| image:: https://codecov.io/github/bdestombe/python-geotechnical-profile/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/bdestombe/python-geotechnical-profile

.. |version| image:: https://img.shields.io/pypi/v/geotechnicalprofile.svg
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/geotechnicalprofile

.. |wheel| image:: https://img.shields.io/pypi/wheel/geotechnicalprofile.svg
    :alt: PyPI Wheel
    :target: https://pypi.python.org/pypi/geotechnicalprofile

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/geotechnicalprofile.svg
    :alt: Supported versions
    :target: https://pypi.python.org/pypi/geotechnicalprofile

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/geotechnicalprofile.svg
    :alt: Supported implementations
    :target: https://pypi.python.org/pypi/geotechnicalprofile

.. |example-notebooks| image:: https://mybinder.org/badge.svg
   :alt: Interactively run the example notebooks online
   :target: https://mybinder.org/v2/gh/bdestombe/python-geotechnical-profile/master?filepath=examples%2Fnotebooks

.. end-badges

A Python package to load raw DTS files, perform a calibration, and plot the result

* Free software: BSD 3-Clause License

Installation
============

::

    pip install geotechnicalprofile

Current version on pip is vv0.2.0. . This is probably several commits behind this repository. Pip
install directly from github to obtain the most recent changes.

Learn by examples
=================
Interactively run the example notebooks online by clicking the launch-binder button.

Documentation
=============

https://python-geotechnical-profile.readthedocs.io/

Development
===========

To run the all tests run:

.. code-block:: zsh

    tox


To bump version and docs:

.. code-block:: zsh

    git status          # to make sure no unversioned modifications are in the repository
    tox                 # Performes tests and creates documentation and runs notebooks
    git status          # Only notebook related files should be shown
    git add --all       # Add all notebook related files to local version
    git commit -m "Updated notebook examples to reflect recent changes"
    bumpversion patch   # (major, minor, patch)
    git push
    rm -rf build        # Clean local folders (not synced) used for pip wheel
    rm -rf src/*.egg-info
    rm -rf dist/*
    python setup.py clean --all sdist bdist_wheel
    twine upload --repository-url https://upload.pypi.org/legacy/ dist/geotechnicalprofile*
