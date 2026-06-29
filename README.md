# hsc_dwarfs
Code for deriving morphological parameters from galaxy image stamps.  Originally designed for:
Lazar et al. (2024), MNRAS, 533, 3771
Updated to include functions used for:
Watkins et al. (2026), MNRAS, 547, 472 

Some demo notebooks are available in this home directory, as well as a stamp image with masks applied by-hand to run with those.

 - `surface_photometry_demo.ipynb`: a notebook demonstrating how to derive radial photometric profiles from a masked stamp.  Requires the software found in `dwarfs` directory.
 - `morphological_parameters.ipynb`: a notebook demonstrating how to derive CAS parameters, as well as M20 and Gini coefficient.  Uses an external package called Statmorph that needs to be installed separately.
 - `dwarfs`: folder containing functions to derive surface photometry, as well as related things like background noise characteristics.
 - `masked.fits`: an example stamp used by the two notebooks, with masks applied by-hand and set to -999 counts.
 - `requirements.txt`: the full list of required Anaconda packages, for easier compatibility.

If you don't want to install dependencies individually, create a distinct Anaconda environment for this package by doing the following: $ conda create --name `envname` --file requirements.txt
