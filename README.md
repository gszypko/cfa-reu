# cfa-reu, a repository for processing and plotting PSP data
### by Greg Szypko

## Acknowledgements
This work was completed as part of the Smithsonian Astrophysical Observatory NSF-REU in solar physics during the summer of 2019, grant number AGS-1560313. None of this would have been possible without mentorship from Dr. Kristoff Paulson, Dr. Michael Stevens, Dr. Anthony Case, Dr. Kelly Korreck, and Dr. Tatiana Niembro Hernandez.

## Overview of Workflow
* Define the path names you plan to use for different data files by modifying `pspconstants.py`:
  * `path` is where you are storing original SPC .cdf files
  *  `mag_path` is where you are storing original FIELDS magnetometer .cdf files
  * `precomp_path` is where pre-processed data files will be stored and read from
  * `default_output_path` is the default output path used for plots in `angslice.py`
  * `known_transients` is a list of `datetime.datetime` tuples containing the beginning and end times of transient events you wish to exclude  
* Run `precompute.py` to preprocess data from the .cdf files. This packages most variables into .npy files, saved to `precomp_path`, which can then be read by the plotting scripts.
  * Radial velocity (`v_r`), proton density (`n_p`), temperature (`temp`), radial magnetic field (`b_r_spc`), and magnetic field magnitude (`b_mag_spc`) are all saved in both unfiltered and filtered form. Filtering is accomplished with a time-domain median filter. To cut down on memory usage, this filtering is split out into (defualt) 16 files, numbered from 000 to 015 at the end of the filename. See `modular_median_filter()` in `psplib.py` for details.
* Once `precompute.py` runs, combine the split out filtered files using `bcombine.py`.
* From here, you should be set to plot your data using `angslice.py`, `radial.py`, `streamfind.py`, or `linfit.py`
