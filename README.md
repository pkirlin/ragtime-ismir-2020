To run on your own machine:
- Clone/download code
- `$ cd ragtime-ismir-2020`
- `$ pipenv install`
- `$ pipenv shell`
- `$ cd src`
- `$ PYTHONPATH=. python ismir-code/verify-dataset.py`  

To run in binder:
- Click to open the binder: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pkirlin/ragtime-ismir-2020/master?urlpath=lab)
- Open a terminal.
- `cd src`
- `PYTHONPATH=.  python ismir-code/verify-dataset.py`  (to run some sanity checks)
- `PYTHONPATH=.  python ismir-code/experiment-121-pattern.py` (to run the 121 pattern experiment)
- `PYTHONPATH=.  python ismir-code/experiment-all-patterns.py` (to find all syncopated patterns)
- `PYTHONPATH=.  python ismir-code/experiment-transitions.py` (to run the transitions-between-measures experiment)
