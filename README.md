# Truthsayer
==============================

## Problem

Detecting real/fake news & assessing degree of political bias is difficult.  This project is an initial attempt at using BERT as a discriminator.  For details see discussion [In Depth](docs/InDepth.md)

# Setup

## Requirements
* conda
* pip3
* python 3.7
* use pip install -r requirements.txt at project root

### Setting up newspaper3k https://newspaper.readthedocs.io/en/latest/

* brew install libxml2 libxslt
* brew install libtiff libjpeg webp little-cms2
* pip3 install newspaper3k

## Data

### Scraping News Sites

Scrape both known sites and zombie sites from mass move

./src/data/scrape.py   <- scrape from known Sites
./src/data/scrape_zombie.py <- scrape zombie sites from mass move
./src/data/zombie_sites <- zombie sites from mass move

Note:  these scripts expect to be run from within ./src/data/

### Creating data set

make data  // run from project root

## Models

src/models should be run from that directory

* src/models/train_model.py - trains a bert model
* src/models/predict_model.py - runs some predictions

## Project Organization
see https://github.com/drivendata/cookiecutter-data-science for details
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
