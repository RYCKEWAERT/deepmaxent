# DeepMaxent

<a name="readme-top"></a>

<p align="center">
  <a href="https://github.com/RYCKEWAERT/deepmaxent/graphs/contributors"><img src="https://img.shields.io/github/contributors/RYCKEWAERT/deepmaxent" alt="GitHub contributors"></a>
  <a href="https://github.com/RYCKEWAERT/deepmaxent/network/members"><img src="https://img.shields.io/github/forks/RYCKEWAERT/deepmaxent" alt="GitHub forks"></a>
  <a href="https://github.com/RYCKEWAERT/deepmaxent/issues"><img src="https://img.shields.io/github/issues/RYCKEWAERT/deepmaxent" alt="GitHub issues"></a>
  <a href="https://github.com/RYCKEWAERT/deepmaxent/blob/main/LICENSE"><img src="https://img.shields.io/github/license/RYCKEWAERT/deepmaxent" alt="License"></a>
  <a href="https://github.com/RYCKEWAERT/deepmaxent/pulls"><img src="https://img.shields.io/github/issues-pr/RYCKEWAERT/deepmaxent" alt="GitHub pull requests"></a>
  <a href="https://github.com/RYCKEWAERT/deepmaxent/stargazers"><img src="https://img.shields.io/github/stars/RYCKEWAERT/deepmaxent" alt="GitHub stars"></a>
  <a href="https://github.com/RYCKEWAERT/deepmaxent/watchers"><img src="https://img.shields.io/github/watchers/RYCKEWAERT/deepmaxent" alt="GitHub watchers"></a>
</p>

<div align="center">
  <img src="images/deepmaxent01.png" alt="Project logo" width="300">
  <h2 align="center">DeepMaxent</h2>
  <p align="center">A neural networks using maximum entropy principle for Species Distribution Modelling developped in B-CUBED project</p>
  <a href="https://github.com/RYCKEWAERT/deepmaxent">View project</a>
  ·
  <a href="https://github.com/RYCKEWAERT/deepmaxent/issues">Report Bug</a>
  ·
  <a href="https://github.com/RYCKEWAERT/deepmaxent/issues">Request Feature</a>
</div>

---


## Overview

**DeepMaxent** is a deep learning algorithm for Species Distribution Modelling (SDM) designed for presence-only data. It combines the maximum entropy principle with neural networks to capture complex, non-linear relationships between species presence and environmental factors.

This approach is particularly suited for handling large-scale biodiversity datasets, enabling researchers to better understand and conserve biodiversity in the face of environmental changes.

## ✨ Key Features

- Deep learning-based SDM using maximum entropy loss
- Handles presence-only biodiversity data
- Captures non-linear species-environment relationships
- Scalable to large datasets

## 🚀 Getting Started

### Tutorial: Costa Rica Plantae 🇨🇷🌿

A comprehensive Jupyter notebook tutorial is available: **[tutorial_deepmaxent.ipynb](tutorial_deepmaxent.ipynb)**

This hands-on tutorial guides you through:
1. Loading and exploring biodiversity occurrence data from GBIF
2. Visualizing species occurrences on interactive maps
3. Processing environmental rasters (WorldClim bioclimatic variables)
4. Preparing training data by aggregating occurrences at raster resolution
5. Building input tensors for the DeepMaxent model

The tutorial uses plant species observations from **Costa Rica** as a case study.
> GBIF data - [DOI: 10.15468/dl.434enu](https://doi.org/10.15468/dl.434enu)
> Chelsa dataset - [CHELSA Bioclim Dataset](https://chelsa-climate.org/bioclim/)



### Tutorial: main_example.py 
Another example script is available: **[main_example.py](main_example.py)**
This script demonstrates how to set up and run the DeepMaxent model using python code instead of notebook. It includes data preprocessing, model training, and evaluation steps.


## 👤 Author

**Maxime RYCKEWAERT** (Cirad)

## 📄 Citation

If you use this code, please cite the following paper:

> *Paper reference coming soon*

## 🔬 About the B-CUBED Project

The [B-CUBED (Biodiversity Building Blocks for Policy)](https://b-cubed.eu) project is a European initiative aimed at standardising biodiversity data storage. Its main objective is to facilitate access, interoperability and use by researchers, policy-makers and the public.

## ⚠️ Note

This repository is under active development. Codes will be progressively documented and a comparative paper is in preparation.

--- 


## 📊 Datasets

### Belgium (2010) - B-CUBED Use Case

**Biodiversity Data**: A subset from 2010, organized into spatial cubes for detailed biodiversity analysis.
- [GBIF Download](https://www.gbif.org/occurrence/download/0096919-240321170329656)
- [DOI: 10.15468/dl.e3j5kv](https://doi.org/10.15468/dl.e3j5kv)

**Bioclimatic Rasters**: 19 variables from CHELSA databases (temperature, precipitation, altitude, etc.)
- Karger, D.N. et al. (2017). Climatologies at high resolution for the Earth land surface areas. *Scientific Data*, 4, 170122. [DOI: 10.1038/sdata.2017.122](https://doi.org/10.1038/sdata.2017.122)
- [CHELSA Bioclim Dataset](https://chelsa-climate.org/bioclim/)

### NCEAS Benchmark Dataset

Presence-only and presence-absence data from six regions (Elith et al., 2020).

> Elith, J. et al. (2020). Presence-only and Presence-absence Data for Comparing Species Distribution Modeling Methods. *Biodiversity Informatics*, 15(2). [DOI: 10.17161/bi.v15i2.13384](https://doi.org/10.17161/bi.v15i2.13384)


### GeoPlant Dataset

GeoPlant: Spatial Plant Species Prediction Dataset. 
A curated dataset of plant species occurrences across Europe, integrated with environmental variables for species distribution modeling.

> Picek, L. et al. (2025). [DOI: 10.52202/079017-4023](https://doi.org/10.52202/079017-4023)