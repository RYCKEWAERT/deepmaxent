# DeepMaxent - A neural networks using maximum entropy principle for Species Distribution Modelling

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
  <img src="images/deepmaxent.png" alt="Project logo" width="300">
  <h2 align="center">DeepMaxent</h2>
  <p align="center">A neural networks using maximum entropy principle for Species Distribution Modelling developped in B-CUBED project</p>
  <a href="https://github.com/RYCKEWAERT/deepmaxent">View project</a>
  ·
  <a href="https://github.com/RYCKEWAERT/deepmaxent/issues">Report Bug</a>
  ·
  <a href="https://github.com/RYCKEWAERT/deepmaxent/issues">Request Feature</a>
  <h1></h1>
</div>



## Objective
This repository aims to document the development and advancements of the DeepMaxent algorithm for species distribution modelling as part of the B-CUBED project: https://b-cubed.eu

## An important note
At this stage, the codes in this repository are still under active development. As the project progresses, all the codes will be properly documented and referenced in this section. Additionally, a paper comparing the different results obtained from these codes is submitted. Stay tuned for updates and further advancements in the development of the project.


## Author
Maxime RYCKEWAERT (Inria)

## About the project
The B-CUBED (Biodiversity Big Data Cube) project is a European initiative aimed at standardising the way in which biodiversity data is stored. The main objective is to standardise biodiversity data to facilitate access, interoperability and use by researchers, policy-makers and the public.

## Deep-Learning algorithms for Species Distribution Modelling
Deep learning models have become increasingly prominent in the field of species distribution modelling. These models are capable of processing vast amounts of biodiversity data, effectively capturing the intricate, non-linear relationships between various environmental factors and the presence or absence of species. 


## Datasets

### A use case (Belgium, 2010) for species classification using B-CUBED data 


#### Biodiversity Data
This dataset is a typical biodiversity dataset from the B-CUBED project in Belgium. It represents a subset from the year 2010, extracted from a more comprehensive dataset. The data is organized into spatial cubes to facilitate detailed biodiversity analysis for that year. For more information and access to the full dataset, please refer to the following resources: 
- https://www.gbif.org/occurrence/download/0096919-240321170329656
- https://doi.org/10.15468/dl.e3j5kv.

#### Bioclimatic Rasters 

This dataset consists of 19 bioclimatic rasters obtained from the WorldClim and CHELSA databases. The rasters represent various environmental factors such as temperature, precipitation, and altitude. 

Karger, D.N., Conrad, O., Böhner, J., Kawohl, T., Kreft, H., Soria-Auza, R.W., Zimmermann, N.E., Linder, P., Kessler, M. (2017). Climatologies at high resolution for the Earth land surface areas. Scientific Data. 4 170122. https://doi.org/10.1038/sdata.2017.122
The full dataset is available : https://chelsa-climate.org/bioclim/ 

### A set from the National Centre for Ecological Analysis and Synthesis (NCEAS)

This dataset is from the openly released recently (Elith et al., 2020), this subset includes presence-only and presence-absence data from six different regions. more details in Elith et al., 2020. If you use this dataset, please cite Elith et al., 2020. 

Elith, J., Graham, C., Valavi, R., Abegg, M., Bruce, C., Ford, A., Guisan, A., Hijmans, R. J., Huettmann, F., Lohmann, L., Loiselle, B., Moritz, C., Overton, J., Peterson, A. T., Phillips, S., Richardson, K., Williams, S., Wiser, S. K., Wohlgemuth, T., & Zimmermann, N. E. (2020). Presence-only and Presence-absence Data for Comparing Species Distribution Modeling Methods. Biodiversity Informatics, 15(2), Article 2. https://doi.org/10.17161/bi.v15i2.13384