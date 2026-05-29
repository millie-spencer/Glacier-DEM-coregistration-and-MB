# Glacier-DEM-coregistration-and-MB

This repository contains Jupyter notebook workflows for geodetic glacier mass balance analysis, developed for the Nevados de Chillán Volcanic Complex in Ñuble, Chile. The pipeline processes multi-source DEMs spanning 1954–2025 and has been updated in response to peer review to incorporate open-source co-registration, heteroscedastic uncertainty propagation, and expanded temporal coverage via ASTER and Pléiades DEMs.

**Notebooks:**
01. DEM preparation and alignment — SRTM download, geoid correction, reprojection to common grid
02. DEM co-registration — Nuth & Kääb (2011) via demcoreg; stable terrain masking using 1975, 2000, and 2019 glacier inventories
03. DEM differencing and uncertainty propagation — elevation change, volume change, and mass balance with heteroscedastic error modelling (xDEM; Hugonnet et al. 2022) and spatial uncertainty propagation (Rolstad et al. 2009)
04. Hydroclimate analysis — Sen's slope, Mann–Kendall trend tests, and segmented regression for precipitation, temperature, and streamflow
05. Geospatial analysis — pixel-level and glacier-level correlations between elevation change and topographic variables (elevation, slope, aspect), outline sensitivity, hypsometric analysis, and spatial autocorrelation (Moran's I)

**DEMs included:**
- Instituto Geográfico Militar (IGM) 1954 topographic map DEM
- SRTM 2000 (downloaded via earthaccess)
- UAV DEMs 2024 (DJI Mavic 3E RTK, Cerro Blanco and Las Termas subcomplexes)
- Pléiades 2025 (full complex, 1m native resolution; DGA / J. Berkhoff, FAU Erlangen-Nürnberg)
- ASTER DEMs 2008 and 2018 (bias-corrected 30m; Hugonnet et al. 2021, available on request)

**Glacier outlines included:**
- NdC 1975 (Landsat-derived)
- DGA 2000 and 2019 (Dirección General de Aguas)

<p align="center">
  <img src="https://github.com/user-attachments/assets/de63be35-a4ec-477f-ae55-b39738f68f3a" width="400"/>
  <br>
  <em>UAV image of Volcán Chillán Viejo facing north, Nevados de Chillán Volcanic Complex</em>
</p>

To follow our workflow or adapt it for your own study area, clone the repository:

```bash
mkdir glacier-mb && cd glacier-mb
git clone https://github.com/millie-spencer/Glacier-DEM-coregistration-and-MB.git
```

Then create the conda environment:

```bash
conda env create -f glacier-env.yml
conda activate glacier-env
```