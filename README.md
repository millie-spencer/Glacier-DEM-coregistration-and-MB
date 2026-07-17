# Seven Decades of Glacier Loss on the Nevados de Chillán Volcanic Complex, Chile

This repository contains the Jupyter notebook workflows used in Spencer et al. (in revision), *Seven Decades of Glacier Loss on the Nevados de Chillán volcanic complex, Chile*, submitted to *The Cryosphere* (egusphere-2025-5795).

The pipeline processes multi-source DEMs spanning 1954–2025 to produce geodetic mass balance estimates for all 28 glaciers on the Nevados de Chillán complex. The workflow was updated in response to peer review to incorporate open-source co-registration, heteroscedastic uncertainty propagation, and expanded temporal coverage via ASTER and Pléiades DEMs.

<p align="center">
  <img src="https://github.com/user-attachments/assets/de63be35-a4ec-477f-ae55-b39738f68f3a" width="400"/>
  <br>
  <em>UAV image of Volcán Chillán Viejo facing north, Nevados de Chillán Volcanic Complex</em>
</p>


## Notebooks

1. **DEM preparation and alignment** — SRTM download, geoid correction, reprojection to common grid
2. **DEM co-registration** — Nuth & Kääb (2011) via demcoreg; stable terrain masking using the union of 1975, 2000, and 2019 glacier inventories
3. **DEM differencing and uncertainty propagation** — elevation change, volume change, and mass balance using a time-weighted mean area approach (Cogley et al., 2011); heteroscedastic per-pixel error modelling (xDEM; Hugonnet et al., 2022) and spatial uncertainty propagation (Rolstad et al., 2009)
4. **Hydroclimate analysis** — Sen's slope and Mann–Kendall trend tests for annual and summertime (JFM) precipitation, temperature, and streamflow; E-Divisive changepoint detection and segmented regression
5. **Geospatial analysis** — pixel-level and glacier-level correlations between elevation change and topographic variables (elevation, slope, aspect); outline sensitivity analysis; hypsometric dH analysis; and spatial autocorrelation (Global and Local Moran's I / LISA)

## DEMs

| Source | Type | Date | Resolution |
|--------|------|------|------------|
| Instituto Geográfico Militar (IGM) | Topographic map | 1954 | 30 m |
| SRTM | C-band radar | February 2000 | 30 m |
| ASTER | Stereo optical | 1 March 2018 | 30 m |
| UAV (DJI Mavic 3E RTK) | RGB photogrammetry | 12–14 March 2024 | 7.82 cm (CB) / 16.70 cm (LT) |
| Pléiades-1A | Stereo optical | 17 April 2025 | 1 m native, resampled to 30 m |

Note: ASTER DEMs are bias-corrected 30 m products from Hugonnet et al. (2021), available on request. The Pléiades DEM was obtained from the Dirección General de Aguas (DGA) under the Chilean Transparency Act.

## Glacier Outlines

- 1975 inventory (DGA, 2011)
- 2000 inventory (DGA, 2014)
- 2019 inventory (DGA, 2022)

## Citation

If you use this code, please cite:

> Spencer, M., Fernandez, A., Tyrrell, E., Clasing, R., Muñoz, E., Mendoza, P. A., Berkhoff, J., and Molotch, N. P.: Seven Decades of Glacier Loss on the Nevados de Chillán volcanic complex, Chile, *The Cryosphere*, in revision, 2025.

Code archived at Zenodo: https://doi.org/10.5281/zenodo.17664874

## Getting Started

Clone the repository:

```bash
git clone https://github.com/millie-spencer/Seven-Decades-Glacier-Loss-Nevados-Chillan.git
cd Seven-Decades-Glacier-Loss-Nevados-Chillan
```

Create and activate the conda environment:

```bash
conda env create -f glacier-env.yml
conda activate glacier-env
```

## Dependencies

Key packages: `rasterio`, `xdem`, `geopandas`, `rasterstats`, `demcoreg`, `pygeotools`, `contextily`, `pyproj`, `scipy`, `matplotlib`

See `glacier-env.yml` for the full environment specification.

## License

See LICENSE file.


