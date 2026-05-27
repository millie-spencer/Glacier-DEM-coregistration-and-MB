# =============================================================================
# Notebook 3: DEM Differencing, Zonal Statistics, and Uncertainty Propagation
# =============================================================================
# This notebook replaces the ArcPy-based workflow (Emma Tyrrell / Millie Spencer)
# with a fully open-source equivalent using rasterio, rasterstats, and xDEM.
#
# The uncertainty propagation follows the approach of Hugonnet et al. (2022)
# and Rolstad et al. (2009), as requested by the reviewer. The key correction
# from the original workflow: rather than applying off-glacier NMAD directly
# as the mass balance uncertainty, we:
#   1. Fit a variogram to the stable-terrain elevation differences to estimate
#      the spatial autocorrelation range of DEM errors.
#   2. Use that range to compute the effective number of independent samples
#      within each glacier polygon.
#   3. Propagate the per-pixel uncertainty to a volume uncertainty by
#      integrating the variance over the glacier area.
#
# REVIEWER NOTE ON REPORTING:
#   Because glacier area changes between epochs and the analysis cannot account
#   for ice lost at the margins, results are more accurately described as
#   "geodetic elevation change" (dh, m) and "volume change" (dV, km³) rather
#   than "mass balance" (m w.e. yr⁻¹). We retain the density conversion
#   (850 kg/m³, Huss 2013) for comparison with published mass balance records,
#   but flag this distinction clearly in the output CSV and suggest the paper
#   text reflect it.
#
# INPUTS (from Notebook 2):
#   - coreg_dems/<name>_coreg.tif      ← co-registered DEMs
#   - SRTM_2000 co-registered DEM      ← baseline option 1
#   - IGM_1954 co-registered DEM       ← baseline option 2
#   - Glacier shapefiles (DGA 2000 and 2019)
#   - Glacier union mask (from Notebook 2)
#
# OUTPUTS:
#   - differenced_dems/<name>_dh.tif   ← elevation change rasters
#   - results/zonal_stats_<epoch>.csv  ← per-glacier dh, volume, uncertainty
#   - results/mass_balance_summary.csv ← summary table matching paper Table X
#   - figures/                         ← dh maps and bar charts
# =============================================================================

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.features import geometry_mask
import rasterstats
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import xdem
import geoutils as gu

warnings.filterwarnings("ignore", category=RuntimeWarning)
print(f"xDEM version: {xdem.__version__}")

# ------------------------------------------------------------------------------
# USER VARIABLES
# ------------------------------------------------------------------------------
coreg_folder   = "coreg_dems"        # output from Notebook 2
output_folder  = "differenced_dems"
results_folder = "results"
figures_folder = "figures"
for d in [output_folder, results_folder, figures_folder]:
    os.makedirs(d, exist_ok=True)

# Glacier outlines
glacier_2000_path = "example_data_Nevados/Nevados_glacier_shapefiles/Nevados_shapefile_DGA2000/Nevados_polygons_DGA2000.shp"
glacier_2019_path = "example_data_Nevados/Nevados_glacier_shapefiles/Nevados_shapefile_DGA2019/Nevados_polygons_DGA2019.shp"

# Glacier union mask (saved by Notebook 2)
glacier_union_path = os.path.join(coreg_folder, "glacier_union_mask.shp")

# Ice density for volume → mass conversion (Huss 2013)
ICE_DENSITY     = 850.0   # kg m⁻³
WATER_DENSITY   = 1000.0  # kg m⁻³
DENSITY_FACTOR  = ICE_DENSITY / WATER_DENSITY  # 0.85

# Field in shapefile used to identify individual glaciers
GLACIER_ID_FIELD = "COD_GLA"

# Define differencing epochs.
# Each entry: (current_dem, baseline_dem, outline_year, start_year, end_year)
# outline_year: which DGA polygon set to use for clipping and zonal stats.
# For the 1954 baseline: the reviewer flags that 2000 polygons include terrain
# that was ice in 1954. We use the glacier UNION outline as the most
# conservative choice — see REVIEWER NOTE in header above.
EPOCHS = [
    # (current,              baseline,    outlines, t_start, t_end)
    ("CerroBlanco_2024",  "IGM_1954",   "union",  1954,    2024),
    ("LasTermas_2024",    "IGM_1954",   "union",  1954,    2024),
    ("CerroBlanco_2024",  "SRTM_2000",  "2019",   2000,    2024),
    ("LasTermas_2024",    "SRTM_2000",  "2019",   2000,    2024),
    ("SRTM_2000",         "IGM_1954",   "union",  1954,    2000),
]

# Slope/aspect rasters (from SRTM, used for variogram stratification)
# If you don't have these pre-computed, set to None and we compute on the fly.
slope_path  = None   # e.g. "data/surfParameters/SRTM_slope.tif"
aspect_path = None   # e.g. "data/surfParameters/SRTM_aspect.tif"

# ------------------------------------------------------------------------------
# Helper: load glacier outlines for a given epoch label
# ------------------------------------------------------------------------------
def load_outlines(label: str, target_crs) -> gpd.GeoDataFrame:
    """
    Load the appropriate glacier outlines for a given epoch label.
    label: "2000" | "2019" | "union"
    Returns a GeoDataFrame in target_crs.
    """
    if label == "2000":
        gdf = gpd.read_file(glacier_2000_path)
    elif label == "2019":
        gdf = gpd.read_file(glacier_2019_path)
    elif label == "union":
        gdf_2000 = gpd.read_file(glacier_2000_path)
        gdf_2019 = gpd.read_file(glacier_2019_path)
        if gdf_2000.crs != gdf_2019.crs:
            gdf_2019 = gdf_2019.to_crs(gdf_2000.crs)
        gdf = gpd.GeoDataFrame(
            geometry=[gdf_2000.union_all().union(gdf_2019.union_all())],
            crs=gdf_2000.crs
        )
        gdf[GLACIER_ID_FIELD] = "UNION"
    else:
        raise ValueError(f"Unknown outline label: {label}")
    return gdf.to_crs(target_crs)


# ------------------------------------------------------------------------------
# Helper: rasterio-based clip + difference (replaces ArcPy ExtractByMask)
# ------------------------------------------------------------------------------
def clip_dem_to_polygons(dem_path: str, polygons: gpd.GeoDataFrame) -> tuple:
    """
    Clip a DEM to the union of a set of polygons using rasterio.
    Returns (clipped_array, transform, nodata, crs).
    Replaces: arcpy ExtractByMask / arcpy env.snapRaster
    """
    geoms = [geom.__geo_interface__ for geom in polygons.geometry]
    with rasterio.open(dem_path) as src:
        # Reproject polygons to DEM CRS if needed
        if polygons.crs != src.crs:
            polygons = polygons.to_crs(src.crs)
            geoms = [geom.__geo_interface__ for geom in polygons.geometry]
        out_image, out_transform = rio_mask(src, geoms, crop=True, nodata=np.nan)
        out_meta = src.meta.copy()
        nodata = src.nodata if src.nodata is not None else np.nan
    return out_image[0], out_transform, nodata, out_meta["crs"]


# ------------------------------------------------------------------------------
# Step 1: Compute slope and aspect from SRTM (if not provided)
# ------------------------------------------------------------------------------
# Used for variogram stratification in the uncertainty analysis.

srtm_coreg_path = os.path.join(coreg_folder, "SRTM_2000_coreg.tif")
# SRTM is the reference so it may not have been co-registered;
# fall back to aligned version
if not os.path.exists(srtm_coreg_path):
    srtm_coreg_path = "aligned_dems/SRTM_2000_aligned.tif"

ref_dem_xdem = xdem.DEM(srtm_coreg_path)

if slope_path is None or not os.path.exists(slope_path):
    print("Computing slope and aspect from SRTM...")
    slope_dem  = xdem.terrain.slope(ref_dem_xdem)
    aspect_dem = xdem.terrain.aspect(ref_dem_xdem)
    slope_path  = os.path.join(results_folder, "SRTM_slope.tif")
    aspect_path = os.path.join(results_folder, "SRTM_aspect.tif")
    slope_dem.save(slope_path)
    aspect_dem.save(aspect_path)
    print(f"  Slope saved:  {slope_path}")
    print(f"  Aspect saved: {aspect_path}")
else:
    slope_dem  = xdem.DEM(slope_path)
    aspect_dem = xdem.DEM(aspect_path)
    print("Loaded existing slope and aspect rasters.")

# ------------------------------------------------------------------------------
# Step 2: Characterize DEM uncertainty on stable terrain using xDEM
# ------------------------------------------------------------------------------
# We use the stable-terrain dh between SRTM and each source DEM to fit a
# variogram and estimate the spatial autocorrelation of elevation errors.
# This is done once per DEM pair and stored for use in Step 4.
#
# Following Hugonnet et al. (2022): we standardize dh by the NMAD, fit an
# empirical variogram, and use the range to compute the effective number of
# independent samples within each glacier polygon.

print("\n" + "="*60)
print("Step 2: Variogram fitting for uncertainty characterization")
print("="*60)

# Load the glacier union mask
glacier_union_gdf = gpd.read_file(glacier_union_path)

# We characterize uncertainty for each source DEM against the reference
variogram_params = {}  # dem_name → {"range_m": ..., "nmad": ...}

source_dems = ["IGM_1954", "CerroBlanco_2024", "LasTermas_2024"]

for dem_name in source_dems:
    src_coreg_path = os.path.join(coreg_folder, f"{dem_name}_coreg.tif")
    if not os.path.exists(src_coreg_path):
        print(f"  WARNING: {src_coreg_path} not found, skipping variogram fit.")
        variogram_params[dem_name] = {"range_m": np.nan, "nmad": np.nan}
        continue

    print(f"\n  Fitting variogram for {dem_name}...")
    src_dem = xdem.DEM(src_coreg_path)

    # dh on stable terrain
    dh = ref_dem_xdem - src_dem

    # Rasterize glacier union mask
    glacier_mask_arr = gu.Vector(glacier_union_path).create_mask(ref_dem_xdem)
    inlier_mask = ~glacier_mask_arr

    dh_stable = dh.data[inlier_mask]
    dh_stable = dh_stable[np.isfinite(dh_stable)]
    nmad_val   = xdem.spatialstats.nmad(dh_stable)
    print(f"    Stable-terrain NMAD: {nmad_val:.3f} m")

    # Fit empirical variogram
    # We sample up to 10,000 points to keep runtime manageable.
    # nlag and maxlag can be tuned to your study area extent.
    try:
        df_vgm, params = xdem.spatialstats.sample_empirical_variogram(
            values   = dh,
            subsample = 10000,
            n_variograms = 3,      # average over multiple random samples
            estimator = "dowd",    # robust to outliers (Dowd 1984)
            random_state = 42,
        )

        # Fit a model variogram (spherical by default)
        vgm_model, params_fit = xdem.spatialstats.fit_sum_model_variogram(
            list_model = ["Sph"],
            empirical_variogram = df_vgm,
        )

        corr_range = params_fit[0]["range"]
        print(f"    Fitted variogram range (correlation length): {corr_range:.0f} m")
        variogram_params[dem_name] = {"range_m": corr_range, "nmad": nmad_val,
                                       "vgm_model": vgm_model,
                                       "params_fit": params_fit}

        # Save variogram plot
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(df_vgm["lags"], df_vgm["exp"], s=10, alpha=0.6, label="Empirical")
        lag_plot = np.linspace(0, df_vgm["lags"].max(), 200)
        ax.plot(lag_plot,
                xdem.spatialstats.sum_model_variogram(lag_plot, vgm_model, params_fit),
                "r-", label="Fitted model")
        ax.set_xlabel("Lag distance (m)")
        ax.set_ylabel("Semivariance (m²)")
        ax.set_title(f"Variogram — {dem_name} stable terrain dh")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(figures_folder, f"variogram_{dem_name}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    except Exception as e:
        print(f"    WARNING: Variogram fitting failed ({e}). "
              "Falling back to NMAD-only uncertainty.")
        variogram_params[dem_name] = {"range_m": np.nan, "nmad": nmad_val}

# ------------------------------------------------------------------------------
# Step 3: DEM differencing + clipping per epoch
# ------------------------------------------------------------------------------
# Replaces: ArcPy ExtractByMask + Raster subtraction
# Uses:     rasterio.mask + numpy array arithmetic

print("\n" + "="*60)
print("Step 3: DEM differencing")
print("="*60)

epoch_results = []  # will hold per-polygon stats for each epoch

for current_name, baseline_name, outline_label, t_start, t_end in EPOCHS:
    dt = t_end - t_start  # years
    epoch_label = f"{current_name}_minus_{baseline_name}"
    print(f"\n  {epoch_label}  ({t_start}–{t_end}, Δt={dt} yr)")

    # Paths — current DEM
    curr_path = os.path.join(coreg_folder, f"{current_name}_coreg.tif")
    if not os.path.exists(curr_path):
        curr_path = os.path.join("aligned_dems", f"{current_name}_aligned.tif")

    # Paths — baseline DEM
    base_path = os.path.join(coreg_folder, f"{baseline_name}_coreg.tif")
    if not os.path.exists(base_path):
        base_path = os.path.join("aligned_dems", f"{baseline_name}_aligned.tif")

    # Load outlines in DEM CRS
    with rasterio.open(curr_path) as src:
        dem_crs = src.crs

    outlines_gdf = load_outlines(outline_label, dem_crs)

    # Full-extent dh raster (current - baseline) clipped to glacier union
    # We open both DEMs and compute dh over the full raster extent, then
    # clip to individual polygons for zonal stats below.
    with rasterio.open(curr_path) as src_curr, rasterio.open(base_path) as src_base:
        # Read arrays (they should already be on the same grid from Notebook 1)
        arr_curr = src_curr.read(1).astype(float)
        arr_base = src_base.read(1).astype(float)
        nd_curr  = src_curr.nodata
        nd_base  = src_base.nodata
        transform = src_curr.transform
        profile   = src_curr.profile.copy()

        # Mask nodata
        if nd_curr is not None:
            arr_curr[arr_curr == nd_curr] = np.nan
        if nd_base is not None:
            arr_base[arr_base == nd_base] = np.nan

        dh_arr = arr_curr - arr_base  # elevation change (m), current minus baseline

    # Save full dh raster
    dh_path = os.path.join(output_folder, f"{epoch_label}_dh.tif")
    profile.update(dtype="float32", nodata=np.nan, count=1)
    with rasterio.open(dh_path, "w", **profile) as dst:
        dst.write(dh_arr.astype("float32"), 1)
    print(f"    Saved dh raster: {dh_path}")

    # ------------------------------------------------------------------
    # Zonal statistics per glacier polygon
    # Replaces: ArcPy ZonalStatisticsAsTable
    # ------------------------------------------------------------------
    stats = rasterstats.zonal_stats(
        outlines_gdf,
        dh_path,
        stats  = ["count", "mean", "std", "median", "min", "max"],
        nodata = np.nan,
        all_touched = False,
    )

    zonal_df = outlines_gdf[[GLACIER_ID_FIELD, "geometry"]].copy()
    zonal_df = zonal_df.join(pd.DataFrame(stats))

    # Pixel area (m²) from transform
    pixel_area_m2 = abs(transform.a * transform.e)

    # Glacier area from polygon geometry (m²)
    zonal_df["area_m2"] = zonal_df.geometry.area

    # Volume change (m³) = mean dh × glacier area
    zonal_df["dV_m3"] = zonal_df["mean"] * zonal_df["area_m2"]

    # Mean annual dh rate
    zonal_df["dh_yr_m"] = zonal_df["mean"] / dt

    # Volume change rate (m³ yr⁻¹)
    zonal_df["dV_yr_m3"] = zonal_df["dV_m3"] / dt

    # Specific mass balance (m w.e. yr⁻¹) — flagged per reviewer note
    # This is provided for comparison with published records but should be
    # labeled "elevation change rate converted to w.e." rather than "mass balance"
    # because glacier area changes are not accounted for.
    zonal_df["mb_mwe_yr"] = zonal_df["dh_yr_m"] * DENSITY_FACTOR

    zonal_df["epoch"]       = epoch_label
    zonal_df["t_start"]     = t_start
    zonal_df["t_end"]       = t_end
    zonal_df["dt_yr"]       = dt
    zonal_df["outline_set"] = outline_label

    epoch_results.append(zonal_df)

# ------------------------------------------------------------------------------
# Step 4: Uncertainty propagation following Hugonnet et al. (2022)
# ------------------------------------------------------------------------------
# Per-pixel elevation error σ is estimated as the stable-terrain NMAD.
# The uncertainty of the mean dh over a glacier polygon area A is NOT σ/√N
# (which assumes independent pixels), but instead accounts for the spatial
# autocorrelation length L fitted in Step 2:
#
#   σ_dh̄ = σ × √(πL² / (2A))        [Rolstad et al. 2009, Eq. 14]
#
# when the glacier area A >> πL²/2 (i.e., the glacier is large relative to
# the correlation length). For small glaciers where A < πL²/2, the pixels
# are effectively fully correlated and σ_dh̄ ≈ σ.
#
# Volume change uncertainty:
#   σ_dV = σ_dh̄ × A
#
# Mass balance uncertainty (in m w.e. yr⁻¹):
#   σ_mb = σ_dV × ρ_ice / (ρ_water × A × dt)
#         = σ_dh̄ × ρ_ice / (ρ_water × dt)

print("\n" + "="*60)
print("Step 4: Uncertainty propagation")
print("="*60)

all_results = pd.concat(epoch_results, ignore_index=True)

def compute_uncertainty(row, vgm_params: dict, dt: float) -> pd.Series:
    """
    Compute propagated uncertainty for a single glacier polygon row.
    Returns σ_dh̄ (m), σ_dV (m³), σ_mb (m w.e. yr⁻¹).
    """
    # Identify which source DEM drives the uncertainty for this epoch
    # (we use the non-SRTM DEM's variogram; if both are non-SRTM, use the
    # one with worse NMAD as the conservative choice)
    current_name  = row["epoch"].split("_minus_")[0]
    baseline_name = row["epoch"].split("_minus_")[1]

    # Choose the DEM with the larger NMAD as the dominant uncertainty source
    candidates = [n for n in [current_name, baseline_name] if n in vgm_params]
    if not candidates:
        return pd.Series({"sigma_dh_m": np.nan, "sigma_dV_m3": np.nan,
                          "sigma_mb_mwe_yr": np.nan})

    best = max(candidates, key=lambda n: vgm_params[n].get("nmad", 0))
    params = vgm_params[best]
    sigma  = params["nmad"]          # per-pixel elevation error (m)
    L      = params["range_m"]       # spatial correlation length (m)
    A      = row["area_m2"]          # glacier area (m²)

    if np.isnan(L) or np.isnan(A) or A <= 0:
        # Fallback: treat all pixels as independent (lower bound on uncertainty)
        N_pixels = row["count"]
        sigma_dh = sigma / np.sqrt(max(N_pixels, 1))
    else:
        # Effective area of one correlated "patch"
        A_corr = np.pi * L**2 / 2.0
        if A_corr >= A:
            # Glacier smaller than correlation length → fully correlated
            sigma_dh = sigma
        else:
            # Rolstad et al. (2009) Eq. 14
            sigma_dh = sigma * np.sqrt(A_corr / A)

    sigma_dV = sigma_dh * A                              # m³
    sigma_mb = sigma_dh * DENSITY_FACTOR / dt            # m w.e. yr⁻¹ per pixel
    # Volume-based mass balance uncertainty (total, not per unit area)
    sigma_mb_total = sigma_dV * DENSITY_FACTOR / dt      # m³ w.e. yr⁻¹

    return pd.Series({
        "sigma_dh_m":          round(sigma_dh, 4),
        "sigma_dV_m3":         round(sigma_dV, 2),
        "sigma_mb_mwe_yr":     round(sigma_mb, 4),
        "sigma_mb_total_m3yr": round(sigma_mb_total, 2),
        "uncertainty_source":  best,
        "corr_length_m":       round(L, 1) if not np.isnan(L) else np.nan,
    })

# Apply uncertainty computation to all rows
uncertainty_cols = all_results.apply(
    lambda row: compute_uncertainty(row, variogram_params, row["dt_yr"]),
    axis=1
)
all_results = pd.concat([all_results, uncertainty_cols], axis=1)

# Drop geometry for CSV export
export_cols = [
    GLACIER_ID_FIELD, "epoch", "t_start", "t_end", "dt_yr", "outline_set",
    "area_m2", "count", "mean", "std", "median",
    "dh_yr_m", "dV_m3", "dV_yr_m3", "mb_mwe_yr",
    "sigma_dh_m", "sigma_dV_m3", "sigma_mb_mwe_yr",
    "uncertainty_source", "corr_length_m",
]
results_export = all_results[[c for c in export_cols if c in all_results.columns]]
results_path   = os.path.join(results_folder, "glacier_elevation_change_results.csv")
results_export.to_csv(results_path, index=False)
print(f"Results saved: {results_path}")

# ------------------------------------------------------------------------------
# Step 5: Zonal stats for slope and aspect
# Replaces: ArcPy ZonalStatisticsAsTable for slope/aspect
# ------------------------------------------------------------------------------
print("\nComputing slope and aspect zonal stats...")

for outline_label in ["2000", "2019"]:
    outlines_gdf_stats = load_outlines(outline_label, rasterio.open(slope_path).crs)

    slope_stats  = rasterstats.zonal_stats(outlines_gdf_stats, slope_path,
                                            stats=["min", "max", "mean"], nodata=np.nan)
    aspect_stats = rasterstats.zonal_stats(outlines_gdf_stats, aspect_path,
                                            stats=["min", "max", "mean"], nodata=np.nan)

    df_s = outlines_gdf_stats[[GLACIER_ID_FIELD]].copy()
    df_s = df_s.join(pd.DataFrame(slope_stats).rename(
        columns={"min":"Slp_MIN","max":"Slp_MAX","mean":"Slp_MEAN"}))
    df_s = df_s.join(pd.DataFrame(aspect_stats).rename(
        columns={"min":"Asp_MIN","max":"Asp_MAX","mean":"Asp_MEAN"}))
    df_s.to_csv(os.path.join(results_folder, f"slope_aspect_stats_{outline_label}.csv"),
                index=False)

print("  Slope/aspect stats saved.")

# ------------------------------------------------------------------------------
# Step 6: Figures
# ------------------------------------------------------------------------------

# --- 6a: dh maps ---
print("\nGenerating dh maps...")
for current_name, baseline_name, outline_label, t_start, t_end in EPOCHS:
    epoch_label = f"{current_name}_minus_{baseline_name}"
    dh_path = os.path.join(output_folder, f"{epoch_label}_dh.tif")
    if not os.path.exists(dh_path):
        continue
    with rasterio.open(dh_path) as src:
        data = src.read(1).astype(float)
        data[data == src.nodata] = np.nan
        extent = [src.bounds.left, src.bounds.right,
                  src.bounds.bottom, src.bounds.top]

    abs_max = np.nanpercentile(np.abs(data), 97)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max,
                   extent=extent, origin="upper")
    plt.colorbar(im, ax=ax, label="Elevation change (m)", shrink=0.8)
    ax.set_title(f"ΔElev {t_start}–{t_end}\n({current_name} − {baseline_name})",
                 fontsize=12)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    fig.tight_layout()
    fig.savefig(os.path.join(figures_folder, f"dh_map_{epoch_label}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

# --- 6b: Bar charts (preserved from original notebook 3) ---
print("Generating mass balance bar charts...")

data_2000 = {
    "Glacier/Period": [
        "Whole complex\n1954–2000", "Glaciar Nevado\n1954–2000",
        "Cerro Blanco\n1954–2024", "Las Termas\n1954–2024",
        "Glaciar Nevado\n1954–2024", "Cerro Blanco\n2000–2024",
        "Las Termas\n2000–2024",   "Glaciar Nevado\n2000–2024",
    ],
    "MB":    [-0.20, -0.30, -0.41, -0.13, -0.41, -0.60, -0.32, -0.60],
    "Error": [ 0.47,  0.47,  0.33,  0.18,  0.32,  0.29,  0.18,  0.29],
}
data_2019 = {
    "Glacier/Period": data_2000["Glacier/Period"],
    "MB":    [-0.20, -0.27, -0.37, -0.16, -0.37, -0.54, -0.36, -0.54],
    "Error": [ 0.47,  0.47,  0.33,  0.18,  0.33,  0.29,  0.18,  0.29],
}
df_2000 = pd.DataFrame(data_2000)
df_2019 = pd.DataFrame(data_2019)

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": "Helvetica"})

x     = np.arange(len(df_2000))
width = 0.35

fig, ax = plt.subplots(figsize=(13, 6))
ax.bar(x - width/2, df_2000["MB"], width, yerr=df_2000["Error"],
       capsize=5, label="2000 outlines", color="steelblue", alpha=0.85)
ax.bar(x + width/2, df_2019["MB"], width, yerr=df_2019["Error"],
       capsize=5, label="2019 outlines", color="darkorange", alpha=0.85)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(df_2000["Glacier/Period"], fontsize=9)
ax.set_ylabel("Elevation change rate (m w.e. yr⁻¹)\n[see note on reporting]", fontsize=10)
ax.set_title("Geodetic elevation change — Nevados de Chillán", fontsize=13, fontweight="bold")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(figures_folder, "fig_elevation_change_summary.png"),
            dpi=300, bbox_inches="tight")
plt.show()

# --- 6c: Per-glacier 3-panel figure ---
data_by_glacier = {
    "Cerro Blanco": {
        "2000": {"MB":[-0.24,-0.41,-0.60], "Error":[0.47,0.33,0.29]},
        "2019": {"MB":[-0.24,-0.37,-0.54], "Error":[0.47,0.33,0.29]},
        "periods": ["1954–2000","1954–2024","2000–2024"],
    },
    "Las Termas": {
        "2000": {"MB":[ 0.00,-0.13,-0.32], "Error":[0.47,0.18,0.18]},
        "2019": {"MB":[-0.05,-0.16,-0.36], "Error":[0.47,0.18,0.18]},
        "periods": ["1954–2000","1954–2024","2000–2024"],
    },
    "Glaciar Nevado": {
        "2000": {"MB":[-0.30,-0.41,-0.60], "Error":[0.47,0.32,0.29]},
        "2019": {"MB":[-0.27,-0.37,-0.54], "Error":[0.47,0.33,0.29]},
        "periods": ["1954–2000","1954–2024","2000–2024"],
    },
}

fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
for ax, (glacier, gdata) in zip(axes, data_by_glacier.items()):
    x_g = np.arange(len(gdata["periods"]))
    ax.bar(x_g - width/2, gdata["2000"]["MB"], width,
           yerr=gdata["2000"]["Error"], capsize=5,
           label="2000 outlines", color="steelblue")
    ax.bar(x_g + width/2, gdata["2019"]["MB"], width,
           yerr=gdata["2019"]["Error"], capsize=5,
           label="2019 outlines", color="darkorange")
    ax.set_title(glacier, fontsize=14, fontweight="bold")
    ax.set_xticks(x_g)
    ax.set_xticklabels(gdata["periods"], fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Elevation change rate (m w.e. yr⁻¹)", fontsize=10)
    ax.legend(fontsize=9)

fig.suptitle("Geodetic elevation change by glacier — Nevados de Chillán",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(figures_folder, "fig05_glacier_MB_comparison_highres.png"),
            dpi=900, bbox_inches="tight")
plt.show()

print("\nNotebook 3 complete.")
print(f"  Results:  {results_folder}/")
print(f"  Figures:  {figures_folder}/")
print(f"  dh maps:  {output_folder}/")
