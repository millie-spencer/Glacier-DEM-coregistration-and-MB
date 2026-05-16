# =============================================================================
# Notebook 2: DEM Co-registration
# =============================================================================
# This notebook co-registers DEMs to a common reference (SRTM 2000) using the
# xDEM package (xDEM Contributors, 2024; https://xdem.readthedocs.io).
#
# xDEM implements the Nuth & Kääb (2011) co-registration method used in the
# original workflow (Shean et al. demcoreg), plus additional methods. Using
# xDEM here also sets up the uncertainty propagation tools used in Notebook 3.
#
# INPUTS  (from Notebook 1, or your own aligned DEMs):
#   - aligned_dems/SRTM_2000_aligned.tif         ← reference DEM
#   - aligned_dems/IGM_1954_aligned.tif
#   - aligned_dems/CerroBlanco_2024_aligned.tif
#   - aligned_dems/LasTermas_2024_aligned.tif
#   - Glacier shapefiles (DGA 2000 and 2019)
#   - Stable-ground mask shapefile (manually digitized off-glacier areas)
#
# OUTPUTS:
#   - coreg_dems/<name>_coreg.tif                ← co-registered DEMs
#   - coreg_dems/coregistration_summary.csv      ← shift offsets and NMAD stats
#
# IMPORTANT NOTE ON STABLE TERRAIN MASKING (Reviewer comment):
#   The stable-ground mask used for co-registration should NOT include areas
#   that were glacierized in the older DEM epoch. Using 2000 glacier outlines
#   as the mask when co-registering a 1954 DEM means terrain that was ice
#   in 1954 (but bare ground by 2000) is incorrectly treated as stable.
#   We therefore use the UNION of all available glacier outlines as the
#   glacier mask, so that any area ever glacierized is excluded from
#   stable-terrain co-registration.
# =============================================================================

# ------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.merge import merge as rio_merge
import matplotlib.pyplot as plt
import xdem
import geoutils as gu

print(f"xDEM version: {xdem.__version__}")

# ------------------------------------------------------------------------------
# USER VARIABLES — update paths as needed
# ------------------------------------------------------------------------------
aligned_dem_folder = "aligned_dems"        # output folder from Notebook 1
output_folder      = "coreg_dems"
os.makedirs(output_folder, exist_ok=True)

# Glacier outlines — we use the UNION of 2000 and 2019 polygons as the mask.
# This ensures that any terrain glacierized in either epoch is excluded from
# the stable-terrain used for co-registration (see note above).
glacier_2000_path = "example_data_Nevados/Nevados_glacier_shapefiles/Nevados_shapefile_DGA2000/Nevados_polygons_DGA2000.shp"
glacier_2019_path = "example_data_Nevados/Nevados_glacier_shapefiles/Nevados_shapefile_DGA2019/Nevados_polygons_DGA2019.shp"

# Stable-ground mask (manually digitized off-glacier stable areas).
# This provides additional spatial filtering beyond the glacier polygon union.
stable_ground_path = "combined_mask_output.shp"   # from Notebook 2 KMZ merge

# DEMs to co-register (source DEMs), in order from oldest to newest.
# SRTM_2000 is the reference — it is NOT co-registered.
reference_name = "SRTM_2000"
source_dems = [
    "IGM_1954",
    "CerroBlanco_2024",
    "LasTermas_2024",
]

# Co-registration method.
# Options: "nuthkaab" | "icp" | "dh_slope" (vertical shift + slope/aspect)
# "nuthkaab" matches the original demcoreg workflow (Nuth & Kääb 2011).
# Reviewers may ask you to compare methods — see the comparison section below.
COREG_METHOD = "nuthkaab"

# ------------------------------------------------------------------------------
# Step 1: Build the glacier union mask
# ------------------------------------------------------------------------------
# We dissolve both polygon sets into a single geometry and take the union.
# This is the conservative masking approach recommended by the reviewer.

print("\nBuilding glacier union mask...")
gdf_2000 = gpd.read_file(glacier_2000_path)
gdf_2019 = gpd.read_file(glacier_2019_path)

# Ensure both are in the same CRS before unioning
if gdf_2000.crs != gdf_2019.crs:
    gdf_2019 = gdf_2019.to_crs(gdf_2000.crs)

glacier_union = gpd.GeoDataFrame(
    geometry=[gdf_2000.union_all().union(gdf_2019.union_all())],
    crs=gdf_2000.crs
)
glacier_union_path = os.path.join(output_folder, "glacier_union_mask.shp")
glacier_union.to_file(glacier_union_path)
print(f"  Glacier union mask saved: {glacier_union_path}")

# ------------------------------------------------------------------------------
# Step 2: Load reference DEM
# ------------------------------------------------------------------------------
ref_path = os.path.join(aligned_dem_folder, f"{reference_name}_aligned.tif")
print(f"\nLoading reference DEM: {ref_path}")
reference_dem = xdem.DEM(ref_path)
print(f"  CRS: {reference_dem.crs}  |  Resolution: {reference_dem.res} m")

# ------------------------------------------------------------------------------
# Step 3: Create the stable-terrain mask
# ------------------------------------------------------------------------------
# The inlier mask is True where terrain IS stable (i.e., NOT glacier).
# xDEM co-registration uses this to fit offsets only over stable ground.
#
# Strategy: combine the glacier union polygon mask with the manually digitized
# stable-ground areas. Pixels must be:
#   (a) outside the glacier union, AND
#   (b) inside the manually digitized stable areas (if provided)

print("\nCreating stable-terrain inlier mask...")

# Rasterize glacier union onto reference DEM grid (True = glacier pixel)
glacier_mask = gu.Vector(glacier_union_path).create_mask(reference_dem)

# If stable-ground shapefile exists, also restrict to those areas
if os.path.exists(stable_ground_path):
    stable_vec   = gu.Vector(stable_ground_path)
    stable_mask  = stable_vec.create_mask(reference_dem)  # True = stable area
    # Combine: stable AND not-glacier
    inlier_mask  = stable_mask & ~glacier_mask
    print("  Using digitized stable-ground areas intersected with glacier union mask.")
else:
    inlier_mask  = ~glacier_mask
    print("  No stable-ground shapefile found — using glacier union mask only.")

n_stable = int(np.sum(inlier_mask))
print(f"  Stable-terrain pixels available for co-registration: {n_stable:,}")

# ------------------------------------------------------------------------------
# Step 4: Co-register each source DEM to the reference
# ------------------------------------------------------------------------------

def get_coreg_pipeline(method: str):
    """
    Return an xDEM co-registration pipeline for the requested method.

    Nuth & Kääb (2011) is the default — it matches the original demcoreg
    workflow and is the standard for glacier geodesy.

    ICP (Besl & McKay 1992) is an alternative that works purely on 3D point
    clouds and can handle larger initial offsets.

    A combined pipeline (bias correction → Nuth & Kääb) is also available
    for DEMs with a known vertical offset (e.g. the 1954 IGM DEM).
    """
    if method == "nuthkaab":
        return xdem.coreg.NuthKaab()
    elif method == "icp":
        return xdem.coreg.ICP()
    elif method == "dh_slope":
        # Vertical shift correction followed by Nuth & Kääb
        return xdem.coreg.BiasCorr() + xdem.coreg.NuthKaab()
    else:
        raise ValueError(f"Unknown co-registration method: {method}")


summary_rows = []

for dem_name in source_dems:
    src_path = os.path.join(aligned_dem_folder, f"{dem_name}_aligned.tif")
    out_path = os.path.join(output_folder, f"{dem_name}_coreg.tif")

    print(f"\n{'='*60}")
    print(f"Co-registering: {dem_name}  →  {reference_name}")
    print(f"{'='*60}")

    # Load source DEM
    source_dem = xdem.DEM(src_path)

    # --- Pre-registration NMAD (off-glacier stable terrain) ---
    dh_before = (reference_dem - source_dem).data
    stable_vals_before = dh_before[inlier_mask]
    nmad_before = xdem.spatialstats.nmad(stable_vals_before[np.isfinite(stable_vals_before)])
    print(f"  NMAD before co-registration: {nmad_before:.3f} m")

    # --- Run co-registration ---
    pipeline = get_coreg_pipeline(COREG_METHOD)
    pipeline.fit(
        reference_elev = reference_dem,
        to_be_aligned_elev = source_dem,
        inlier_mask = inlier_mask,
    )

    # Apply the fitted transformation to the source DEM
    dem_coreg = pipeline.apply(source_dem)

    # --- Post-registration NMAD ---
    dh_after = (reference_dem - dem_coreg).data
    stable_vals_after = dh_after[inlier_mask]
    nmad_after = xdem.spatialstats.nmad(stable_vals_after[np.isfinite(stable_vals_after)])
    print(f"  NMAD after  co-registration: {nmad_after:.3f} m")
    print(f"  NMAD improvement: {(nmad_before - nmad_after):.3f} m  "
          f"({(nmad_before - nmad_after) / nmad_before * 100:.1f}%)")

    # --- Extract shift offsets from NuthKaab (if applicable) ---
    if COREG_METHOD in ("nuthkaab", "dh_slope"):
        try:
            # For a plain NuthKaab pipeline
            if COREG_METHOD == "nuthkaab":
                nk = pipeline
            else:
                # Second step in the pipeline is NuthKaab
                nk = pipeline.pipeline[1]
            dx = nk.meta.get("shift_x", np.nan)
            dy = nk.meta.get("shift_y", np.nan)
            dz = nk.meta.get("shift_z", np.nan)
        except Exception:
            dx = dy = dz = np.nan
    else:
        dx = dy = dz = np.nan

    print(f"  Estimated shifts — x: {dx:.2f} m, y: {dy:.2f} m, z: {dz:.2f} m")

    # --- Save co-registered DEM ---
    dem_coreg.save(out_path)
    print(f"  Saved: {out_path}")

    summary_rows.append({
        "DEM":          dem_name,
        "reference":    reference_name,
        "method":       COREG_METHOD,
        "shift_x_m":   round(dx, 3),
        "shift_y_m":   round(dy, 3),
        "shift_z_m":   round(dz, 3),
        "NMAD_before_m": round(nmad_before, 3),
        "NMAD_after_m":  round(nmad_after, 3),
        "NMAD_improvement_m": round(nmad_before - nmad_after, 3),
    })

# Save summary CSV
summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(output_folder, "coregistration_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"\nCo-registration summary saved: {summary_path}")
print(summary_df.to_string(index=False))

# ------------------------------------------------------------------------------
# Step 5: Visual QC — dh maps before and after co-registration
# ------------------------------------------------------------------------------
# Plot elevation difference (dh = reference - source) over stable terrain
# before and after co-registration to visually confirm improvement.

print("\nGenerating QC plots...")

fig, axes = plt.subplots(
    len(source_dems), 2,
    figsize=(14, 5 * len(source_dems)),
    constrained_layout=True
)
if len(source_dems) == 1:
    axes = [axes]

for i, dem_name in enumerate(source_dems):
    src_path    = os.path.join(aligned_dem_folder, f"{dem_name}_aligned.tif")
    coreg_path  = os.path.join(output_folder, f"{dem_name}_coreg.tif")

    src_dem    = xdem.DEM(src_path)
    coreg_dem  = xdem.DEM(coreg_path)

    dh_before = (reference_dem - src_dem).data.squeeze()
    dh_after  = (reference_dem - coreg_dem).data.squeeze()

    # Mask to stable terrain only for display
    dh_before_display = np.where(inlier_mask, dh_before, np.nan)
    dh_after_display  = np.where(inlier_mask, dh_after,  np.nan)

    clim = np.nanpercentile(np.abs(dh_before_display[np.isfinite(dh_before_display)]), 95)

    for ax, data, title in zip(
        axes[i],
        [dh_before_display, dh_after_display],
        [f"{dem_name}\nBefore co-reg (stable terrain dh)",
         f"{dem_name}\nAfter co-reg (stable terrain dh)"]
    ):
        im = ax.imshow(data, cmap="RdBu_r", vmin=-clim, vmax=clim)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        plt.colorbar(im, ax=ax, shrink=0.7, label="Elevation difference (m)")

qc_path = os.path.join(output_folder, "coregistration_qc.png")
fig.savefig(qc_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"QC plot saved: {qc_path}")

# ------------------------------------------------------------------------------
# Step 6 (Optional): Method comparison
# ------------------------------------------------------------------------------
# Uncomment this block to compare Nuth & Kääb vs ICP on one DEM.
# Useful for responding to the reviewer's suggestion to explore alternatives.

# COMPARE_DEM = "IGM_1954"
# src_path = os.path.join(aligned_dem_folder, f"{COMPARE_DEM}_aligned.tif")
# src_dem  = xdem.DEM(src_path)
#
# results = {}
# for method in ["nuthkaab", "icp", "dh_slope"]:
#     pipeline = get_coreg_pipeline(method)
#     pipeline.fit(reference_dem, src_dem, inlier_mask=inlier_mask)
#     dem_out = pipeline.apply(src_dem)
#     dh = (reference_dem - dem_out).data
#     stable = dh[inlier_mask]
#     results[method] = xdem.spatialstats.nmad(stable[np.isfinite(stable)])
#     print(f"  {method:12s}  NMAD: {results[method]:.3f} m")
#
# best_method = min(results, key=results.get)
# print(f"\nLowest post-coreg NMAD: {best_method} ({results[best_method]:.3f} m)")

print("\nNotebook 2 complete. Co-registered DEMs are in:", output_folder)
print("Pass the coreg_dems/ folder to Notebook 3 for differencing and uncertainty analysis.")
