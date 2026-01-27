import os
import json
import argparse
import numpy as np
import xarray as xr
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from rasterio.crs import CRS
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser("Warp DEM tiles to ERA5 subdomain lat/lon grid and save .npy")
    p.add_argument("--dem_dir", type=str, required=True, help="folder of DEM tiles (.tif/.tiff/.dem)")
    p.add_argument("--ref_nc", type=str, required=True, help="ERA5 subdomain netcdf containing lat/lon coords")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--out_name", type=str, default="dem_on_era5_grid")
    p.add_argument("--lat_name", type=str, default="lat")
    p.add_argument("--lon_name", type=str, default="lon")
    p.add_argument("--dtype", type=str, default="float32")
    p.add_argument("--fill_value", type=float, default=np.nan)
    p.add_argument("--resampling", type=str, default="bilinear",
                   choices=["nearest", "bilinear", "average"])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def list_dem_files(dem_dir):
    exts = (".tif", ".tiff", ".dem")
    files = [os.path.join(dem_dir, f) for f in os.listdir(dem_dir) if f.lower().endswith(exts)]
    files.sort()
    return files


def load_latlon(ref_nc, lat_name, lon_name):
    ds = xr.open_dataset(ref_nc)
    if lat_name not in ds.coords:
        raise KeyError(f"'{lat_name}' not in coords: {list(ds.coords)}")
    if lon_name not in ds.coords:
        raise KeyError(f"'{lon_name}' not in coords: {list(ds.coords)}")
    lat = ds[lat_name].values
    lon = ds[lon_name].values
    ds.close()
    return np.asarray(lat), np.asarray(lon)


def make_dst_transform(lat, lon):
    """
    Build raster transform for regular lat/lon grid (cell-centered coords).
    Handles lat ascending/descending, then later we flip back to match ref order.
    """
    lat = np.asarray(lat)
    lon = np.asarray(lon)

    lat_desc = lat[0] > lat[-1]
    if not lat_desc:
        lat_use = lat[::-1]
    else:
        lat_use = lat

    dlat = float(abs(lat_use[1] - lat_use[0])) if lat_use.size > 1 else 0.25
    dlon = float(abs(lon[1] - lon[0])) if lon.size > 1 else 0.25

    west = float(lon.min() - dlon / 2)
    north = float(lat_use.max() + dlat / 2)
    transform = from_origin(west, north, dlon, dlat)

    H = int(lat_use.size)
    W = int(lon.size)
    return transform, H, W, lat_desc


def get_resampling(name):
    return {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "average": Resampling.average,
    }[name]


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    dem_files = list_dem_files(args.dem_dir)
    if not dem_files:
        raise SystemExit("No DEM files found.")

    lat, lon = load_latlon(args.ref_nc, args.lat_name, args.lon_name)
    dst_transform, H, W, lat_is_desc = make_dst_transform(lat, lon)

    out_npy = os.path.join(args.out_dir, f"{args.out_name}.npy")
    out_meta = os.path.join(args.out_dir, f"{args.out_name}.meta.json")
    if (os.path.exists(out_npy) or os.path.exists(out_meta)) and not args.overwrite:
        raise FileExistsError("Output exists. Use --overwrite.")

    # memmap: write big array on disk with tiny RAM usage
    mmap = np.lib.format.open_memmap(out_npy, mode="w+", dtype=np.dtype(args.dtype), shape=(H, W))
    mmap[:] = args.fill_value

    # a boolean mask to track where we already have valid values
    filled = np.zeros((H, W), dtype=bool)

    resampling = get_resampling(args.resampling)

    for fp in tqdm(dem_files, desc="Reprojecting tiles"):
        with rasterio.open(fp) as src:
            src_data = src.read(1)

            # mark nodata as NaN for safer reproject
            src_nodata = src.nodata
            if src_nodata is not None:
                src_data = src_data.astype("float32", copy=False)
                src_data[src_data == src_nodata] = np.nan

            tmp = np.full((H, W), np.nan, dtype="float32")  # per-tile buffer

            reproject(
                source=src_data,
                destination=tmp,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=CRS.from_epsg(4326),
                resampling=resampling,
            )

            valid = ~np.isnan(tmp)
            if not np.any(valid):
                continue

            # 写入策略：哪里还没填过，就写入；你也可以改成“覆盖写入”或“取平均”
            write_mask = valid & (~filled)
            if np.any(write_mask):
                mmap[write_mask] = tmp[write_mask].astype(mmap.dtype, copy=False)
                filled[write_mask] = True

    # 如果参考 lat 原本是升序，我们前面为了 transform 翻成降序了，这里翻回去
    if not lat_is_desc:
        mmap[:] = mmap[::-1, :]
        filled[:] = filled[::-1, :]

    del mmap  # flush

    meta = {
        "shape": [int(len(lat)), int(len(lon))],
        "dtype": str(args.dtype),
        "fill_value": args.fill_value if not (isinstance(args.fill_value, float) and np.isnan(args.fill_value)) else "NaN",
        "ref_nc": args.ref_nc,
        "lat_name": args.lat_name,
        "lon_name": args.lon_name,
        "crs": "EPSG:4326",
        "resampling": args.resampling,
        "note": "DEM tiles reprojected/resampled onto ERA5 subdomain lat/lon grid. Output shape matches ref lat/lon."
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("✅ Saved:", out_npy)
    print("✅ Meta :", out_meta)


if __name__ == "__main__":
    main()
