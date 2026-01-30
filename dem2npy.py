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

    # ✅ 你 ERA5 是 latitude/longitude；运行时传 --lat_name latitude --lon_name longitude
    p.add_argument("--lat_name", type=str, default="lat")
    p.add_argument("--lon_name", type=str, default="lon")

    p.add_argument("--dtype", type=str, default="float32")
    p.add_argument("--fill_value", type=float, default=np.nan)
    p.add_argument("--resampling", type=str, default="bilinear",
                   choices=["nearest", "bilinear", "average"])

    # ✅ 新增：拼接策略
    # overwrite: 有效值直接覆盖（推荐）
    # first:     先到先得（你原来的逻辑）
    # mean:      重叠区在线均值
    # max:       重叠区取最大（有时可避免边缘小坑）
    p.add_argument("--merge_method", type=str, default="overwrite",
                   choices=["overwrite", "first", "mean", "max"])

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
    lat_use = lat if lat_desc else lat[::-1]

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

    # ✅ 建议：输出 fill_value 用 NaN（float32 下最稳）
    # 如果用户传了非 NaN 的 fill_value，也照样支持
    mmap = np.lib.format.open_memmap(out_npy, mode="w+", dtype=np.dtype(args.dtype), shape=(H, W))
    mmap[:] = args.fill_value

    # 对 mean/max/first 需要的辅助数组
    filled = np.zeros((H, W), dtype=bool)              # for "first"
    sum_arr = None
    cnt_arr = None
    if args.merge_method == "mean":
        sum_arr = np.zeros((H, W), dtype="float64")    # 用 float64 累加更稳
        cnt_arr = np.zeros((H, W), dtype="uint32")

    resampling = get_resampling(args.resampling)

    for fp in tqdm(dem_files, desc=f"Reprojecting tiles ({args.merge_method})"):
        with rasterio.open(fp) as src:
            if src.crs is None:
                raise ValueError(f"DEM tile has no CRS: {fp} (crs=None). Please set CRS before warping.")

            src_data = src.read(1).astype("float32", copy=False)

            # nodata -> NaN（即使 nodata=None 也没事）
            src_nodata = src.nodata
            if src_nodata is not None:
                src_data[src_data == src_nodata] = np.nan

            tmp = np.full((H, W), np.nan, dtype="float32")

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

            if args.merge_method == "overwrite":
                # ✅ 推荐：有效值直接覆盖
                mmap[valid] = tmp[valid].astype(mmap.dtype, copy=False)

            elif args.merge_method == "first":
                # 你原来的逻辑：哪里没填过才写
                write_mask = valid & (~filled)
                if np.any(write_mask):
                    mmap[write_mask] = tmp[write_mask].astype(mmap.dtype, copy=False)
                    filled[write_mask] = True

            elif args.merge_method == "max":
                # 重叠区取最大（需要当前 mmap 有可比较值；NaN 处理）
                cur = np.array(mmap, copy=False)
                cur_valid = ~np.isnan(cur)
                # 若 cur 是 NaN 且 tmp 有值 -> 用 tmp
                take_tmp = valid & (~cur_valid)
                if np.any(take_tmp):
                    mmap[take_tmp] = tmp[take_tmp].astype(mmap.dtype, copy=False)
                # 二者都有值 -> 取 max
                both = valid & cur_valid
                if np.any(both):
                    mmap[both] = np.maximum(cur[both], tmp[both]).astype(mmap.dtype, copy=False)

            elif args.merge_method == "mean":
                # 在线均值：sum/cnt
                sum_arr[valid] += tmp[valid].astype("float64", copy=False)
                cnt_arr[valid] += 1

    # mean 模式：写回 mmap
    if args.merge_method == "mean":
        mean_mask = cnt_arr > 0
        # 先填 fill_value，再填 mean
        mmap[:] = args.fill_value
        mmap[mean_mask] = (sum_arr[mean_mask] / cnt_arr[mean_mask]).astype(mmap.dtype)

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
        "merge_method": args.merge_method,
        "note": "DEM tiles reprojected/resampled onto ERA5 subdomain lat/lon grid. Output shape matches ref lat/lon."
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("✅ Saved:", out_npy)
    print("✅ Meta :", out_meta)


if __name__ == "__main__":
    main()
