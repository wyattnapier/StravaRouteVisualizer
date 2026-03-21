#!/usr/bin/env python3
"""
strava_to_3d.py
===============
Fetches GPX data from a Strava activity, downloads elevation data,
and generates multi-color 3D-printable STL files:
  - baseplate.stl       : flat base layer
  - terrain.stl         : surrounding terrain mesh
  - route.stl           : raised route path (highlight color)

Usage:
    python strava_to_3d.py [--activity-id <ID>] [options]
    (omit --activity-id to automatically use your most recent activity)

Dependencies:
    pip install requests stravalib gpxpy numpy scipy trimesh \
                elevation rasterio shapely matplotlib tqdm

You need a Strava API app — get one at:
    https://www.strava.com/settings/api
"""

import os
import sys
import math
import argparse
import json
import struct
import tempfile
from pathlib import Path

import numpy as np

# ─── Lazy imports with helpful error messages ──────────────────────────────────

def require(pkg, pip_name=None):
    import importlib
    try:
        return importlib.import_module(pkg)
    except ImportError:
        pip = pip_name or pkg
        print(f"  Missing package '{pip}'. Install with:  pip install {pip}")
        sys.exit(1)

# ─── Constants ─────────────────────────────────────────────────────────────────

STRAVA_AUTH_URL   = "https://www.strava.com/oauth/token"
STRAVA_API_BASE   = "https://www.strava.com/api/v3"
OPENTOPO_BASE     = "https://portal.opentopography.org/API/globaldem"

# Elevation dataset: SRTMGL1 (30m), SRTMGL3 (90m), AW3D30 (30m)
DEM_DATASET       = "SRTMGL1"

# 3D print dimensions (mm)
DEFAULT_WIDTH_MM  = 150.0   # x extent of model
DEFAULT_DEPTH_MM  = 150.0   # y extent
BASE_THICKNESS_MM = 3.0     # flat base layer height
TERRAIN_HEIGHT_MM = 20.0    # max terrain relief height above base
ROUTE_RAISE_MM    = 1.5     # how much route stands above terrain
ROUTE_WIDTH_MM    = 1.2     # route tube diameter

# Mesh resolution
TERRAIN_GRID_N    = 200     # NxN terrain grid points
ROUTE_TUBE_SIDES  = 8       # polygon sides for route tube


# ══════════════════════════════════════════════════════════════════════════════
#  1. STRAVA AUTHENTICATION
# ══════════════════════════════════════════════════════════════════════════════

def strava_get_token(client_id: str, client_secret: str, refresh_token: str) -> str:
    """Exchange refresh token for a fresh access token."""
    requests = require("requests")
    resp = requests.post(STRAVA_AUTH_URL, data={
        "client_id":     client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type":    "refresh_token",
    })
    resp.raise_for_status()
    token = resp.json()["access_token"]
    print(f"  ✓ Strava access token obtained")
    return token


def strava_get_activity(activity_id: int, token: str) -> dict:
    requests = require("requests")
    url = f"{STRAVA_API_BASE}/activities/{activity_id}"
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"})
    resp.raise_for_status()
    data = resp.json()
    print(f"  ✓ Activity: '{data.get('name', activity_id)}'  "
          f"({data.get('type','?')}, {data.get('distance',0)/1000:.1f} km)")
    return data


def strava_get_streams(activity_id: int, token: str) -> dict:
    """Fetch lat/lng/altitude streams from Strava."""
    requests = require("requests")
    url = f"{STRAVA_API_BASE}/activities/{activity_id}/streams"
    params = {"keys": "latlng,altitude,distance", "key_by_type": "true"}
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, params=params)
    resp.raise_for_status()
    return resp.json()


def strava_get_latest_activity_id(token: str) -> int:
    """Return the ID of the athlete's most recent activity that has GPS data."""
    requests = require("requests")
    url = f"{STRAVA_API_BASE}/athlete/activities"
    # Fetch up to 10 recent activities and pick the first with a map
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"},
                        params={"per_page": 10, "page": 1})
    resp.raise_for_status()
    activities = resp.json()
    if not activities:
        raise ValueError("No activities found on this Strava account.")
    for act in activities:
        if act.get("map", {}).get("summary_polyline"):
            print(f"  ✓ Most recent GPS activity: '{act['name']}' "
                  f"(ID {act['id']}, {act.get('type','?')}, "
                  f"{act.get('distance', 0)/1000:.1f} km)")
            return act["id"]
    raise ValueError(
        "None of your 10 most recent activities have GPS data. "
        "Specify --activity-id explicitly."
    )


def extract_route_coords(streams: dict):
    """
    Returns arrays: lats, lons, alts (metres), dists (metres).
    altitude may be empty — we'll fill from DEM later.
    """
    latlng = streams.get("latlng", {}).get("data", [])
    if not latlng:
        raise ValueError("No latlng stream found for this activity.")
    lats = np.array([p[0] for p in latlng])
    lons = np.array([p[1] for p in latlng])
    alts_raw = streams.get("altitude", {}).get("data", [])
    alts = np.array(alts_raw) if alts_raw else np.zeros(len(lats))
    dists_raw = streams.get("distance", {}).get("data", [])
    dists = np.array(dists_raw) if dists_raw else np.zeros(len(lats))
    print(f"  ✓ Route: {len(lats)} points, "
          f"lat [{lats.min():.4f}–{lats.max():.4f}], "
          f"lon [{lons.min():.4f}–{lons.max():.4f}]")
    return lats, lons, alts, dists


# ══════════════════════════════════════════════════════════════════════════════
#  2. GPX EXPORT (optional — also saves a .gpx for reference)
# ══════════════════════════════════════════════════════════════════════════════

def save_gpx(lats, lons, alts, output_path: Path):
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<gpx version="1.1" creator="strava_to_3d">',
             '  <trk><trkseg>']
    for lat, lon, alt in zip(lats, lons, alts):
        lines.append(f'    <trkpt lat="{lat:.7f}" lon="{lon:.7f}">'
                     f'<ele>{alt:.1f}</ele></trkpt>')
    lines += ['  </trkseg></trk>', '</gpx>']
    output_path.write_text("\n".join(lines))
    print(f"  ✓ GPX saved → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  3. ELEVATION DATA  (OpenTopography API — free, no key needed for SRTM)
# ══════════════════════════════════════════════════════════════════════════════

def download_dem(lat_min, lat_max, lon_min, lon_max,
                 margin_deg=0.02, output_path: Path = None,
                 api_key: str = None) -> Path:
    """
    Download a GeoTIFF DEM from OpenTopography.
    Returns path to the .tif file.
    Requires a free API key from opentopography.org.
    """
    requests = require("requests")
    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix=".tif"))

    if not api_key:
        print("\nERROR: OpenTopography now requires a free API key.")
        print("  1. Sign up at: https://portal.opentopography.org/requestService?service=api")
        print("  2. Then run with: --opentopo-key YOUR_KEY")
        print("     or set:        export OPENTOPO_API_KEY=YOUR_KEY")
        print("  (Or skip terrain with --no-dem to use Strava altitude only)")
        sys.exit(1)

    south = lat_min - margin_deg
    north = lat_max + margin_deg
    west  = lon_min - margin_deg
    east  = lon_max + margin_deg

    params = {
        "demtype":      DEM_DATASET,
        "south":        south,
        "north":        north,
        "west":         west,
        "east":         east,
        "outputFormat": "GTiff",
        "API_Key":      api_key,
    }
    print(f"  Downloading DEM ({DEM_DATASET}) "
          f"[{south:.4f},{west:.4f}] → [{north:.4f},{east:.4f}] …")
    resp = requests.get(OPENTOPO_BASE, params=params, stream=True, timeout=120)
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
    size_kb = output_path.stat().st_size // 1024
    print(f"  ✓ DEM downloaded ({size_kb} KB) → {output_path}")
    return output_path


def read_dem_grid(tif_path: Path, lat_min, lat_max, lon_min, lon_max,
                  n: int = TERRAIN_GRID_N):
    """
    Read DEM and resample to an n×n grid over [lon_min,lon_max] × [lat_min,lat_max].
    Returns (grid_lons, grid_lats, elev_grid) each shape (n,n).
    """
    rasterio = require("rasterio")
    from rasterio.transform import rowcol

    grid_lons = np.linspace(lon_min, lon_max, n)
    grid_lats = np.linspace(lat_max, lat_min, n)   # top-to-bottom for raster
    lon2d, lat2d = np.meshgrid(grid_lons, grid_lats)

    with rasterio.open(tif_path) as ds:
        transform = ds.transform
        nodata = ds.nodata
        band = ds.read(1).astype(float)
        if nodata is not None:
            band[band == nodata] = np.nan

        rows, cols = rasterio.transform.rowcol(
            transform,
            lon2d.ravel(), lat2d.ravel()
        )
        rows = np.clip(np.array(rows), 0, ds.height - 1)
        cols = np.clip(np.array(cols), 0, ds.width  - 1)
        elev = band[rows, cols].reshape(n, n)

    # Fill NaN with nearest neighbor
    mask = np.isnan(elev)
    if mask.any():
        from scipy.ndimage import distance_transform_edt
        idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
        elev[mask] = elev[idx[0][mask], idx[1][mask]]

    print(f"  ✓ Terrain grid {n}×{n}  "
          f"elev [{np.nanmin(elev):.0f}–{np.nanmax(elev):.0f}] m")
    return lon2d, lat2d, elev


def sample_elevation_at_points(tif_path: Path, lats, lons):
    """Bilinear-sample DEM at route points."""
    rasterio = require("rasterio")
    with rasterio.open(tif_path) as ds:
        nodata = ds.nodata
        band = ds.read(1).astype(float)
        if nodata is not None:
            band[band == nodata] = np.nan
        rows, cols = rasterio.transform.rowcol(ds.transform, lons, lats)
        rows = np.clip(np.array(rows), 0, ds.height - 1)
        cols = np.clip(np.array(cols), 0, ds.width  - 1)
        alts = band[rows, cols]
    return alts


# ══════════════════════════════════════════════════════════════════════════════
#  4. COORDINATE → MM CONVERSION
# ══════════════════════════════════════════════════════════════════════════════

def geo_to_mm(lats, lons, lat_min, lat_max, lon_min, lon_max,
              width_mm, depth_mm):
    """Map geographic coords to mm within [0, width_mm] × [0, depth_mm]."""
    lat_range = lat_max - lat_min or 1e-9
    lon_range = lon_max - lon_min or 1e-9
    x = (lons - lon_min) / lon_range * width_mm
    y = (lats - lat_min) / lat_range * depth_mm
    return x, y


def elev_to_mm(elev, elev_min, elev_max, terrain_height_mm, base_thickness_mm):
    """Scale elevation values to mm above base."""
    elev_range = elev_max - elev_min or 1.0
    return base_thickness_mm + (elev - elev_min) / elev_range * terrain_height_mm


# ══════════════════════════════════════════════════════════════════════════════
#  5. STL WRITING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def write_binary_stl(triangles: np.ndarray, path: Path):
    """
    triangles: (N,3,3) array of float32 — N triangles, 3 vertices, xyz.
    """
    n = len(triangles)
    header = b"strava_to_3d STL" + b"\x00" * (80 - 16)
    with open(path, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", n))
        v0 = triangles[:, 0, :]
        v1 = triangles[:, 1, :]
        v2 = triangles[:, 2, :]
        e1 = v1 - v0
        e2 = v2 - v0
        normals = np.cross(e1, e2)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normals /= norms
        for i in range(n):
            f.write(struct.pack("<fff", *normals[i]))
            f.write(struct.pack("<fff", *triangles[i, 0]))
            f.write(struct.pack("<fff", *triangles[i, 1]))
            f.write(struct.pack("<fff", *triangles[i, 2]))
            f.write(struct.pack("<H", 0))  # attribute byte count
    print(f"  ✓ STL saved: {path}  ({n:,} triangles, "
          f"{path.stat().st_size//1024} KB)")


# ══════════════════════════════════════════════════════════════════════════════
#  6. BUILD BASEPLATE STL
# ══════════════════════════════════════════════════════════════════════════════

def build_baseplate(width_mm, depth_mm, base_thickness_mm) -> np.ndarray:
    """Solid rectangular box: 2 faces × 2 tris + 4 sides × 2 tris = 12 tris."""
    W, D, H = width_mm, depth_mm, base_thickness_mm
    tris = []
    # Bottom face (z=0)
    tris += [[[0,0,0],[W,0,0],[W,D,0]],
             [[0,0,0],[W,D,0],[0,D,0]]]
    # Top face (z=H)
    tris += [[[0,0,H],[W,D,H],[W,0,H]],
             [[0,0,H],[0,D,H],[W,D,H]]]
    # Front (y=0)
    tris += [[[0,0,0],[W,0,H],[W,0,0]],
             [[0,0,0],[0,0,H],[W,0,H]]]
    # Back (y=D)
    tris += [[[0,D,0],[W,D,0],[W,D,H]],
             [[0,D,0],[W,D,H],[0,D,H]]]
    # Left (x=0)
    tris += [[[0,0,0],[0,D,0],[0,D,H]],
             [[0,0,0],[0,D,H],[0,0,H]]]
    # Right (x=W)
    tris += [[[W,0,0],[W,D,H],[W,D,0]],
             [[W,0,0],[W,0,H],[W,D,H]]]
    return np.array(tris, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  7. BUILD TERRAIN STL  (heightmap mesh + closed sides + bottom)
# ══════════════════════════════════════════════════════════════════════════════

def build_terrain(lon2d, lat2d, elev_grid,
                  lat_min, lat_max, lon_min, lon_max,
                  width_mm, depth_mm,
                  elev_min, elev_max,
                  base_thickness_mm, terrain_height_mm) -> np.ndarray:
    n = elev_grid.shape[0]

    # Convert grid to mm
    x2d, y2d = geo_to_mm(lat2d, lon2d, lat_min, lat_max, lon_min, lon_max,
                          width_mm, depth_mm)
    z2d = elev_to_mm(elev_grid, elev_min, elev_max,
                     terrain_height_mm, base_thickness_mm)

    tris = []

    # ── Top surface (grid quads → 2 triangles each) ──
    for i in range(n - 1):
        for j in range(n - 1):
            v00 = [x2d[i,   j  ], y2d[i,   j  ], z2d[i,   j  ]]
            v10 = [x2d[i+1, j  ], y2d[i+1, j  ], z2d[i+1, j  ]]
            v01 = [x2d[i,   j+1], y2d[i,   j+1], z2d[i,   j+1]]
            v11 = [x2d[i+1, j+1], y2d[i+1, j+1], z2d[i+1, j+1]]
            tris.append([v00, v10, v11])
            tris.append([v00, v11, v01])

    # ── Bottom face at z = base_thickness_mm ──
    z_bot = base_thickness_mm
    tris.append([[0,      0,      z_bot],
                 [width_mm, depth_mm, z_bot],
                 [width_mm, 0,      z_bot]])
    tris.append([[0,      0,      z_bot],
                 [0,      depth_mm, z_bot],
                 [width_mm, depth_mm, z_bot]])

    # ── Side walls ──
    # Front edge (i = n-1, j = 0..n-2)  latitude-min row
    i = n - 1
    for j in range(n - 1):
        xa, ya, za = x2d[i, j],   y2d[i, j],   z2d[i, j]
        xb, yb, zb = x2d[i, j+1], y2d[i, j+1], z2d[i, j+1]
        tris.append([[xa, ya, za], [xb, yb, z_bot], [xa, ya, z_bot]])
        tris.append([[xa, ya, za], [xb, yb, zb],    [xb, yb, z_bot]])

    # Back edge (i = 0)
    i = 0
    for j in range(n - 1):
        xa, ya, za = x2d[i, j],   y2d[i, j],   z2d[i, j]
        xb, yb, zb = x2d[i, j+1], y2d[i, j+1], z2d[i, j+1]
        tris.append([[xa, ya, z_bot], [xb, yb, z_bot], [xa, ya, za]])
        tris.append([[xb, yb, z_bot], [xb, yb, zb],    [xa, ya, za]])

    # Left edge (j = 0)
    j = 0
    for i in range(n - 1):
        xa, ya, za = x2d[i,   j], y2d[i,   j], z2d[i,   j]
        xb, yb, zb = x2d[i+1, j], y2d[i+1, j], z2d[i+1, j]
        tris.append([[xa, ya, z_bot], [xa, ya, za], [xb, yb, z_bot]])
        tris.append([[xb, yb, z_bot], [xa, ya, za], [xb, yb, zb]])

    # Right edge (j = n-1)
    j = n - 1
    for i in range(n - 1):
        xa, ya, za = x2d[i,   j], y2d[i,   j], z2d[i,   j]
        xb, yb, zb = x2d[i+1, j], y2d[i+1, j], z2d[i+1, j]
        tris.append([[xa, ya, za], [xa, ya, z_bot], [xb, yb, z_bot]])
        tris.append([[xa, ya, za], [xb, yb, z_bot], [xb, yb, zb]])

    return np.array(tris, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  8. BUILD ROUTE STL  (tube along polyline)
# ══════════════════════════════════════════════════════════════════════════════

def tube_around_segment(p0, p1, radius, sides):
    """Return triangles forming a cylinder segment between p0 and p1."""
    p0 = np.array(p0, dtype=float)
    p1 = np.array(p1, dtype=float)
    axis = p1 - p0
    length = np.linalg.norm(axis)
    if length < 1e-9:
        return []
    axis /= length

    # Build perpendicular frame
    ref = np.array([0, 0, 1]) if abs(axis[2]) < 0.9 else np.array([1, 0, 0])
    u = np.cross(axis, ref)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    angles = np.linspace(0, 2 * math.pi, sides, endpoint=False)
    ring0 = [p0 + radius * (math.cos(a) * u + math.sin(a) * v) for a in angles]
    ring1 = [p1 + radius * (math.cos(a) * u + math.sin(a) * v) for a in angles]

    tris = []
    for i in range(sides):
        j = (i + 1) % sides
        tris.append([ring0[i], ring0[j], ring1[j]])
        tris.append([ring0[i], ring1[j], ring1[i]])

    # End caps
    for i in range(1, sides - 1):
        tris.append([ring0[0], ring0[i+1], ring0[i]])
        tris.append([ring1[0], ring1[i],   ring1[i+1]])

    return tris


def build_route(route_x, route_y, route_z_terrain,
                route_raise_mm, route_width_mm,
                sides=ROUTE_TUBE_SIDES) -> np.ndarray:
    """
    route_z_terrain: Z values sampled from terrain surface.
    The tube centre is raised by route_raise_mm above terrain.
    """
    radius = route_width_mm / 2.0
    tris = []
    n = len(route_x)
    for i in range(n - 1):
        p0 = [route_x[i],   route_y[i],   route_z_terrain[i]   + route_raise_mm]
        p1 = [route_x[i+1], route_y[i+1], route_z_terrain[i+1] + route_raise_mm]
        tris.extend(tube_around_segment(p0, p1, radius, sides))
    return np.array(tris, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  9. SUBSAMPLE ROUTE  (avoid too many tube segments)
# ══════════════════════════════════════════════════════════════════════════════

def subsample_route(lats, lons, alts, max_points=500):
    n = len(lats)
    if n <= max_points:
        return lats, lons, alts
    idx = np.round(np.linspace(0, n - 1, max_points)).astype(int)
    return lats[idx], lons[idx], alts[idx]


# ══════════════════════════════════════════════════════════════════════════════
#  10. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate multi-color 3D STL from Strava activity")
    p.add_argument("--activity-id",   required=False, type=int, default=None,
                   help="Strava activity ID. Omit to use your most recent activity.")
    p.add_argument("--client-id",     default=os.environ.get("STRAVA_CLIENT_ID"))
    p.add_argument("--client-secret", default=os.environ.get("STRAVA_CLIENT_SECRET"))
    p.add_argument("--refresh-token", default=os.environ.get("STRAVA_REFRESH_TOKEN"))
    p.add_argument("--output-dir",    default="./stl_output")
    p.add_argument("--width-mm",      type=float, default=DEFAULT_WIDTH_MM)
    p.add_argument("--depth-mm",      type=float, default=DEFAULT_DEPTH_MM)
    p.add_argument("--base-mm",       type=float, default=BASE_THICKNESS_MM)
    p.add_argument("--terrain-mm",    type=float, default=TERRAIN_HEIGHT_MM)
    p.add_argument("--route-raise-mm",type=float, default=ROUTE_RAISE_MM)
    p.add_argument("--route-width-mm",type=float, default=ROUTE_WIDTH_MM)
    p.add_argument("--terrain-grid",  type=int,   default=TERRAIN_GRID_N)
    p.add_argument("--margin-deg",    type=float, default=0.02,
                   help="Extra lat/lon margin around route for terrain (degrees)")
    p.add_argument("--dem-tif",       default=None,
                   help="Path to an already-downloaded DEM GeoTIFF (skips download)")
    p.add_argument("--no-dem",        action="store_true",
                   help="Skip DEM download; use Strava altitude stream only")
    p.add_argument("--opentopo-key",  default=os.environ.get("OPENTOPO_API_KEY"),
                   help="OpenTopography API key (free at opentopography.org). "
                        "Can also be set via OPENTOPO_API_KEY env var.")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Validate credentials ──
    for name, val in [("--client-id",     args.client_id),
                      ("--client-secret", args.client_secret),
                      ("--refresh-token", args.refresh_token)]:
        if not val:
            print(f"ERROR: {name} is required (or set env var "
                  f"{name.lstrip('-').upper().replace('-','_')})")
            sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Strava auth & data ──
    print("\n[1/6] Authenticating with Strava …")
    token = strava_get_token(args.client_id, args.client_secret, args.refresh_token)

    # Resolve activity ID — fetch most recent if not specified
    if args.activity_id is None:
        print("  No --activity-id specified, fetching most recent activity …")
        args.activity_id = strava_get_latest_activity_id(token)

    print("\n[2/6] Fetching activity streams …")
    activity  = strava_get_activity(args.activity_id, token)
    streams   = strava_get_streams(args.activity_id, token)
    lats, lons, alts_strava, dists = extract_route_coords(streams)

    # Save GPX
    save_gpx(lats, lons, alts_strava, out_dir / f"activity_{args.activity_id}.gpx")

    # Bounding box
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()

    # ── 2. Elevation data ──
    if args.no_dem:
        print("\n[3/6] Using Strava altitude stream (--no-dem specified) …")
        alts = alts_strava
        # Build a fake flat terrain grid at mean altitude for visualization
        n = args.terrain_grid
        grid_lons = np.linspace(lon_min - args.margin_deg,
                                lon_max + args.margin_deg, n)
        grid_lats = np.linspace(lat_max + args.margin_deg,
                                lat_min - args.margin_deg, n)
        lon2d, lat2d = np.meshgrid(grid_lons, grid_lats)
        elev_grid = np.full((n, n), np.mean(alts) if len(alts) else 0.0)
        tif_path = None
    else:
        print("\n[3/6] Downloading/reading DEM elevation data …")
        if args.dem_tif:
            tif_path = Path(args.dem_tif)
            print(f"  Using provided DEM: {tif_path}")
        else:
            tif_path = out_dir / f"dem_{args.activity_id}.tif"
            download_dem(lat_min, lat_max, lon_min, lon_max,
                         margin_deg=args.margin_deg, output_path=tif_path,
                         api_key=args.opentopo_key)

        lon2d, lat2d, elev_grid = read_dem_grid(
            tif_path,
            lat_min - args.margin_deg, lat_max + args.margin_deg,
            lon_min - args.margin_deg, lon_max + args.margin_deg,
            n=args.terrain_grid
        )
        alts = sample_elevation_at_points(tif_path, lats, lons)
        print(f"  Route elevation from DEM: "
              f"[{alts.min():.0f}–{alts.max():.0f}] m")

    # Global elevation range (terrain + route combined)
    elev_min = min(elev_grid.min(), alts.min())
    elev_max = max(elev_grid.max(), alts.max())

    # Terrain bounding box (with margin)
    lat_min_t = lat2d.min()
    lat_max_t = lat2d.max()
    lon_min_t = lon2d.min()
    lon_max_t = lon2d.max()

    # ── 3. Convert route coords to mm ──
    print("\n[4/6] Converting coordinates to mm …")
    lats_sub, lons_sub, alts_sub = subsample_route(lats, lons, alts)

    route_x, route_y = geo_to_mm(
        lats_sub, lons_sub,
        lat_min_t, lat_max_t, lon_min_t, lon_max_t,
        args.width_mm, args.depth_mm
    )
    route_z_terrain = elev_to_mm(
        alts_sub, elev_min, elev_max,
        args.terrain_mm, args.base_mm
    )
    print(f"  Route mm extent: "
          f"x[{route_x.min():.1f}–{route_x.max():.1f}] "
          f"y[{route_y.min():.1f}–{route_y.max():.1f}] "
          f"z[{route_z_terrain.min():.1f}–{route_z_terrain.max():.1f}]")

    # ── 4. Build STL meshes ──
    print("\n[5/6] Building STL meshes …")

    print("  → Baseplate …")
    base_tris = build_baseplate(args.width_mm, args.depth_mm, args.base_mm)

    print("  → Terrain …")
    terrain_tris = build_terrain(
        lon2d, lat2d, elev_grid,
        lat_min_t, lat_max_t, lon_min_t, lon_max_t,
        args.width_mm, args.depth_mm,
        elev_min, elev_max,
        args.base_mm, args.terrain_mm
    )

    print("  → Route tube …")
    route_tris = build_route(
        route_x, route_y, route_z_terrain,
        args.route_raise_mm, args.route_width_mm
    )

    # ── 5. Write STL files ──
    print("\n[6/6] Writing STL files …")
    write_binary_stl(base_tris,    out_dir / "baseplate.stl")
    write_binary_stl(terrain_tris, out_dir / "terrain.stl")
    write_binary_stl(route_tris,   out_dir / "route.stl")

    # ── Summary ──
    print(f"""
╔══════════════════════════════════════════════════════════╗
║  Done!  Output in: {str(out_dir):<38}║
╠══════════════════════════════════════════════════════════╣
║  baseplate.stl  →  Color 1 (e.g. white/grey base)       ║
║  terrain.stl    →  Color 2 (e.g. earthy brown/green)    ║
║  route.stl      →  Color 3 (e.g. bright accent color)   ║
╠══════════════════════════════════════════════════════════╣
║  Model size: {args.width_mm:.0f} × {args.depth_mm:.0f} × {args.base_mm+args.terrain_mm+args.route_raise_mm:.0f} mm (W × D × H)            ║
╠══════════════════════════════════════════════════════════╣
║  Multi-material printing tips:                           ║
║  • Import all 3 STLs into your slicer as a single body  ║
║  • Align them — they share the same coordinate space    ║
║  • Assign a filament/color to each STL part             ║
║  • PrusaSlicer / BambuStudio / OrcaSlicer all support   ║
║    multi-color with MMU/AMS or filament changes          ║
╚══════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
