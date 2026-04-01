#!/usr/bin/env python3
"""
strava_to_3d.py - fetch a Strava activity and generate multi-color STL files.
Usage: python strava_to_3d.py [--activity-id ID]
"""

import os
import sys
import math
import argparse
import struct
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import requests
import rasterio
import rasterio.transform
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import RegularGridInterpolator

try:
    from dotenv import load_dotenv
    _env = Path(__file__).parent / ".env"
    if _env.exists():
        load_dotenv(_env)
        print(f"  Loaded environment from {_env}")
except ImportError:
    pass

STRAVA_AUTH_URL = "https://www.strava.com/oauth/token"
STRAVA_API_BASE = "https://www.strava.com/api/v3"
OPENTOPO_BASE = "https://portal.opentopography.org/API/globaldem"
DEM_DATASET = "SRTMGL1"  # 30m resolution

DEFAULT_WIDTH_MM = 150.0
DEFAULT_DEPTH_MM = 150.0
BASE_THICKNESS_MM = 3.0
TERRAIN_HEIGHT_MM = 20.0
ROUTE_RAISE_MM = 1.5
ROUTE_WIDTH_MM = 1.2
ROUTE_EMBED_MM = 1.0  # how far below terrain the ribbon goes

TERRAIN_GRID_N = 200
ROUTE_TUBE_SIDES = 8


def strava_get_token(client_id, client_secret, refresh_token):
    resp = requests.post(STRAVA_AUTH_URL, data={
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    })
    resp.raise_for_status()
    token = resp.json()["access_token"]
    print(f"Strava access token obtained")
    return token


def strava_get_activity(activity_id, token):
    url = f"{STRAVA_API_BASE}/activities/{activity_id}"
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"})
    resp.raise_for_status()
    data = resp.json()
    print(f"Activity: '{data.get('name', activity_id)}'  "
          f"({data.get('type','?')}, {data.get('distance',0)/1000:.1f} km)")
    return data


def strava_get_streams(activity_id, token):
    url = f"{STRAVA_API_BASE}/activities/{activity_id}/streams"
    params = {"keys": "latlng,altitude,distance", "key_by_type": "true"}
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, params=params)
    resp.raise_for_status()
    return resp.json()


def strava_get_latest_activity_id(token):
    # grab the most recent activities and let the user choose one with GPS data
    url = f"{STRAVA_API_BASE}/athlete/activities"
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"},
                        params={"per_page": 25, "page": 1})
    resp.raise_for_status()
    activities = resp.json()

    if not activities:
        raise ValueError("No activities found on this Strava account.")

    gps_activities = [
        act for act in activities
        if act.get("map", {}).get("summary_polyline")
    ]

    if not gps_activities:
        raise ValueError(
            "None of your 10 most recent activities have GPS data. "
            "Specify --activity-id explicitly."
        )

    print("\nMost recent activities with GPS data:\n")

    for i, act in enumerate(gps_activities, 1):
        name = act.get("name", "Unnamed")
        typ = act.get("type", "?")
        dist_km = act.get("distance", 0) / 1000
        print(f"{i}. {name} ({typ}, {dist_km:.1f} km) [ID {act['id']}]")

    while True:
        try:
            choice = int(input("\nSelect activity number: "))
            if 1 <= choice <= len(gps_activities):
                selected = gps_activities[choice - 1]
                print(f"\nSelected: '{selected['name']}' (ID {selected['id']})")
                return selected["id"]
        except ValueError:
            pass

        print("Invalid selection. Try again.")


def extract_route_coords(streams):
    # pull out lats, lons, alts, dists from the stream data
    latlng = streams.get("latlng", {}).get("data", [])
    if not latlng:
        raise ValueError("No latlng stream found for this activity.")
    lats = np.array([p[0] for p in latlng])
    lons = np.array([p[1] for p in latlng])
    alts_raw = streams.get("altitude", {}).get("data", [])
    alts = np.array(alts_raw) if alts_raw else np.zeros(len(lats))
    dists_raw = streams.get("distance", {}).get("data", [])
    dists = np.array(dists_raw) if dists_raw else np.zeros(len(lats))
    print(f"Route: {len(lats)} points, "
          f"lat [{lats.min():.4f}-{lats.max():.4f}], "
          f"lon [{lons.min():.4f}-{lons.max():.4f}]")
    return lats, lons, alts, dists


def save_gpx(lats, lons, alts, output_path):
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<gpx version="1.1" creator="strava_to_3d">',
             '  <trk><trkseg>']
    for lat, lon, alt in zip(lats, lons, alts):
        lines.append(f'    <trkpt lat="{lat:.7f}" lon="{lon:.7f}">'
                     f'<ele>{alt:.1f}</ele></trkpt>')
    lines += ['  </trkseg></trk>', '</gpx>']
    output_path.write_text("\n".join(lines))
    print(f"  saved gpx to {output_path}")


def download_dem(lat_min, lat_max, lon_min, lon_max,
                 margin_deg=0.02, output_path=None, api_key=None):
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
        "demtype": DEM_DATASET,
        "south": south, "north": north,
        "west": west, "east": east,
        "outputFormat": "GTiff",
        "API_Key": api_key,
    }
    print(f"  Downloading DEM ({DEM_DATASET}) [{south:.4f},{west:.4f}] to [{north:.4f},{east:.4f}]...")
    resp = requests.get(OPENTOPO_BASE, params=params, stream=True, timeout=120)
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
    size_kb = output_path.stat().st_size // 1024
    print(f"  downloaded DEM ({size_kb} KB) to {output_path}")
    return output_path


def read_dem_grid(tif_path, lat_min, lat_max, lon_min, lon_max, n=TERRAIN_GRID_N):
    # resample DEM to an nxn grid, returns (lon2d, lat2d, elev)
    grid_lons = np.linspace(lon_min, lon_max, n)
    grid_lats = np.linspace(lat_max, lat_min, n)  # top-to-bottom for raster
    lon2d, lat2d = np.meshgrid(grid_lons, grid_lats)

    with rasterio.open(tif_path) as ds:
        nodata = ds.nodata
        band = ds.read(1).astype(float)
        if nodata is not None:
            band[band == nodata] = np.nan

        rows, cols = rasterio.transform.rowcol(
            ds.transform,
            lon2d.ravel(), lat2d.ravel()
        )
        rows = np.clip(np.array(rows), 0, ds.height - 1)
        cols = np.clip(np.array(cols), 0, ds.width  - 1)
        elev = band[rows, cols].reshape(n, n)

    mask = np.isnan(elev)
    if mask.any():
        idx = distance_transform_edt(mask, return_distances=False, return_indices=True)
        elev[mask] = elev[idx[0][mask], idx[1][mask]]

    print(f"  terrain grid {n}x{n}, "
          f"elev [{np.nanmin(elev):.0f}-{np.nanmax(elev):.0f}] m")
    return lon2d, lat2d, elev


def sample_elevation_at_points(tif_path, lats, lons, radius_px=2):
    # sample DEM at each route point, take max in a small neighborhood
    # so the route never dips below the terrain mesh surface
    with rasterio.open(tif_path) as ds:
        nodata = ds.nodata
        band = ds.read(1).astype(float)
        if nodata is not None:
            band[band == nodata] = np.nan
        rows, cols = rasterio.transform.rowcol(ds.transform, lons, lats)
        rows = np.clip(np.array(rows), 0, ds.height - 1)
        cols = np.clip(np.array(cols), 0, ds.width  - 1)
        alts = np.zeros(len(lats))
        r = radius_px
        for i, (row, col) in enumerate(zip(rows, cols)):
            patch = band[
                max(0, row - r):min(ds.height, row + r + 1),
                max(0, col - r):min(ds.width,  col + r + 1)
            ]
            valid = patch[~np.isnan(patch)]
            alts[i] = valid.max() if len(valid) else band[row, col]
    return alts


def geo_to_mm(lats, lons, lat_min, lat_max, lon_min, lon_max,
              width_mm, depth_mm):
    lat_range = lat_max - lat_min or 1e-9
    lon_range = lon_max - lon_min or 1e-9
    x = (lons - lon_min) / lon_range * width_mm
    y = (lats - lat_min) / lat_range * depth_mm
    return x, y


def elev_to_mm(elev, elev_min, elev_max, terrain_height_mm, base_thickness_mm):
    elev_range = elev_max - elev_min or 1.0
    z = base_thickness_mm + (elev - elev_min) / elev_range * terrain_height_mm
    # Clamp: surface vertices must never land exactly at z_bot.
    # If they do, surface edges become coplanar with bottom face edges,
    # causing 4 triangles per edge (non-manifold) instead of 2.
    return np.maximum(z, base_thickness_mm + 0.01)


def write_binary_stl(triangles, path):
    # triangles is (N,3,3) float32 - writes binary STL format
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
            f.write(struct.pack("<H", 0))
    print(f"STL saved: {path}  ({n:,} triangles, "
          f"{path.stat().st_size//1024} KB)")


def _triangles_to_xml_mesh(triangles, obj_id):
    verts = triangles.reshape(-1, 3)
    quantized = np.round(verts, decimals=4)
    unique, inverse = np.unique(quantized, axis=0, return_inverse=True)
    vert_lines = []
    for v in unique:
        vert_lines.append(f'          <vertex x="{v[0]}" y="{v[1]}" z="{v[2]}" />')
    tri_lines = []
    for t in range(len(triangles)):
        v1, v2, v3 = inverse[t*3], inverse[t*3+1], inverse[t*3+2]
        tri_lines.append(f'          <triangle v1="{v1}" v2="{v2}" v3="{v3}" />')
    return (
        f'    <object id="{obj_id}" type="model">\n'
        f'      <mesh>\n'
        f'        <vertices>\n'
        + "\n".join(vert_lines) + "\n"
        f'        </vertices>\n'
        f'        <triangles>\n'
        + "\n".join(tri_lines) + "\n"
        f'        </triangles>\n'
        f'      </mesh>\n'
        f'    </object>'
    )


def write_3mf(terrain_tris, route_tris, output_path):
    # pack all 3 meshes into a .3mf with extruder assignments
    # baseplate + terrain = extruder 1, route = extruder 2
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
        '  <Default Extension="rels" ContentType='
        '"application/vnd.openxmlformats-package.relationships+xml" />\n'
        '  <Default Extension="model" ContentType='
        '"application/vnd.ms-package.3dmanufacturing-3dmodel+xml" />\n'
        '</Types>'
    )

    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        '  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type='
        '"http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel" />\n'
        '</Relationships>'
    )
    obj2 = _triangles_to_xml_mesh(terrain_tris, 2)
    obj3 = _triangles_to_xml_mesh(route_tris, 3)

    model = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<model unit="millimeter" xml:lang="en-US"\n'
        '  xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">\n'
        '  <resources>\n'
        + obj2 + "\n"
        + obj3 + "\n"
        '  </resources>\n'
        '  <build>\n'
        '    <item objectid="2" />\n'
        '    <item objectid="3" />\n'
        '  </build>\n'
        '</model>'
    )

    model_settings = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<config>\n'
        '  <object id="2">\n'
        '    <metadata key="name" value="terrain" />\n'
        '    <metadata key="extruder" value="1" />\n'
        '  </object>\n'
        '  <object id="3">\n'
        '    <metadata key="name" value="route" />\n'
        '    <metadata key="extruder" value="2" />\n'
        '  </object>\n'
        '</config>'
    )

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("3D/3dmodel.model", model)
        zf.writestr("Metadata/model_settings.config", model_settings)

    size_kb = output_path.stat().st_size // 1024
    print(f"3MF saved: {output_path}  ({size_kb} KB)")


def build_baseplate(width_mm, depth_mm, base_thickness_mm):
    W, D, H = width_mm, depth_mm, base_thickness_mm
    tris = []
    tris += [[[0,0,0],[W,0,0],[W,D,0]],        # bottom
             [[0,0,0],[W,D,0],[0,D,0]]]
    tris += [[[0,0,H],[W,D,H],[W,0,H]],        # top
             [[0,0,H],[0,D,H],[W,D,H]]]
    tris += [[[0,0,0],[W,0,H],[W,0,0]],        # front
             [[0,0,0],[0,0,H],[W,0,H]]]
    tris += [[[0,D,0],[W,D,0],[W,D,H]],        # back
             [[0,D,0],[W,D,H],[0,D,H]]]
    tris += [[[0,0,0],[0,D,0],[0,D,H]],        # left
             [[0,0,0],[0,D,H],[0,0,H]]]
    tris += [[[W,0,0],[W,D,H],[W,D,0]],        # right
             [[W,0,0],[W,0,H],[W,D,H]]]
    return np.array(tris, dtype=np.float32)

def weld_vertices(tris, tol=1e-3):
    verts = tris.reshape(-1,3)

    quant = np.round(verts / tol)
    uniq, inverse = np.unique(quant, axis=0, return_inverse=True)

    welded = uniq * tol
    tris_welded = welded[inverse].reshape(-1,3,3)

    return tris_welded.astype(np.float32)

def build_terrain(lon2d, lat2d, elev_grid,
                  lat_min, lat_max, lon_min, lon_max,
                  width_mm, depth_mm,
                  elev_min, elev_max,
                  base_thickness_mm, terrain_height_mm):

    n = elev_grid.shape[0]

    x2d, y2d = geo_to_mm(
        lat2d, lon2d,
        lat_min, lat_max, lon_min, lon_max,
        width_mm, depth_mm
    )

    z2d = elev_to_mm(
        elev_grid,
        elev_min, elev_max,
        terrain_height_mm,
        base_thickness_mm
    )

    z_bot = base_thickness_mm

    tris = []

    # -----------------------------
    # Top surface
    # -----------------------------

    for i in range(n-1):
        for j in range(n-1):

            v00 = [x2d[i,j], y2d[i,j], z2d[i,j]]
            v10 = [x2d[i+1,j], y2d[i+1,j], z2d[i+1,j]]
            v01 = [x2d[i,j+1], y2d[i,j+1], z2d[i,j+1]]
            v11 = [x2d[i+1,j+1], y2d[i+1,j+1], z2d[i+1,j+1]]

            tris.append([v00,v10,v11])
            tris.append([v00,v11,v01])

    # -----------------------------
    # Bottom
    # -----------------------------

    for i in range(n-1):
        for j in range(n-1):

            v00 = [x2d[i,j], y2d[i,j], z_bot]
            v10 = [x2d[i+1,j], y2d[i+1,j], z_bot]
            v01 = [x2d[i,j+1], y2d[i,j+1], z_bot]
            v11 = [x2d[i+1,j+1], y2d[i+1,j+1], z_bot]

            tris.append([v00,v11,v10])
            tris.append([v00,v01,v11])

    # -----------------------------
    # Walls
    # -----------------------------

    # front
    i = n-1
    for j in range(n-1):

        t0=[x2d[i,j],y2d[i,j],z2d[i,j]]
        t1=[x2d[i,j+1],y2d[i,j+1],z2d[i,j+1]]

        b0=[x2d[i,j],y2d[i,j],z_bot]
        b1=[x2d[i,j+1],y2d[i,j+1],z_bot]

        tris.append([t0,b0,b1])
        tris.append([t0,b1,t1])

    # back
    i = 0
    for j in range(n-1):

        t0=[x2d[i,j],y2d[i,j],z2d[i,j]]
        t1=[x2d[i,j+1],y2d[i,j+1],z2d[i,j+1]]

        b0=[x2d[i,j],y2d[i,j],z_bot]
        b1=[x2d[i,j+1],y2d[i,j+1],z_bot]

        tris.append([t0,b1,b0])
        tris.append([t0,t1,b1])

    # left
    j = 0
    for i in range(n-1):

        t0=[x2d[i,j],y2d[i,j],z2d[i,j]]
        t1=[x2d[i+1,j],y2d[i+1,j],z2d[i+1,j]]

        b0=[x2d[i,j],y2d[i,j],z_bot]
        b1=[x2d[i+1,j],y2d[i+1,j],z_bot]

        tris.append([t0,b1,b0])
        tris.append([t0,t1,b1])

    # right
    j = n-1
    for i in range(n-1):

        t0=[x2d[i,j],y2d[i,j],z2d[i,j]]
        t1=[x2d[i+1,j],y2d[i+1,j],z2d[i+1,j]]

        b0=[x2d[i,j],y2d[i,j],z_bot]
        b1=[x2d[i+1,j],y2d[i+1,j],z_bot]

        tris.append([t0,b0,b1])
        tris.append([t0,b1,t1])

    raw = np.array(tris, dtype=np.float32)
    return weld_vertices(raw)


def carve_trench_into_grid(elev_grid, x2d, y2d,
                            route_x, route_y,
                            route_width_mm,
                            elev_min):
    """Stamp a rectangular trench into the elevation grid along the route.

    For each route segment we zero-out (set to elev_min) every grid cell
    whose centre falls within half the route width of the segment centreline.
    elev_min maps to base_thickness_mm in elev_to_mm, so the trench floor
    sits at the very bottom of the terrain solid — exactly where the route
    piece slots in.

    A small clearance (10 % of route width) is added so the trench is
    fractionally wider than the ribbon, giving a snug but printable fit.
    """
    clearance = route_width_mm * 0.30
    half_w    = route_width_mm / 2.0 + clearance

    n_pts = len(route_x)
    for i in range(n_pts - 1):
        ax, ay = route_x[i],     route_y[i]
        bx, by = route_x[i + 1], route_y[i + 1]

        seg_dx = bx - ax
        seg_dy = by - ay
        seg_len = math.sqrt(seg_dx**2 + seg_dy**2)
        if seg_len < 1e-9:
            continue

        tx, ty = seg_dx / seg_len, seg_dy / seg_len   # unit tangent
        nx, ny = -ty, tx                               # unit normal

        gx = x2d - ax
        gy = y2d - ay

        along  =  gx * tx + gy * ty
        across =  gx * nx + gy * ny

        mask = (along >= 0) & (along <= seg_len) & (np.abs(across) <= half_w)
        elev_grid[mask] = elev_min

    return elev_grid


def build_route(route_x, route_y, route_z_terrain,
                route_raise_mm, route_width_mm, sides=ROUTE_TUBE_SIDES):
    """Build the route ribbon as a watertight solid.

    The ribbon extends from z=0 (the model floor) up to the terrain surface
    plus route_raise_mm.  This lets it fill the full depth of the trench
    carved into the terrain so the two pieces interlock when assembled.
    """
    half_w = route_width_mm / 2.0
    n = len(route_x)
    if n < 2:
        return np.zeros((0, 3, 3), dtype=np.float32)

    # compute cross-section at each point using averaged tangent direction
    rings = []
    for i in range(n):
        if i == 0:
            dx, dy = route_x[1] - route_x[0], route_y[1] - route_y[0]
        elif i == n - 1:
            dx, dy = route_x[-1] - route_x[-2], route_y[-1] - route_y[-2]
        else:
            dx = route_x[i+1] - route_x[i-1]
            dy = route_y[i+1] - route_y[i-1]

        seg_len = math.sqrt(dx * dx + dy * dy)
        if seg_len < 1e-9:
            rings.append(rings[-1] if rings else None)
            continue

        px = -dy / seg_len * half_w
        py =  dx / seg_len * half_w
        zb = 0.0                                   # bottom at model floor
        zt = route_z_terrain[i] + route_raise_mm   # top proud of terrain

        rings.append([
            [route_x[i] + px, route_y[i] + py, zb],  # bottom-left
            [route_x[i] - px, route_y[i] - py, zb],  # bottom-right
            [route_x[i] - px, route_y[i] - py, zt],  # top-right
            [route_x[i] + px, route_y[i] + py, zt],  # top-left
        ])

    tris = []

    # start cap
    if rings[0] is not None:
        bl, br, tr, tl = rings[0]
        tris.append([bl, br, tr])
        tris.append([bl, tr, tl])

    # connect adjacent rings
    for i in range(n - 1):
        if rings[i] is None or rings[i+1] is None:
            continue
        bl0, br0, tr0, tl0 = rings[i]
        bl1, br1, tr1, tl1 = rings[i+1]

        tris.append([tl0, tr0, tr1]); tris.append([tl0, tr1, tl1])  # top
        tris.append([br0, bl0, bl1]); tris.append([br0, bl1, br1])  # bottom
        tris.append([tl0, tl1, bl1]); tris.append([tl0, bl1, bl0])  # left
        tris.append([tr0, br0, br1]); tris.append([tr0, br1, tr1])  # right

    # end cap
    if rings[-1] is not None:
        bl, br, tr, tl = rings[-1]
        tris.append([br, bl, tl])
        tris.append([br, tl, tr])

    return np.array(tris, dtype=np.float32)




def subsample_route(lats, lons, alts, max_points=500):
    n = len(lats)
    if n <= max_points:
        return lats, lons, alts
    idx = np.round(np.linspace(0, n - 1, max_points)).astype(int)
    return lats[idx], lons[idx], alts[idx]


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate multi-color 3D STL from Strava activity")
    p.add_argument("--activity-id", type=int, default=None,
                   help="Strava activity ID (omit for most recent)")
    p.add_argument("--client-id", default=os.environ.get("STRAVA_CLIENT_ID"))
    p.add_argument("--client-secret", default=os.environ.get("STRAVA_CLIENT_SECRET"))
    p.add_argument("--refresh-token", default=os.environ.get("STRAVA_REFRESH_TOKEN"))
    p.add_argument("--output-dir", default="./stl_output")
    p.add_argument("--width-mm", type=float, default=DEFAULT_WIDTH_MM)
    p.add_argument("--depth-mm", type=float, default=DEFAULT_DEPTH_MM)
    p.add_argument("--base-mm", type=float, default=BASE_THICKNESS_MM)
    p.add_argument("--terrain-mm", type=float, default=TERRAIN_HEIGHT_MM)
    p.add_argument("--route-raise-mm", type=float, default=ROUTE_RAISE_MM)
    p.add_argument("--route-width-mm", type=float, default=ROUTE_WIDTH_MM)
    p.add_argument("--terrain-grid", type=int, default=TERRAIN_GRID_N)
    p.add_argument("--margin-deg", type=float, default=0.02,
                   help="extra margin around route in degrees")
    p.add_argument("--dem-tif", default=None, help="path to existing DEM tif")
    p.add_argument("--no-dem", action="store_true",
                   help="skip DEM, use strava altitude only")
    p.add_argument("--opentopo-key", default=os.environ.get("OPENTOPO_API_KEY"),
                   help="OpenTopography API key")
    p.add_argument("--no-3mf", action="store_true", help="skip .3mf generation")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.client_id or not args.client_secret or not args.refresh_token:
        print("ERROR: need --client-id, --client-secret, and --refresh-token")
        print("  (or set STRAVA_CLIENT_ID, STRAVA_CLIENT_SECRET, STRAVA_REFRESH_TOKEN)")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/6] Authenticating with Strava ...")
    token = strava_get_token(args.client_id, args.client_secret, args.refresh_token)

    if args.activity_id is None:
        print("  No --activity-id specified, fetching most recent activity ...")
        args.activity_id = strava_get_latest_activity_id(token)

    print("\n[2/6] Fetching activity streams ...")
    activity  = strava_get_activity(args.activity_id, token)
    streams   = strava_get_streams(args.activity_id, token)
    lats, lons, alts_strava, dists = extract_route_coords(streams)
    save_gpx(lats, lons, alts_strava, out_dir / f"activity_{args.activity_id}.gpx")

    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()

    if args.no_dem:
        print("\n[3/6] Using Strava altitude stream (--no-dem specified) ...")
        alts = alts_strava
        n = args.terrain_grid
        grid_lons = np.linspace(lon_min - args.margin_deg,
                                lon_max + args.margin_deg, n)
        grid_lats = np.linspace(lat_max + args.margin_deg,
                                lat_min - args.margin_deg, n)
        lon2d, lat2d = np.meshgrid(grid_lons, grid_lats)
        elev_grid = np.full((n, n), np.mean(alts) if len(alts) else 0.0)
        tif_path = None
    else:
        print("\n[3/6] Downloading/reading DEM elevation data ...")
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
              f"[{alts.min():.0f}-{alts.max():.0f}] m")

    elev_min = min(elev_grid.min(), alts.min())
    elev_max = max(elev_grid.max(), alts.max())

    lat_min_t = lat2d.min()
    lat_max_t = lat2d.max()
    lon_min_t = lon2d.min()
    lon_max_t = lon2d.max()

    print("\n[4/6] Converting coordinates to mm ...")
    lats_sub, lons_sub, alts_sub = subsample_route(lats, lons, alts)

    route_x, route_y = geo_to_mm(
        lats_sub, lons_sub,
        lat_min_t, lat_max_t, lon_min_t, lon_max_t,
        args.width_mm, args.depth_mm
    )

    # interpolate route elevation from terrain grid so route Z
    # matches the mesh surface exactly
    grid_lats_1d = lat2d[:, 0]
    grid_lons_1d = lon2d[0, :]
    interp = RegularGridInterpolator(
        (grid_lats_1d[::-1], grid_lons_1d),
        elev_grid[::-1],
        method='linear', bounds_error=False, fill_value=None
    )
    route_elev_on_grid = interp(np.column_stack([lats_sub, lons_sub]))

    route_z_terrain = elev_to_mm(
        route_elev_on_grid, elev_min, elev_max,
        args.terrain_mm, args.base_mm
    )
    print(f"  Route mm extent: "
          f"x[{route_x.min():.1f}-{route_x.max():.1f}] "
          f"y[{route_y.min():.1f}-{route_y.max():.1f}] "
          f"z[{route_z_terrain.min():.1f}-{route_z_terrain.max():.1f}]")

    print("\n[5/6] Building STL meshes ...")

    # ------------------------------------------------------------------
    # Build the route ribbon first (uses the unmodified elevation grid
    # so its Z profile matches the original terrain surface).
    # ------------------------------------------------------------------
    print("  building route ribbon...")
    route_tris = build_route(
        route_x, route_y, route_z_terrain,
        args.route_raise_mm, args.route_width_mm
    )

    # ------------------------------------------------------------------
    # Carve a matching trench into the elevation grid, then build the
    # terrain around it.  The trench floor sits at elev_min which maps
    # to base_thickness_mm, so the terrain solid has an open channel
    # that the route piece slots into exactly.
    # ------------------------------------------------------------------
    print("  carving route trench into elevation grid...")

    # geo_to_mm needs the terrain grid bounds to compute x2d/y2d
    n_grid = elev_grid.shape[0]
    x2d_grid, y2d_grid = geo_to_mm(
        lat2d, lon2d,
        lat_min_t, lat_max_t, lon_min_t, lon_max_t,
        args.width_mm, args.depth_mm
    )

    elev_grid_carved = elev_grid.copy()
    elev_grid_carved = carve_trench_into_grid(
        elev_grid_carved, x2d_grid, y2d_grid,
        route_x, route_y,
        args.route_width_mm,
        elev_min
    )

    print("  building terrain (with trench)...")
    terrain_minus_route_tris = build_terrain(
        lon2d, lat2d, elev_grid_carved,
        lat_min_t, lat_max_t, lon_min_t, lon_max_t,
        args.width_mm, args.depth_mm,
        elev_min, elev_max,
        args.base_mm, args.terrain_mm,
    )

    print("\n[6/6] Writing STL files ...")
    write_binary_stl(route_tris,               out_dir / "route.stl")
    write_binary_stl(terrain_minus_route_tris,  out_dir / "terrain_minus_route.stl")

    if not args.no_3mf:
        write_3mf(terrain_minus_route_tris, route_tris,
                  out_dir / "output.3mf")

    print(f"\ndone! files in {out_dir}/")
    print(f"  route.stl               — print in accent colour")
    print(f"  terrain_minus_route.stl — print in base colour")
    print(f"  Drop route into the channel in terrain_minus_route to assemble.")
    if not args.no_3mf:
        print(f"  output.3mf = terrain_minus_route (obj 2) + route (obj 3)")
    print(f"  model size: {args.width_mm:.0f}x{args.depth_mm:.0f}x"
          f"{args.base_mm+args.terrain_mm+args.route_raise_mm:.0f}mm")


if __name__ == "__main__":
    main()
