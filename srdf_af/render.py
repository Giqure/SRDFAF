"""Render perspective images from Matterport3D skybox cube maps.

Each MP3D viewpoint has 6 skybox face images forming a cube map.
This module projects them into heading-aligned perspective views
for use as VLM input during instruction generation and judging.
"""

import math
import zipfile
from pathlib import Path

import cv2
import numpy as np

# ── Skybox Face Mapping ───────────────────────────────────────────────
# Matterport3D skybox files: {viewpoint}_skybox{0..5}_sami.jpg
# Mapping to standard cube faces.  Adjust if rendered images look wrong.

SKYBOX_INDEX: dict[str, int] = {
    "+X": 3,   # right
    "-X": 1,   # left
    "+Y": 4,   # up
    "-Y": 5,   # down
    "+Z": 2,   # front
    "-Z": 0,   # back
}


# ── Skybox Loading ────────────────────────────────────────────────────


def extract_skybox(skybox_dir: str, scan: str) -> None:
    """Extract skybox zip in-place if not already extracted.

    MP3D zips use a flat layout::

        {scan}//matterport_skybox_images/{vp}_skybox{N}_sami.jpg

    We extract to ``{skybox_dir}/{scan}/matterport_skybox_images/``.
    """
    scan_dir = Path(skybox_dir) / scan
    zip_path = scan_dir / "matterport_skybox_images.zip"
    img_dir = scan_dir / "matterport_skybox_images"

    # Skip if already extracted (check for any .jpg files)
    if img_dir.is_dir() and any(img_dir.glob("*_skybox*_sami.jpg")):
        return
    if not zip_path.exists():
        return

    img_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".jpg") or "skybox" not in name:
                continue
            # Extract just the filename, ignore any directory prefix
            fname = name.split("/")[-1]
            target = img_dir / fname
            if not target.exists():
                target.write_bytes(zf.read(name))


def load_skybox(skybox_dir: str, scan: str, viewpoint: str) -> dict[str, np.ndarray] | None:
    """Load 6 cube-map face images for a viewpoint.

    MP3D stores skybox images in a flat directory::

        matterport_skybox_images/{viewpoint}_skybox{N}_sami.jpg

    Returns dict keyed by "+X", "-X", "+Y", "-Y", "+Z", "-Z" or None on failure.
    """
    img_dir = Path(skybox_dir) / scan / "matterport_skybox_images"
    if not img_dir.is_dir():
        return None

    faces: dict[str, np.ndarray] = {}
    for face_key, skybox_idx in SKYBOX_INDEX.items():
        img_path = img_dir / f"{viewpoint}_skybox{skybox_idx}_sami.jpg"
        if not img_path.exists():
            return None
        img = cv2.imread(str(img_path))
        if img is None:
            return None
        faces[face_key] = img
    return faces


# ── Cube Map → Perspective Projection ─────────────────────────────────


def cube_to_perspective(
    faces: dict[str, np.ndarray],
    heading: float,
    elevation: float = 0.0,
    fov_deg: float = 90.0,
    size: int = 480,
) -> np.ndarray:
    """Project cube map to a perspective view at given heading/elevation.

    Uses the OpenGL cube-map UV specification for correct face sampling.

    Args:
        faces: dict with keys "+X", "-X", "+Y", "-Y", "+Z", "-Z"
        heading: horizontal rotation in radians (0 = +Z / front, positive = rightward)
        elevation: vertical rotation in radians (positive = upward)
        fov_deg: field of view in degrees
        size: output image size (square)
    """
    fov = math.radians(fov_deg)
    f = size / (2 * math.tan(fov / 2))

    # Pixel grid → camera-space ray directions (x-right, y-up, z-forward)
    j, i = np.meshgrid(np.arange(size), np.arange(size))
    x_cam = j.astype(np.float64) - size / 2
    y_cam = size / 2 - i.astype(np.float64)
    z_cam = np.full_like(x_cam, f)

    # Rotate: Ry(heading) · Rx(elevation) · ray
    ch, sh = math.cos(heading), math.sin(heading)
    ce, se = math.cos(elevation), math.sin(elevation)

    # Rx(elevation)
    x1 = x_cam
    y1 = ce * y_cam - se * z_cam
    z1 = se * y_cam + ce * z_cam

    # Ry(heading)
    rx = ch * x1 + sh * z1
    ry = y1
    rz = -sh * x1 + ch * z1

    # Determine dominant cube face per pixel
    ax, ay, az = np.abs(rx), np.abs(ry), np.abs(rz)
    abs_stack = np.stack([ax, ay, az], axis=-1)
    dominant = np.argmax(abs_stack, axis=-1)  # 0=x, 1=y, 2=z

    output = np.zeros((size, size, 3), dtype=np.uint8)

    # OpenGL cube-map UV spec: face → (sc, tc) expressions
    face_specs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = [
        ("+X", (dominant == 0) & (rx > 0),  -rz, -ry),
        ("-X", (dominant == 0) & (rx <= 0),  rz, -ry),
        ("+Y", (dominant == 1) & (ry > 0),   rx,  rz),
        ("-Y", (dominant == 1) & (ry <= 0),  rx, -rz),
        ("+Z", (dominant == 2) & (rz > 0),   rx, -ry),
        ("-Z", (dominant == 2) & (rz <= 0), -rx, -ry),
    ]

    ma_vals = np.where(dominant == 0, ax, np.where(dominant == 1, ay, az))

    for face_key, mask, sc_full, tc_full in face_specs:
        if not np.any(mask):
            continue

        face_img = faces[face_key]
        fh, fw = face_img.shape[:2]

        ma = ma_vals[mask]
        sc = sc_full[mask]
        tc = tc_full[mask]

        # UV in [0, 1]
        u = (sc / ma + 1.0) * 0.5
        v = (tc / ma + 1.0) * 0.5

        px = np.clip((u * (fw - 1)).astype(np.int32), 0, fw - 1)
        py = np.clip((v * (fh - 1)).astype(np.int32), 0, fh - 1)

        output[mask] = face_img[py, px]

    return output


# ── Batch Rendering ───────────────────────────────────────────────────


def render_trajectory(
    skybox_dir: str,
    scan: str,
    viewpoints: list[str],
    headings: list[float],
    output_dir: str,
    fov: float = 90.0,
    size: int = 480,
) -> list[str]:
    """Render perspective images for one trajectory. Returns output paths."""
    out_scan = Path(output_dir) / scan
    out_scan.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    for vid, heading in zip(viewpoints, headings):
        deg = round(math.degrees(heading)) % 360
        out_path = out_scan / f"{vid}_{deg}.jpg"

        if out_path.exists():
            paths.append(str(out_path))
            continue

        faces = load_skybox(skybox_dir, scan, vid)
        if faces is None:
            continue

        img = cube_to_perspective(faces, heading, fov_deg=fov, size=size)
        cv2.imwrite(str(out_path), img)
        paths.append(str(out_path))

    return paths


def render_all(
    r2r_data: list[dict],
    skybox_dir: str,
    conn_dir: str,
    output_dir: str,
    fov: float = 90.0,
    size: int = 480,
) -> int:
    """Render all perspective images needed for R2R trajectories.

    Deduplicates by (scan, viewpoint, heading_deg) so shared viewpoints
    are rendered only once.  Returns total number of images rendered.
    """
    from srdf_af.data import load_connectivity, trajectory_headings

    # Pass 1: Collect unique (scan, viewpoint, heading_deg) tuples
    tasks: dict[tuple[str, str, int], float] = {}
    conn_cache: dict[str, dict] = {}
    scans_to_extract: set[str] = set()

    for entry in r2r_data:
        scan = entry["scan"]
        if scan not in conn_cache:
            try:
                conn_cache[scan] = load_connectivity(conn_dir, scan)
            except FileNotFoundError:
                continue
            scans_to_extract.add(scan)

        headings = trajectory_headings(
            conn_cache[scan], entry["path"], entry.get("heading", 0.0)
        )
        for vid, h in zip(entry["path"], headings):
            deg = round(math.degrees(h)) % 360
            tasks.setdefault((scan, vid, deg), h)

    print(f"Unique views to render: {len(tasks)} across {len(conn_cache)} scans")

    # Pass 2: Extract skybox zips (idempotent)
    for scan in sorted(scans_to_extract):
        extract_skybox(skybox_dir, scan)

    # Pass 3: Render
    rendered = 0
    skipped = 0
    for (scan, vid, deg), heading in tasks.items():
        out_path = Path(output_dir) / scan / f"{vid}_{deg}.jpg"
        if out_path.exists():
            skipped += 1
            continue

        faces = load_skybox(skybox_dir, scan, vid)
        if faces is None:
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        img = cube_to_perspective(faces, heading, fov_deg=fov, size=size)
        cv2.imwrite(str(out_path), img)
        rendered += 1

        if rendered % 500 == 0:
            print(f"  rendered {rendered}  (skipped {skipped} existing)")

    print(f"Done: rendered {rendered}, skipped {skipped}, total {rendered + skipped}/{len(tasks)}")
    return rendered
