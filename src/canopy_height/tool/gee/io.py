"""Earth Engine download on an exact grid.

getDownloadURL chunks with crs + crs_transform + dimensions, so the pixels land on the analysis grid and nothing is
resampled after the download; the chunks are stitched into the full raster ('mosaic'), a chunk that hits the memory
/ size limit is split 2 x 2. 429 / 5xx answers are retried after a 15-20 s back-off, and so are getInfo calls
(a project over its Earth Engine quota runs in restricted mode and answers 'Too Many Requests' to any call).
Float layers are unmasked to SENTINEL before the download and SENTINEL becomes NaN.
CANCEL (set by the download stage on Ctrl+C) stops worker threads at their next chunk or retry wait; the main
thread is stopped by the KeyboardInterrupt itself.
"""
import random
import threading
import time

import numpy as np

RETRIES, RETRY_WAIT, TIMEOUT, BUSY_MAX, MIN_CHUNK = 3, 15, 900, 40, 64
SENTINEL = -9999.0
TAG = "canopy-height-tool"
AUTH_HINT = ("Authenticate once with:  earthengine authenticate  (and check that the Google Cloud project is "
             "registered for Earth Engine: https://code.earthengine.google.com/register).")
CANCEL = threading.Event()
_ee = None
_project = None


class TooBig(Exception):
    pass


class Cancelled(Exception):
    pass


def _check_cancel():
    if CANCEL.is_set() and threading.current_thread() is not threading.main_thread():
        raise Cancelled("download cancelled")


def _sleep(sec):
    """time.sleep in short steps that ends with Cancelled once CANCEL is set (worker threads only)."""
    end = time.monotonic() + sec
    while True:
        _check_cancel()
        left = end - time.monotonic()
        if left <= 0:
            return
        time.sleep(min(left, 0.5))


def ee_init(project=None, tag=TAG):
    """Initialise Earth Engine for a Google Cloud project (once per process) and return the `ee` module.

    project=None reuses the project of an earlier call. getInfo is patched to retry on 429 / 5xx.
    """
    global _ee, _project
    if _ee is not None and (not project or project == _project):
        return _ee
    if not project:
        raise RuntimeError("no Earth Engine project: set gee_project in project.yaml (a Google Cloud project "
                           "registered for Earth Engine). " + AUTH_HINT)
    try:
        import ee
    except ImportError as e:
        raise RuntimeError("earthengine-api is not installed in this Python environment") from e
    try:
        retry(lambda: ee.Initialize(project=project))
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Earth Engine initialisation failed for project {project!r}: {e}. " + AUTH_HINT) from e
    ee.data.setWorkloadTag(tag)
    _gi = ee.ComputedObject.getInfo
    if not getattr(ee.ComputedObject, "_chm_tool_retry", False):
        ee.ComputedObject.getInfo = lambda self, *a, **k: retry(lambda: _gi(self, *a, **k))
        ee.ComputedObject._chm_tool_retry = True
    _ee, _project = ee, project
    return ee


def ee_check(project):
    """Initialise and send one tiny request; raises RuntimeError with a hint when the project cannot be used."""
    ee = ee_init(project)
    try:
        ee.Number(1).getInfo()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Earth Engine request failed for project {project!r}: {e}. " + AUTH_HINT) from e
    return f"Earth Engine ready (project {project})"


def _transient(msg):
    m = msg.lower()
    return any(k in m for k in ("too many", "concurrency", "429", "500", "502", "503", "504", "please try again",
                                "unavailable", "timed out"))


def backoff():
    _sleep(15 + random.random() * 5)


def retry(fn, n=BUSY_MAX):
    for i in range(n):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if i == n - 1 or not _transient(str(e)):
                raise
            backoff()


def fetch(image, crs, x0, y1, res, w, h, nb):
    """One chunk -> float64 [nb, h, w], SENTINEL -> NaN."""
    import requests
    from rasterio.io import MemoryFile
    last, tries, busy = None, 0, 0
    while tries < RETRIES:
        _check_cancel()
        try:
            url = image.getDownloadURL({"crs": crs, "crs_transform": [res, 0, x0, 0, -res, y1],
                                        "dimensions": [w, h], "format": "GEO_TIFF"})
            r = requests.get(url, timeout=TIMEOUT)
            if r.status_code != 200:
                msg = r.text[:300]
                if "memory limit" in msg or "must be less than or equal to" in msg or "too large" in msg.lower():
                    raise TooBig(msg)
                if (r.status_code in (429, 500, 502, 503, 504) or "concurrency" in msg.lower()) and busy < BUSY_MAX:
                    busy += 1
                    backoff()
                    continue
                raise RuntimeError(f"HTTP {r.status_code}: {msg}")
            with MemoryFile(r.content) as mf, mf.open() as d:
                t = d.transform
                assert (d.count, d.width, d.height) == (nb, w, h), (d.count, d.width, d.height, nb, w, h)
                assert abs(t.c - x0) < 1e-6 and abs(t.f - y1) < 1e-6 and abs(t.a - res) < 1e-9, (t, x0, y1)
                a = d.read().astype(np.float64)
            a[a == SENTINEL] = np.nan
            return a
        except (TooBig, Cancelled):
            raise
        except Exception as e:  # noqa: BLE001
            if "memory limit" in str(e) or "must be less than or equal to" in str(e):
                raise TooBig(str(e)) from e
            if _transient(str(e)) and busy < BUSY_MAX:
                busy += 1
                backoff()
                continue
            last = e
        tries += 1
        _sleep(RETRY_WAIT)
    raise RuntimeError(f"failed after {RETRIES} tries at ({x0},{y1},{w},{h}): {last}")


def fetch_split(image, crs, x0, y1, res, w, h, nb):
    try:
        return fetch(image, crs, x0, y1, res, w, h, nb)
    except TooBig:
        if min(w, h) <= MIN_CHUNK:
            raise
        out = np.empty((nb, h, w))
        hw, hh = (w + 1) // 2, (h + 1) // 2
        for dy, sh in ((0, hh), (hh, h - hh)):
            for dx, sw in ((0, hw), (hw, w - hw)):
                out[:, dy:dy + sh, dx:dx + sw] = fetch_split(image, crs, x0 + dx * res, y1 - dy * res, res, sw, sh, nb)
        return out


def compute_pixels(image, crs, transform, w, h):
    """computePixels (NUMPY_NDARRAY) of image on the grid (crs, affine transform, w x h) -> numpy structured array
    (one field per band, the band's own type). Retries like fetch; raises TooBig when the request is over Earth
    Engine's size limit (the caller splits it)."""
    a, b, c, d, e, f = [float(x) for x in transform]
    params = {"expression": image, "fileFormat": "NUMPY_NDARRAY",
              "grid": {"dimensions": {"width": int(w), "height": int(h)},
                       "affineTransform": {"scaleX": a, "shearX": b, "translateX": c,
                                           "shearY": d, "scaleY": e, "translateY": f},
                       "crsCode": crs}}
    last, tries, busy = None, 0, 0
    while tries < RETRIES:
        _check_cancel()
        try:
            return _ee.data.computePixels(params)
        except Cancelled:
            raise
        except Exception as ex:  # noqa: BLE001
            msg = str(ex)
            if "must be less than or equal to" in msg or "memory limit" in msg or "too large" in msg.lower():
                raise TooBig(msg) from ex
            if _transient(msg) and busy < BUSY_MAX:
                busy += 1
                backoff()
                continue
            last = ex
        tries += 1
        _sleep(RETRY_WAIT)
    raise RuntimeError(f"computePixels failed after {RETRIES} tries on {crs} {transform[2]:.1f},{transform[5]:.1f} "
                       f"{w}x{h}: {last}")


def download_grid(image, epsg, x0, y1, W, H, res, nb, chunk):
    """Whole grid, fetched in chunk x chunk pieces and stitched -> float64 [nb, H, W]."""
    full = np.empty((nb, H, W))
    for r0 in range(0, H, chunk):
        for c0 in range(0, W, chunk):
            _check_cancel()
            w, h = min(chunk, W - c0), min(chunk, H - r0)
            full[:, r0:r0 + h, c0:c0 + w] = fetch_split(image, f"EPSG:{epsg}", x0 + c0 * res, y1 - r0 * res, res,
                                                        w, h, nb)
    return full
