"""Earth Engine side of the canopy-height tool: a port of the paper's international-site data pipeline
(gee_io.py + gee_layers.py) with the site constants of its config.py turned into
explicit parameters. The requests are the pipeline's, unchanged, so a tool download of a pipeline cell reproduces
the pipeline raster bit for bit (Sentinel-1 up to ~1e-10 dB of GEE float-summation order).

  io      ee_init, retry / back-off, chunked getDownloadURL on an exact grid (fetch, fetch_split, download_grid)
  layers  image recipes of every layer (Embedding, DEM, S1, S2, GEDI, ETH, UMD, Tolan) and WorldCover shares

Submodules are not imported here; `layers` imports earthengine-api (no requests are sent on import).
"""
