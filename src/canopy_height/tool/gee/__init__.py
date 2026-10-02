"""Earth Engine side of the canopy-height tool: the requests used for the paper's data, with the site constants as
explicit parameters, so a tool download of a paper cell reproduces the paper's raster bit for bit (Sentinel-1 up
to ~1e-10 dB of GEE float-summation order).

  io      ee_init, retry / back-off, chunked getDownloadURL on an exact grid (fetch, fetch_split, download_grid)
  layers  image recipes of every layer (Embedding, DEM, S1, S2, GEDI, ETH, UMD, Tolan) and WorldCover shares

Submodules are not imported here; `layers` imports earthengine-api (no requests are sent on import).
"""
