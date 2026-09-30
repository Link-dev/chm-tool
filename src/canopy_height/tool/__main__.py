"""`python -m canopy_height.tool` = `chm-tool`. The guard matters: DataLoader workers started with 'spawn'
(Windows) re-import this module and must not run the command again."""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
