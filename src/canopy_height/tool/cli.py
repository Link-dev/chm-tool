"""Command line of the canopy-height tool: `chm-tool <command> --help` (= `python -m canopy_height.tool`).

    chm-tool init DIR --aoi study_area.shp --gee-project my-project     new project folder
    chm-tool run DIR [--from STAGE] [--to STAGE]                         plan, download, stack, train, predict, report
    chm-tool status DIR                                                  state of every stage
    chm-tool app [DIR] [--port 8501]                                     the Streamlit interface

Single stages: plan | download | stack | train | predict | report DIR. `import-cells DIR --rows CSV` replaces
the plan stage by a table of existing 256 x 256 rasters.
Settings not given to `init` keep the paper's defaults (project.DEFAULTS); edit DIR/project.yaml to change them.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from .project import ALIASES, BENCHMARKS, INPUTS, MODELS, STAGES, Project


def cmd_init(a):
    settings = dict(name=a.name, input=a.input, year=a.year, gee_project=a.gee_project, epsg=a.epsg,
                    train_cells=a.train_cells, ring_max_km=a.ring_max_km, models=a.models, benchmarks=a.benchmarks,
                    als=a.als, als_resolution=a.als_resolution, als_scale=a.als_scale, device=a.device,
                    workers=a.workers, base_model=a.base_model, s1_method=a.s1_method)
    p = Project.create(a.dir, a.aoi, **settings)
    print(f"project {p.root}")
    for k in ("input", "year", "gee_project", "s1_method", "epsg", "train_cells", "models", "benchmarks", "als"):
        print(f"  {k}: {p[k]}")


def cmd_stage(a):
    from . import workflow
    p = Project(a.dir)
    try:
        if a.cmd == "download":
            workflow.download(p, workers=a.workers)
        elif a.cmd == "train":
            workflow.train(p, models=a.models)
        else:
            getattr(workflow, a.cmd)(p)
    finally:
        p.close_log()


def cmd_run(a):
    from . import workflow
    p = Project(a.dir)
    try:
        workflow.run(p, a.from_stage, a.to_stage, workers=a.workers)
    finally:
        p.close_log()


def cmd_status(a):
    p = Project(a.dir)
    st = p.status()
    print(f"{'stage':9} {'state':8} {'started':19}  {'finished':19}  {'seconds':>8}  message")
    for s in STAGES:
        r = st.get(s, {})
        sec = r.get("seconds")
        print(f"{s:9} {r.get('state', 'pending'):8} {r.get('started', '') or '':19}  {r.get('finished', '') or '':19}  "
              f"{'' if sec is None else sec:>8}  {r.get('message', '') or ''}")
    print(f"log: {p.log_file}")


def cmd_import_cells(a):
    from . import workflow
    role_map = {}
    for kv in a.role_map:
        k, sep, v = kv.partition("=")
        if not sep or v not in ("aoi", "ring"):
            raise ValueError(f"--role-map expects KIND=aoi or KIND=ring, got {kv!r}")
        role_map[k] = v
    p = Project(a.dir)
    try:
        workflow.import_cells(p, a.rows, role_map=role_map, default_role=a.default_role)
    finally:
        p.close_log()


def cmd_app(a):
    app = Path(__file__).with_name("app.py")
    if not app.exists():
        raise FileNotFoundError(app)
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(a.port)]
    env = dict(os.environ, PYTHONNOUSERSITE="1")      # packages of the user site must not shadow the tool's
    return subprocess.call(cmd + (["--", str(Path(a.dir).resolve())] if a.dir else []), env=env)


def build_parser():
    ap = argparse.ArgumentParser(prog="chm-tool", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="create a project folder from a study area")
    p.add_argument("dir")
    p.add_argument("--aoi", required=True, help="study area: shapefile, .zip, GeoJSON, GeoPackage or KML")
    p.add_argument("--name")
    p.add_argument("--input", choices=list(INPUTS) + list(ALIASES),
                   help="input representation (default AE): " + "; ".join(f"{k} = {v['label']}" for k, v in INPUTS.items()))
    p.add_argument("--year", type=int, help="year of the annual inputs, 2019-2024 (default 2020)")
    p.add_argument("--gee-project", help="Google Cloud project registered for Earth Engine")
    p.add_argument("--s1-method", choices=["local", "gee"],
                   help="Sentinel-1 processing: local = raw scenes from Earth Engine, the chain on this computer "
                        "(default, ~1/10 of the Earth Engine compute); gee = on Earth Engine (the paper, bit for bit)")
    p.add_argument("--epsg", type=int, help="analysis CRS (default: UTM zone of the study area)")
    p.add_argument("--train-cells", type=int, help="size of the training region in 2.56 km cells (default 400)")
    p.add_argument("--ring-max-km", type=float, help="maximum distance of training cells (default 50 km)")
    p.add_argument("--models", nargs="+", choices=MODELS)
    p.add_argument("--benchmarks", nargs="+", choices=list(BENCHMARKS))
    p.add_argument("--als", nargs="+", help="ALS canopy-height rasters for an accuracy check")
    p.add_argument("--als-resolution", type=float, help="resolution of the ALS rasters in m (default 1)")
    p.add_argument("--als-scale", type=float, help="factor to metres of the ALS values (0.01 for centimetres)")
    p.add_argument("--base-model", help="UNet-ALS checkpoint (default: the bundled weights)")
    p.add_argument("--device", help="cuda:0, cpu, ... (default: GPU if available)")
    p.add_argument("--workers", type=int, help="parallel Earth Engine requests (default 3)")
    p.set_defaults(func=cmd_init)

    for s in STAGES:
        p = sub.add_parser(s, help=f"run the {s} stage")
        p.add_argument("dir")
        if s == "download":
            p.add_argument("--workers", type=int, help="parallel Earth Engine requests (default: project setting)")
        if s == "train":
            p.add_argument("--models", nargs="+", choices=MODELS, help="default: the project's models")
        p.set_defaults(func=cmd_stage)

    p = sub.add_parser("run", help="run stages in order (resumes finished parts)")
    p.add_argument("dir")
    p.add_argument("--from", dest="from_stage", choices=STAGES, default=STAGES[0])
    p.add_argument("--to", dest="to_stage", choices=STAGES, default=STAGES[-1])
    p.add_argument("--workers", type=int, help="parallel Earth Engine requests (default: project setting)")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("status", help="state of every stage")
    p.add_argument("dir")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("import-cells", help="use existing 256 x 256 rasters as the cells (replaces plan)")
    p.add_argument("dir")
    p.add_argument("--rows", required=True, help="CSV table of the rasters")
    p.add_argument("--role-map", nargs="+", default=["test=aoi"], help="KIND=ROLE pairs (default test=aoi)")
    p.add_argument("--default-role", choices=["aoi", "ring"], default="ring")
    p.set_defaults(func=cmd_import_cells)

    p = sub.add_parser("app", help="start the Streamlit interface")
    p.add_argument("dir", nargs="?", help="project folder to open (optional)")
    p.add_argument("--port", type=int, default=8501)
    p.set_defaults(func=cmd_app)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        return a.func(a) or 0
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"chm-tool {a.cmd}: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
