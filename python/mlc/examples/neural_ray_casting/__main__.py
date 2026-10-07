"""Neural ray casting: train models of what a camera sees in a grid world, export them to ONNX, and score them."""

import json
from pathlib import Path

import torch
from pydantic import Field
from pydantic_settings import CliApp, CliSubCommand

from mlc.core import emit
from mlc.core.cli import Command
from mlc.examples.neural_ray_casting.export import run, to_onnx
from mlc.examples.neural_ray_casting.training import CLASSES, VARIANTS, make_data, predict, scores, test_views, train
from mlc.examples.neural_ray_casting.world import REPO, load_worlds

NOTE = REPO / "content/notes/part-7-application-domains/games/neural-ray-casting"
MODELS_DIR = NOTE / "_models"
SWEEP_RESULTS = MODELS_DIR / "sweep.json"
TEST_VIEWS = 300


class Train(Command):
    """Train models on one map and save their weights; reports held-out scores as training goes."""

    world: str = Field("small-maze", description="Map id from world.ts, e.g. small-maze, columns, maze.")
    models: list[str] = Field(
        ["walls", "pe", "density"], description=f"Trained models to fit, from: {', '.join(VARIANTS)}."
    )
    steps: int = Field(3000, ge=1, description="Adam steps per model.")
    batch: int = Field(4096, ge=8, description="Rays per step (density models take an eighth).")
    train_rays: int = Field(200_000, ge=1000, description="Size of the training set of rays.")
    device: str = Field("cpu", description="Device for the density models: cpu, or mps for the Apple GPU.")
    out: Path = Field(Path("trained"), description="Directory for the weights, `<world>-<model>.pt`.")

    def cli_cmd(self) -> None:
        w = load_worlds()[self.world]
        data = make_data(w, self.train_rays, TEST_VIEWS)
        self.out.mkdir(parents=True, exist_ok=True)
        curves, rows = [], []
        for name in self.models:
            model, history = train(name, w, data, steps=self.steps, batch=self.batch, device=self.device)
            torch.save(model.state_dict(), self.out / f"{w.id}-{name}.pt")
            curves.append(emit.line(name, [h["step"] for h in history], [100 * h["within10"] for h in history]))
            last = history[-1]
            rows.append(
                [name, VARIANTS[name].family, 100 * last["within10"], 100 * last["median_rel"], last["seconds"]]
            )
        emit.table(
            ["model", "class", "rays within 10 % (%)", "median relative error (%)", "seconds"],
            rows,
            title=f"{w.id}: held-out views after {self.steps} steps",
        )
        emit.chart(*curves, title="Held-out rays within 10 %", x_label="step", y_label="%")


class Sweep(Command):
    """Run models on every map: reuse saved weights where they exist, train the rest, and report the best per map."""

    worlds: list[str] = Field([], description="Map ids to run; empty means every map in world.ts.")
    models: list[str] = Field(
        ["density-128", "walls"],
        description="Trained models to run on each map (default: the two best classes so far).",
    )
    steps: int = Field(3000, ge=1, description="Adam steps for any model that has to be trained.")
    batch: int = Field(4096, ge=8, description="Rays per step (density models take an eighth).")
    device: str = Field("cpu", description="Device for training density models: cpu, or mps for the Apple GPU.")
    out: Path = Field(Path("trained"), description="Directory of `<world>-<model>.pt` weights, read and written.")
    retrain: bool = Field(False, description="Train every model even when saved weights exist.")
    results: Path = Field(
        SWEEP_RESULTS, description="JSON file the scores are merged into, one entry per (map, model); the page's table."
    )

    def cli_cmd(self) -> None:
        worlds = load_worlds()
        ids = self.worlds or list(worlds)
        self.out.mkdir(parents=True, exist_ok=True)
        table = []
        entries: dict[tuple[str, str], dict] = {}
        for wid in ids:
            w = worlds[wid]
            data = test_views(w)
            row: list[str | float | None] = [f"{wid} ({w.width} × {w.height})"]
            best = ("", -1.0)
            for name in self.models:
                path = self.out / f"{wid}-{name}.pt"
                seconds = None
                if path.exists() and not self.retrain:
                    model = VARIANTS[name].build(w)
                    model.load_state_dict(torch.load(path))
                    source = "saved"
                else:
                    model, history = train(name, w, data, steps=self.steps, batch=self.batch, device=self.device)
                    torch.save(model.state_dict(), path)
                    source, seconds = "trained", round(history[-1]["seconds"], 1)
                s = scores(predict(model, data.test_rays), data.test_distance)
                variant = VARIANTS[name]
                entries[(wid, name)] = {
                    "world": wid,
                    "model": name,
                    "family": variant.family,
                    "familyLabel": CLASSES[variant.family],
                    "label": variant.label,
                    "params": sum(p.numel() for p in model.parameters()),
                    "within10": round(s["within10"], 4),
                    "medianRelative": round(s["median_rel"], 4),
                    "rmse": round(s["rmse"], 4),
                    "trainSeconds": seconds,
                }
                print(f"  {wid:12s} {name:12s} ({source}) within 10 %: {100 * s['within10']:.1f}%", flush=True)
                row.append(round(100 * s["within10"], 1))
                if s["within10"] > best[1]:
                    best = (name, s["within10"])
            table.append([*row, best[0]])
            # Written after every map, so an interrupted sweep keeps what it finished; the file is re-read first, so
            # entries written meanwhile by another sweep are kept.
            self.results.parent.mkdir(parents=True, exist_ok=True)
            saved = json.loads(self.results.read_text()) if self.results.exists() else []
            merged = {(e["world"], e["model"]): e for e in saved} | entries
            ordered = sorted(merged.values(), key=lambda e: (e["world"], list(VARIANTS).index(e["model"])))
            self.results.write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n")
        emit.table(
            ["map", *[f"{m}: rays within 10 % (%)" for m in self.models], "best"],
            table,
            title=f"{TEST_VIEWS} held-out views of 48 rays per map",
        )


class Export(Command):
    """Export saved weights to ONNX for the page, check each against PyTorch, and write the manifest."""

    trained: Path = Field(Path("trained"), description="Directory of `<world>-<model>.pt` weights from `train`.")
    dest: Path = Field(MODELS_DIR, description="Where the .onnx files and manifest.json go.")

    def cli_cmd(self) -> None:
        worlds = load_worlds()
        self.dest.mkdir(parents=True, exist_ok=True)
        order = list(VARIANTS)
        found = []
        for path in self.trained.glob("*.pt"):
            for wid in sorted(worlds, key=len, reverse=True):
                name = path.stem[len(wid) + 1 :]
                if path.stem.startswith(wid + "-") and name in VARIANTS:
                    found.append((wid, name, path))
                    break
        manifest, rows = [], []
        tests = {}
        for wid, name, path in sorted(found, key=lambda f: (f[0], order.index(f[1]))):
            w, variant = worlds[wid], VARIANTS[name]
            if wid not in tests:
                tests[wid] = test_views(w)
            data = tests[wid]
            model = variant.build(w)
            model.load_state_dict(torch.load(path))
            file = self.dest / f"{wid}-{name}.onnx"
            meta = {
                "world": wid,
                "class": variant.family,
                "model": name,
                "contract": "rays[N,3]=(x,y,phi) -> distance[N]",
            }
            diff = to_onnx(model, data.test_rays, file, meta)
            s = scores(run(file, data.test_rays.numpy()), data.test_distance)
            params = sum(p.numel() for p in model.parameters())
            manifest.append(
                {
                    "id": f"{wid}/{name}",
                    "file": file.name,
                    "world": wid,
                    "family": variant.family,
                    "familyLabel": CLASSES[variant.family],
                    "name": name,
                    "label": variant.label,
                    "params": params,
                    "within10": round(s["within10"], 4),
                    "medianRelative": round(s["median_rel"], 4),
                    "rmse": round(s["rmse"], 4),
                }
            )
            rows.append([wid, name, params, round(file.stat().st_size / 1024), diff])
        (self.dest / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        emit.table(["map", "model", "parameters", "size (KB)", "max |ONNX − PyTorch|"], rows, title="Exported")


class Evaluate(Command):
    """Score the exported models on held-out views of their maps, running each with ONNX Runtime."""

    models_dir: Path = Field(MODELS_DIR, description="Directory holding manifest.json and the .onnx files.")

    def cli_cmd(self) -> None:
        worlds = load_worlds()
        manifest = json.loads((self.models_dir / "manifest.json").read_text())
        by_world: dict[str, list[dict]] = {}
        for m in manifest:
            by_world.setdefault(m["world"], []).append(m)
        summary = []
        for wid, entries in by_world.items():
            w = worlds[wid]
            data = test_views(w)
            rows = []
            for m in entries:
                s = scores(run(self.models_dir / m["file"], data.test_rays.numpy()), data.test_distance)
                rows.append([m["familyLabel"], m["label"], m["params"], 100 * s["within10"], 100 * s["median_rel"]])
                summary.append([wid, m["familyLabel"], m["label"], 100 * s["within10"]])
            emit.table(
                ["class", "model", "parameters", "rays within 10 % (%)", "median relative error (%)"],
                rows,
                title=f"{wid} ({w.width} × {w.height}): {TEST_VIEWS} held-out views of 48 rays",
            )
        best = {}
        for wid, family, label, within in summary:
            if within > best.get((wid, family), (-1.0, ""))[0]:
                best[(wid, family)] = (within, label)
        families = list(dict.fromkeys(f for _, f in best))
        emit.table(
            ["map", *families],
            [[wid, *[round(best[(wid, f)][0], 1) if (wid, f) in best else None for f in families]] for wid in by_world],
            title="Best model of each class: held-out rays within 10 % (%)",
        )
        emit.metrics({"maps": len(by_world), "models": len(manifest)}, title="Shipped models")


class NeuralRayCasting(Command):
    """Models of the distance a ray travels to the first wall of a grid world: train, sweep, export, evaluate."""

    train: CliSubCommand[Train] = Field(description=Train.__doc__)
    sweep: CliSubCommand[Sweep] = Field(description=Sweep.__doc__)
    export: CliSubCommand[Export] = Field(description=Export.__doc__)
    evaluate: CliSubCommand[Evaluate] = Field(description=Evaluate.__doc__)

    def cli_cmd(self) -> None:
        CliApp.run_subcommand(self)


if __name__ == "__main__":
    CliApp.run(NeuralRayCasting)
