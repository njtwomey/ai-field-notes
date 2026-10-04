# aifn lab

A standalone app for exploring aifn: every module's specimens, each a page of figures. It depends on `aifn`, `aifn-methods`,
`aifn-js/modules.json` and third-party packages only (`make lab-check` enforces the boundary).

| Command          | What it does                                                                            |
| ---------------- | --------------------------------------------------------------------------------------- |
| `make lab`       | Dev server on http://localhost:5190/ (pages at `/<module>/<specimen>`, figures `#<id>`) |
| `make lab-check` | Renders every page on the server and reports any that throw                             |
| `make lab-shots` | Screenshots pages and figures in headless Chrome into `.scratch/lab-shots/`             |

## Screenshots: `make lab-shots`

`aifn-js/sandbox/lab/screenshot.ts` starts its own Vite server (port 5191, or the next free one; `--url` reuses a running lab),
drives one headless Chrome over the DevTools protocol (no extra dependency), opens each page in turn and always shuts
both down. It waits for fonts and for every chart to report `data-chart-ready` (set by `viz/EChart.tsx` when ECharts
finishes a frame with no animation left), then saves:

```
.scratch/lab-shots/<module>/<slug>/<theme>[-mobile]/page.png         full page
                                                  /<figure-id>.png    each Figure, clipped to its box
                                                  /console.txt        console errors and warnings
.scratch/lab-shots/index.md                                           every file of the run, with its URL
```

A page fails (exit code 1) on a console error, an uncaught exception, a figure that failed to render, a KaTeX error or
a zero-size chart.

```bash
make lab-shots ARGS="--only autodiff"                                   # a module
make lab-shots ARGS="--only metrics/reliability-diagram-and-calibration-errors --theme dark"
make lab-shots ARGS="--only 'ui-kit#heatmap-with-overlays-contours-and-a-draggable-start'"   # one figure
make lab-shots ARGS="--mobile --width 1280 --height 800 --dpr 1"        # adds a 390 × 844 pass
make lab-shots ARGS="--only ui-kit --act 'heatmap-with-overlays-contours-and-a-draggable-start:drag 0.75 0.3 0.3 0.7'"
make lab-shots ARGS="--only ui-kit --profile --act '<figure-id>:drag 0.75 0.3 0.3 0.7'"   # measure the drag
```

Every run also clicks each enabled slider at 20% and 80% of its track with real mouse events and fails the page if a
value does not move (`--no-sliders` skips this).

`--act '<figure-id>:drag x0 y0 x1 y1'` and `'<figure-id>:click x y'` act on a figure before the capture; coordinates
are fractions of the figure's chart area. `--profile` turns each drag into 60 pointer moves at display rate and
writes `profile.json` (input-to-paint, frame times, dropped frames, long tasks, script time, ECharts `setOption` calls
counted by the dev-only `window.__labStats`, and the heaviest functions by CPU self time) and a one-line
`profile.txt`. With no `--only`, every page is shot (about 1 s per page per theme).
