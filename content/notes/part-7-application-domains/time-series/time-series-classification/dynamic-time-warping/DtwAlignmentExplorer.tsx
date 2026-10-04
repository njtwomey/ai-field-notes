import { useMemo, useState } from 'react'
import { linspace, toFlat, toRows, type Tensor } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import { dynamicProgram } from 'aifn/optim/programming'
import {
  dtw,
  dtwProgram,
  keoghEnvelope,
  lbKeogh,
  lbKim,
} from 'aifn/signal/similarity'
import {
  Figure,
  ControlGroup,
  Select,
  NumberSelector,
  Player,
  Plots,
  Plot,
  Raster,
  Curve,
  Segments,
  Readout,
  formatNumber,
  useAxis,
} from 'aifn-render'

const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')
const range = (a: number, b: number) => Array.from({ length: b - a }, (_, i) => a + i)

const COST_OPTIONS = [
  { value: 'squared', label: 'squared Euclidean (L₂²)' },
  { value: 'absolute', label: 'absolute difference (L₁)' },
]

export function DtwAlignmentExplorer() {
  const [n, setN] = useState(50)
  const [warp, setWarp] = useState(1.6)
  const [windowWidth, setWindowWidth] = useState(8)
  const [cost, setCost] = useState<'squared' | 'absolute'>('squared')
  const [step, setStep] = useState(0)

  const w = Math.min(windowWidth, n)

  // Two versions of one shape: y runs through it at a nonlinearly warped pace, t ↦ t^warp
  const grid = useMemo(() => Array.from(toFlat(linspace(0, 1, n))), [n])
  const x = useMemo(
    () => grid.map((u) => Math.sin(2 * Math.PI * 1.5 * u) + 0.5 * Math.sin(2 * Math.PI * 4 * u)),
    [grid],
  )
  const yv = useMemo(
    () => grid.map((u) => u ** warp).map((u) => Math.sin(2 * Math.PI * 1.5 * u) + 0.5 * Math.sin(2 * Math.PI * 4 * u)),
    [grid, warp],
  )

  const result = useMemo(() => dtw(x, yv, { window: w, cost }), [x, yv, w, cost])
  const run = useMemo(
    () => trace(dynamicProgram(dtwProgram(x, yv, { window: w, cost })), {}, n + 1, { keep: 'all' }),
    [x, yv, w, cost, n],
  )

  const at = Math.min(step, run.steps.length - 1)
  const table = toRows(run.steps[at].table as Tensor) as number[][]
  const z = table.map((r) => r.map((v) => (Number.isFinite(v) ? Math.log10(1 + v) : NaN)))
  const complete = at === run.steps.length - 1

  const top = useMemo(
    () => Math.log10(1 + Math.max(...Array.from(toFlat(result.accumulated)).filter(Number.isFinite))),
    [result],
  )

  const euclid = useMemo(() => dtw(x, yv, { window: 0, cost }).distance, [x, yv, cost])
  const env = useMemo(() => keoghEnvelope(yv, w), [yv, w])

  const idx = range(0, n)
  const bandLo = idx.map((i) => Math.max(0, i - w))
  const bandHi = idx.map((i) => Math.min(n - 1, i + w))

  const ix = useAxis({ label: 'j (index into y)', range: [-0.5, n - 0.5], key: n })
  const iy = useAxis({ label: 'i (index into x)', range: [-0.5, n - 0.5], key: n, equal: ix })
  const tx = useAxis({ label: 'sample index', range: [-0.5, n - 0.5], key: n })
  const vy = useAxis({ label: 'value (y shifted down by 3)', range: [-5, 2] })

  const links = complete ? result.path.filter((_, k) => k % Math.max(1, Math.round(result.path.length / 40)) === 0) : []

  return (
    <Figure
      title="Dynamic time warping with a Sakoe–Chiba band"
      purpose="DTW non-linearly aligns two versions of a time series by finding the minimal-cost monotonic path across the accumulated distance grid. The Sakoe–Chiba band limits how far the warping path can stray from the diagonal, accelerating DP and tightening lower bounds."
      defaultSize="XL"
      controls={
        <ControlGroup>
          <NumberSelector
            label="Series length n"
            value={n}
            onChange={(val) => {
              setN(val)
              setStep(0)
            }}
            min={20}
            max={100}
            step={10}
            suggestions={[30, 50, 80]}
          />
          <NumberSelector
            label="Time warp exponent"
            value={warp}
            onChange={setWarp}
            min={0.5}
            max={3.0}
            step={0.1}
            suggestions={[1.0, 1.6, 2.2]}
          />
          <NumberSelector
            label="Band half-width w"
            value={windowWidth}
            onChange={setWindowWidth}
            min={0}
            max={50}
            step={1}
            suggestions={[0, 4, 8, 16]}
          />
          <Select
            label="Local cost"
            options={COST_OPTIONS}
            value={cost}
            onChange={(v) => setCost(v as 'squared' | 'absolute')}
          />
          <Player
            label="Fill DP table"
            value={at}
            onChange={setStep}
            count={run.steps.length}
          />
        </ControlGroup>
      }
      readouts={
        <>
          <Readout label="DTW distance" value={fmt(result.distance)} />
          <Readout label="Euclidean distance (w=0)" value={fmt(euclid)} />
          <Readout label="LB_Keogh" value={fmt(lbKeogh(x, yv, w, { cost }))} />
          <Readout label="LB_Kim" value={fmt(lbKim(x, yv))} />
          <Readout label="Optimal path length" value={result.path.length} />
        </>
      }
      caption="Left: log₁₀(1 + D) accumulated cost table, constrained within the Sakoe–Chiba band |i − j| ≤ w, and the optimal warping path. Right: original series x (top) and nonlinearly warped series y (offset downwards) joined by alignment segments, alongside y's envelope used by LB_Keogh."
    >
      <Plots cols={2} widths={[1, 1]}>
        <Plot x={ix} y={iy}>
          <Raster x={idx} y={idx} z={z} scale="sequential" range={[0, top]} valueLabel="log₁₀(1 + D)" />
          <Curve name="band edges" x={bandLo} y={idx} emphasis dashed width={1} silent />
          <Curve name="band edges" x={bandHi} y={idx} emphasis dashed width={1} silent />
          {complete && (
            <Curve
              name="warping path"
              x={result.path.map((q) => q[1])}
              y={result.path.map((q) => q[0])}
              slot={0}
              width={2.5}
            />
          )}
        </Plot>
        <Plot x={tx} y={vy}>
          <Curve name="x" x={idx} y={x} slot={1} />
          <Curve name="y" x={idx} y={yv.map((v) => v - 3)} slot={2} />
          <Curve name="envelope of y (upper)" x={idx} y={Array.from(toFlat(env.upper))} muted dashed width={1} />
          <Curve name="envelope of y (lower)" x={idx} y={Array.from(toFlat(env.lower))} muted dashed width={1} />
          <Segments
            name="matches"
            segments={links.map(([i, j]) => ({ from: [i, x[i]] as const, to: [j, yv[j] - 3] as const }))}
          />
        </Plot>
      </Plots>
    </Figure>
  )
}
