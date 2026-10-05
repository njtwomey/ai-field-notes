/**
 * Time-series similarity: the matrix profile of a series with planted motifs and a discord, computed anytime by
 * SCRIMP++ and played diagonal batch by batch, with a draggable subsequence's MASS distance profile; and dynamic time
 * warping with a Sakoe–Chiba band, its cost table filled row by row on the shared DP engine, the warping path and the
 * lower bounds. Every number is `aifn/signal/similarity`'s; the series is aifn-methods `motifSeries`.
 */
import { useMemo, useState } from 'react'
import { stream } from 'aifn-compute/foundation/random'
import { linspace, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { dynamicProgram } from 'aifn-compute/optim/programming'
import {
  discords,
  distanceProfile,
  dtw,
  dtwProgram,
  keoghEnvelope,
  lbKeogh,
  lbKim,
  matrixProfile,
  motifs,
  scrimpProfile,
  scrimpSteps,
} from 'aifn-compute/signal/similarity'
import { motifSeries } from 'aifn-methods/data/synthetic'
import { Player } from 'aifn-render/controls'
import { ControlRow, Figure } from 'aifn-render/layout'
import { choice, float, int, row, useFigureState } from 'aifn-render/state'
import { formatValue } from '@lab/views'
import { Annotation, Curve, Handle, Plot, Plots, Points, Raster, Readout, Segments, useAxis } from 'aifn-render/viz'

const f3 = (v: number) => (Number.isFinite(v) ? formatValue(Number(v.toPrecision(3))) : '—')
const range = (a: number, b: number) => Array.from({ length: b - a }, (_, i) => a + i)

// ── Matrix profile ───────────────────────────────────────────────────────────────────────────────────────────────────

export function MatrixProfileFigure() {
  const state = useFigureState({
    data: row('1 · series', {
      n: int(600, { ge: 100, le: 4000, suggestions: [300, 600, 2000], label: 'length' }),
      m: int(40, { ge: 4, le: 300, suggestions: [20, 40, 80], label: 'subsequence length m' }),
      seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
    }),
    run: row('2 · SCRIMP++', {
      perStep: int(12, { ge: 1, le: 1000, suggestions: [4, 12, 50], label: 'diagonals per step' }),
      prescrimp: choice(
        [
          { value: 'yes', label: 'PreSCRIMP first' },
          { value: 'no', label: 'diagonals only' },
        ],
        'yes',
        { label: 'start' },
      ),
    }),
  })
  const { n, seed } = state.data
  const m = Math.min(state.data.m, Math.floor(n / 4))
  const series = useMemo(() => motifSeries(stream(`lab/matrix-profile/${seed}`), { n, m }), [n, m, seed])
  const y = useMemo(() => Array.from(toFlat(series.y)), [series])
  const t = useMemo(() => range(0, n), [n])
  const run = useMemo(
    () =>
      trace(
        scrimpSteps(series.y, m, { diagonalsPerStep: state.run.perStep, prescrimp: state.run.prescrimp === 'yes' }),
        undefined,
        100000,
        { keep: 'all', stream: stream('lab/scrimp') },
      ),
    [series, m, state.run.perStep, state.run.prescrimp],
  )
  const exact = useMemo(() => matrixProfile(series.y, m), [series, m])
  const [step, setStep] = useState(0)
  const at = Math.min(step, run.steps.length - 1)
  const s = run.steps[at]
  const mp = scrimpProfile(s, m)
  const profile = Array.from(toFlat(mp.profile)).map((v) => (Number.isFinite(v) ? v : NaN))
  const exactProfile = Array.from(toFlat(exact.profile))
  const top = motifs(mp, { count: 1 })[0]
  const odd = discords(mp, { count: 1 })[0]
  const [probe, setProbe] = useState(Math.round(n / 7))
  const p = Math.max(0, Math.min(n - m, Math.round(probe)))
  const dp = useMemo(() => Array.from(toFlat(distanceProfile(y.slice(p, p + m), series.y))), [y, p, m, series])
  const best = dp.reduce((b, v, i) => (Math.abs(i - p) > Math.ceil(m / 4) && v < dp[b] ? i : b), p === 0 ? m : 0)
  const windowAt = (start: number) => ({ x: t.slice(start, start + m), y: y.slice(start, start + m) })
  const tx = useAxis({ label: 'time', range: [0, n], key: n })
  const vy = useAxis({ label: 'value', hold: 'union', key: `${n}-${seed}` })
  const py = useAxis({ label: 'profile (NN dist.)', range: [0, Math.sqrt(4 * m)], key: m })
  const dy = useAxis({ label: `distance to [${p}, ${p + m})`, range: [0, Math.sqrt(4 * m)], key: m })
  const k = t.slice(0, n - m + 1)
  return (
    <Figure
      title="The matrix profile: motifs and discords"
      purpose="Every subsequence's distance to its nearest match: the lowest values are a repeated shape (a motif), the highest a shape seen nowhere else (a discord); SCRIMP++ has both long before it has finished."
      state={state}
      defaultSize="XL"
      controls={
        <ControlRow label="3 · steps">
          <Player
            value={at}
            onChange={setStep}
            count={run.steps.length}
            label="step"
            format={(i) => `${run.steps[i]?.done ?? 0} / ${s.diagonals} diagonals`}
          />
        </ControlRow>
      }
      readouts={
        <>
          <Readout label="diagonals done" value={`${s.done} of ${s.diagonals}`} />
          <Readout label="top motif" value={top ? `${top.a} & ${top.b} (d = ${f3(top.distance)})` : '—'} />
          <Readout label="top discord" value={odd ? `${odd.at} (d = ${f3(odd.distance)})` : '—'} />
          <Readout label="planted" value={`motif at ${series.motifs.join(' & ')}, discord at ${series.discord}`} />
          <Readout label="dragged window" value={`[${p}, ${p + m}), best match ${best} (d = ${f3(dp[best])})`} />
        </>
      }
      caption={
        <>
          aifn-methods <code>motifSeries</code> (a random walk with one shape planted twice and a fast burst once), aifn{' '}
          <code>scrimpSteps</code>, <code>motifs</code>, <code>discords</code> and <code>distanceProfile</code> (MASS),
          z-normalised, m = {m}, exclusion zone ⌈m/4⌉. Top: the series with the current top motif pair (slots 1 and 2)
          and top discord (red). Middle: the profile so far against the exact STOMP profile (dashed); step 0 is
          PreSCRIMP&apos;s approximation, then each step evaluates {state.run.perStep} more diagonals in random order
          and the profile can only fall. Bottom: the distance profile of the dragged window (drag its marker on the top
          chart); its minimum outside the exclusion zone is the window&apos;s matrix-profile value.
        </>
      }
    >
      <Plots rows={3} cols={1} heights={[3, 2, 2]}>
        <Plot x={tx} y={vy}>
          <Curve name="series" x={t} y={y} muted width={1} />
          {top && <Curve name="motif" x={windowAt(top.a).x} y={windowAt(top.a).y} slot={0} width={2.5} />}
          {top && <Curve name="its match" x={windowAt(top.b).x} y={windowAt(top.b).y} slot={1} width={2.5} />}
          {odd && <Curve name="discord" x={windowAt(odd.at).x} y={windowAt(odd.at).y} tone="destructive" width={2.5} />}
          <Curve name="dragged window" x={windowAt(p).x} y={windowAt(p).y} emphasis width={1.5} dashed />
          <Handle kind="x" at={p} onDrag={setProbe} label="window" />
        </Plot>
        <Plot x={tx} y={py}>
          <Curve name="exact (STOMP)" x={k} y={exactProfile} emphasis dashed width={1} />
          <Curve name="SCRIMP++ so far" x={k} y={profile} slot={2} />
          {top && <Points name="motif" x={[top.a, top.b]} y={[top.distance, top.distance]} slot={0} size={8} />}
          {odd && <Points name="discord" x={[odd.at]} y={[odd.distance]} tone="destructive" size={8} />}
        </Plot>
        <Plot x={tx} y={dy}>
          <Curve name="distance profile" x={k} y={dp} slot={3} />
          <Points name="best match" x={[best]} y={[dp[best]]} emphasis size={8} />
          <Annotation x={p} text="window" dashed />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── DTW ──────────────────────────────────────────────────────────────────────────────────────────────────────────────

export function DtwFigure() {
  const state = useFigureState({
    data: row('1 · two series', {
      n: int(50, { ge: 5, le: 200, suggestions: [30, 50, 100], label: 'length' }),
      warp: float(1.6, { ge: 0.3, le: 3, suggestions: [1, 1.6, 2.5], label: 'time warp (exponent)' }),
    }),
    dtw: row('2 · DTW', {
      window: int(8, { ge: 0, le: 200, suggestions: [0, 4, 8, 50], label: 'band half-width w' }),
      cost: choice(['squared', 'absolute'] as const, 'squared', { label: 'local cost' }),
    }),
  })
  const { n, warp } = state.data
  const w = Math.min(state.dtw.window, n)
  const cost = state.dtw.cost
  // Two versions of one shape: y runs through it at a warped pace, t ↦ t^warp.
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
  const [step, setStep] = useState(0)
  const at = Math.min(step, run.steps.length - 1)
  const table = toRows(run.steps[at].table as Tensor) as number[][]
  const z = table.map((r) => r.map((v) => (Number.isFinite(v) ? Math.log10(1 + v) : NaN)))
  const complete = at === run.steps.length - 1
  // Hold the colour scale at the full table's range while it fills.
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
  const tx = useAxis({ label: 'index', range: [-0.5, n - 0.5], key: n })
  const vy = useAxis({ label: 'value (y shifted down by 3)', range: [-5, 2] })
  const links = complete ? result.path.filter((_, k) => k % Math.max(1, Math.round(result.path.length / 40)) === 0) : []
  return (
    <Figure
      title="Dynamic time warping with a Sakoe–Chiba band"
      purpose="DTW aligns two versions of one shape run at different paces by the cheapest monotone path through their cost table; the band limits how far the path may stray from the diagonal, and the lower bounds sit below it without filling the table."
      state={state}
      defaultSize="XL"
      controls={
        <ControlRow label="3 · fill the table">
          <Player value={at} onChange={setStep} count={run.steps.length} label="rows filled" />
        </ControlRow>
      }
      readouts={
        <>
          <Readout label="DTW distance" value={f3(result.distance)} />
          <Readout label="Euclidean (w = 0)" value={f3(euclid)} />
          <Readout label="LB_Keogh" value={f3(lbKeogh(x, yv, w, { cost }))} />
          <Readout label="LB_Kim" value={f3(lbKim(x, yv))} />
          <Readout label="path length" value={result.path.length} />
        </>
      }
      caption={
        <>
          aifn <code>dtwProgram</code> stepped by <code>dynamicProgram</code> (row i of the accumulated cost D(i, j) =
          c(xᵢ, yⱼ) + min of its three predecessors per step), <code>dtw</code>, <code>keoghEnvelope</code>,{' '}
          <code>lbKeogh</code> and <code>lbKim</code>. Left: log₁₀(1 + D), empty outside the band |i − j| ≤ {w} (its
          edges dashed), and the optimal warping path once the table is full. Right: x (top) and y (shifted down) with
          matched pairs joined, and y&apos;s band envelope that LB_Keogh measures x against. A wider band never costs
          more; w = 0 is the Euclidean distance. Play the table filling from row 0.
        </>
      }
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
          <Curve name="envelope of y" x={idx} y={Array.from(toFlat(env.upper))} muted dashed width={1} />
          <Curve name="envelope of y" x={idx} y={Array.from(toFlat(env.lower))} muted dashed width={1} />
          <Segments
            name="matches"
            segments={links.map(([i, j]) => ({ from: [i, x[i]] as const, to: [j, yv[j] - 3] as const }))}
          />
        </Plot>
      </Plots>
    </Figure>
  )
}
