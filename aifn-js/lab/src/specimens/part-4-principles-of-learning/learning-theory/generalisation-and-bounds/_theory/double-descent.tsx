/**
 * Double descent with random features (`aifn-applied/theory/double-descent`): test error, training error and weight
 * norm against the number of features p, with the fit along a slice through input space at a chosen p.
 */
import { useMemo } from 'react'
import type { DoubleDescentResult } from 'aifn-applied/theory/double-descent'
import { Player, usePlayhead } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { call, choice, float, int, row, slider, useComputed, useFigureState } from '@lab/state'
import { Curve, formatNumber, Handle, Plot, Plots, Readout, useAxis } from '@lab/viz'

const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : '—')

const EMPTY: DoubleDescentResult = {
  features: [1],
  n: 1,
  testError: new Float64Array(1),
  testErrorMedian: new Float64Array(1),
  trainError: new Float64Array(1),
  weightNorm: new Float64Array(1),
  nullError: 1,
  slice: { t: new Float64Array(0), truth: new Float64Array(0), fits: new Float64Array(0) },
}

export function DoubleDescentSpecimen() {
  const state = useFigureState({
    data: row('1 · problem', {
      n: int(40, { ge: 5, le: 200, suggestions: [20, 40, 80], label: 'training points n' }),
      dimension: int(8, { ge: 2, le: 50, suggestions: [5, 8, 20], label: 'input dimension d' }),
      noise: slider(0, 1, 0.3, { step: 0.05, label: 'noise σ' }),
      seed: int(6, { ge: 0, le: 9999, label: 'seed' }),
    }),
    model: row('2 · model', {
      kind: choice(
        [
          { value: 'relu', label: 'random ReLU features' },
          { value: 'fourier', label: 'random Fourier features' },
        ],
        'relu',
        { label: 'features' },
      ),
      ridge: float(0, { ge: 0, scale: 'log10', suggestions: [0, 1e-4, 1e-3, 1e-2, 1e-1], label: 'ridge λ' }),
      repeats: int(10, { ge: 1, le: 100, suggestions: [5, 10, 30], label: 'repeats per p' }),
    }),
  })
  const { n, dimension, noise, seed } = state.data
  const { kind, ridge, repeats } = state.model
  const run = useComputed(
    () =>
      call<DoubleDescentResult>(
        'applied/theory/double-descent/doubleDescent',
        call('foundation/random/stream', `double-descent/${seed}`),
        { n, dimension, noise, kind, ridge, repeats },
      ),
    [n, dimension, noise, kind, ridge, repeats, seed],
    { mode: 'worker', initial: EMPTY },
  )
  const r = run.value
  const [k, setK] = usePlayhead(r.features.length)
  const at = Math.min(k, r.features.length - 1)
  const p = r.features[at]
  const pAxis = useAxis({
    label: 'features p',
    range: [1, Math.max(2, r.features[r.features.length - 1])],
    log: true,
    key: r.features.length,
  })
  const errAxis = useAxis({ label: 'mean squared error', log: true, range: [1e-3, 100] })
  const normAxis = useAxis({
    label: '‖w‖₂',
    log: true,
    hold: 'union',
    key: `${n}/${dimension}/${noise}/${kind}/${ridge}`,
  })
  const tAxis = useAxis({ label: 'position t along a direction in input space', range: [-3, 3] })
  const yAxis = useAxis({ label: 'f(x), fit', range: [-3, 3] })
  const ps = r.features
  const clip = (a: Float64Array) => Array.from(a, (v) => Math.min(100, Math.max(1e-3, v)))
  const fit = useMemo(() => {
    const m = r.slice.t.length
    return Array.from(r.slice.fits.subarray(at * m, (at + 1) * m), (v) => Math.max(-3, Math.min(3, v)))
  }, [r, at])
  const nearest = (v: number) => {
    let best = 0
    ps.forEach((q, i) => {
      if (Math.abs(Math.log(q) - Math.log(v)) < Math.abs(Math.log(ps[best]) - Math.log(v))) best = i
    })
    setK(best)
  }
  const marker = <Handle kind="x" at={p} onDrag={nearest} label={`p = ${p}`} />
  return (
    <Figure
      title="Double descent"
      purpose="With p random features fitted by least squares to n noisy points, the test error falls, rises to a peak at the interpolation threshold p = n, where the single interpolating solution has a huge norm, and falls again as the minimum-norm interpolant gets smoother; a ridge penalty removes the peak."
      state={state}
      defaultSize="L"
      controls={
        <ControlRow label="3 · features">
          <Player
            className="col-span-full"
            value={at}
            onChange={setK}
            count={ps.length}
            label="p"
            format={(i) => `p = ${ps[i] ?? 1}`}
          />
        </ControlRow>
      }
      readouts={{
        [`p = ${p}`]: (
          <>
            <Readout label="p / n" value={fmt(p / n, 2)} />
            <Readout label="test MSE (median)" value={fmt(r.testErrorMedian[at])} />
            <Readout label="train MSE" value={fmt(r.trainError[at])} />
            <Readout label="‖w‖₂" value={fmt(r.weightNorm[at])} />
            <Readout label="predict 0" value={fmt(r.nullError)} />
          </>
        ),
      }}
      caption="aifn doubleDescent: x ~ N(0, I_d), y = (β·x + sin 2γ·x)/√2 + σε; features max(0, a·x/√d) or cos(a·x/√d + b) with random a, b; output weights by lstsq (minimum norm once p ≥ n) or ridge. Left: median and mean test error over the repeats, and the training error, which reaches 0 at p = n (dashed ink line). Middle: the mean weight norm. Right: the first repeat's fit along a line through input space at the chosen p, against f. Play p, or drag the p marker on the left or middle chart; errors are clipped to [10⁻³, 100]."
    >
      {/* Marks the figure busy while the worker computes, so screenshots wait for the answer. */}
      <span aria-busy={run.stale} className="hidden" />
      <Plots cols={3}>
        <Plot x={pAxis} y={errAxis} title="error against features">
          <Curve name="p = n" x={[n, n]} y={[1e-3, 100]} emphasis dashed thin />
          <Curve name="test (median)" slot={0} x={ps} y={clip(r.testErrorMedian)} showPoints stale={run.stale} />
          <Curve name="test (mean, dashed)" slot={0} x={ps} y={clip(r.testError)} thin dashed stale={run.stale} />
          <Curve name="train" slot={1} x={ps} y={clip(r.trainError)} stale={run.stale} />
          {marker}
        </Plot>
        <Plot x={pAxis} y={normAxis} title="weight norm">
          <Curve name="p = n" x={[n, n]} y={[1e-3, 1e6]} emphasis dashed thin />
          <Curve name="‖w‖₂" slot={2} x={ps} y={Array.from(r.weightNorm, (v) => Math.max(1e-3, v))} stale={run.stale} />
          {marker}
        </Plot>
        <Plot x={tAxis} y={yAxis} title={`fit along a slice, p = ${p}`}>
          <Curve name="f" x={r.slice.t} y={r.slice.truth} emphasis dashed />
          <Curve name="fit" slot={0} x={r.slice.t} y={fit} stale={run.stale} />
        </Plot>
      </Plots>
    </Figure>
  )
}
