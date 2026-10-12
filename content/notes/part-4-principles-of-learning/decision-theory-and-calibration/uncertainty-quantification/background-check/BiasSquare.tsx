import { useMemo } from 'react'
import { Figure, float, formatNumber, Handle, Plot, Raster, Readout, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

const X = toFlat(linspace(-14, 14, 561))
const DX = X[1] - X[0]
const MU = toFlat(linspace(0, 1, 41))
const HALF_LOG_2PI = 0.5 * Math.log(2 * Math.PI)
const normal = (x: number, m: number, s: number) => Math.exp(-0.5 * ((x - m) / s) ** 2 - Math.log(s) - HALF_LOG_2PI)
const clamp01 = (v: number) => Math.min(1, Math.max(0, v))

type Cell = { qf: number; m: number; fg: number; out: number; err: number }

/** Shares of foreground and outliers rejected by the bias (μ0, μ1), and the error rate on accepted foreground. */
function rejectionRates(cells: readonly Cell[], mu0: number, mu1: number) {
  let fg = 0
  let out = 0
  let err = 0
  let acc = 0
  for (const c of cells) {
    if (c.m * c.qf <= mu0 + (mu1 - mu0) * c.qf) {
      fg += c.fg
      out += c.out
    } else {
      acc += c.fg
      err += c.err
    }
  }
  return { fg, out, err: acc > 0 ? err / acc : NaN }
}

/**
 * The affine bias's parameter square for a known problem on a line: two unit-variance Gaussian classes at ±sep/2 with
 * equal priors, and outliers from a wide Gaussian centred between them. Each cell is one bias (μ0, μ1); a point is
 * rejected when m(x) q_f(x) ≤ μ0 + (μ1 − μ0) q_f(x), with m the larger within-foreground class probability. The maps
 * show the share of foreground and of outliers rejected.
 */
export function BiasSquare() {
  const state = useFigureState({
    mu0: float(0.2, { min: 0, max: 1, step: 0.01, label: 'μ(0)', onChart: true }),
    mu1: float(0.2, { min: 0, max: 1, step: 0.01, label: 'μ(1)', onChart: true }),
    sep: float(3, { min: 0, max: 6, step: 0.1, label: 'class separation (sd)' }),
    spread: float(4, { min: 1.5, max: 8, step: 0.1, label: 'outlier standard deviation' }),
  })

  // Per grid point: q_f, m, and the masses of foreground, outliers and foreground errors (if accepted) there.
  const cells = useMemo(() => {
    const pts = Array.from(X, (x) => {
      const a = 0.5 * normal(x, -state.sep / 2, 1)
      const c = 0.5 * normal(x, state.sep / 2, 1)
      return { pf: a + c, m: Math.max(a, c) / (a + c), err: Math.min(a, c), out: normal(x, 0, state.spread) }
    })
    const peak = Math.max(...pts.map((p) => p.pf))
    return pts.map((p) => ({ qf: p.pf / peak, m: p.m, fg: p.pf * DX, out: p.out * DX, err: p.err * DX }))
  }, [state.sep, state.spread])

  const rates = (mu0: number, mu1: number) => rejectionRates(cells, mu0, mu1)

  const maps = useMemo(() => {
    const fg: number[][] = []
    const out: number[][] = []
    for (const m1 of MU) {
      const fr: number[] = []
      const or: number[] = []
      for (const m0 of MU) {
        const r = rejectionRates(cells, m0, m1)
        fr.push(r.fg)
        or.push(r.out)
      }
      fg.push(fr)
      out.push(or)
    }
    return { fg, out }
  }, [cells])

  const at = rates(state.mu0, state.mu1)
  const noReject = rates(0, 0)
  const drag = ([x, y]: [number, number]) => {
    state.set('mu0', clamp01(x))
    state.set('mu1', clamp01(y))
  }

  const xa = useAxis({ label: 'μ(0)', range: [0, 1] })
  const ya = useAxis({ label: 'μ(1)', range: [0, 1] })
  const xb = useAxis({ label: 'μ(0)', range: [0, 1] })
  const yb = useAxis({ label: 'μ(1)', range: [0, 1] })
  return (
    <Figure
      title="What each background bias rejects"
      state={state}
      caption="Every point of the square is an affine background bias (μ(0), μ(1)) for two Gaussian classes on a line, with outliers from a wider Gaussian centred between them. Left: the share of foreground cases each bias rejects. Right: the share of outliers it rejects. Drag the marker on either map. The left edge, μ(0) = 0, is Chow's rule: it rejects foreground and outliers alike by ambiguity alone. The bottom edge, μ(1) = 0, rejects by density. The diagonal is the constant bias."
      readouts={
        <>
          <Readout label="foreground rejected" value={formatNumber(at.fg)} />
          <Readout label="outliers rejected" value={formatNumber(at.out)} />
          <Readout label="error on accepted foreground" value={formatNumber(at.err)} />
          <Readout label="error without rejection" value={formatNumber(noReject.err)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={xa} y={ya} height={320}>
          <Raster x={MU} y={MU} z={maps.fg} valueLabel="foreground rejected" />
          <Handle kind="point" at={[state.mu0, state.mu1]} label="bias" onDrag={drag} />
        </Plot>
        <Plot x={xb} y={yb} height={320}>
          <Raster x={MU} y={MU} z={maps.out} valueLabel="outliers rejected" />
          <Handle kind="point" at={[state.mu0, state.mu1]} label="bias" onDrag={drag} />
        </Plot>
      </div>
    </Figure>
  )
}
