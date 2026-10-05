import { useMemo } from 'react'
import { Curve, Figure, formatNumber, Handle, int, Plot, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

const ANGLES = toFlat(linspace(0, 180, 181))
const X_RANGE: [number, number] = [0, 180]
const Y_RANGE: [number, number | undefined] = [0, 1]

/** Probability that a pair at angle θ (degrees) shares a bucket in at least one of L tables of k random-hyperplane bits. */
const retrieve = (theta: number, k: number, L: number) => 1 - (1 - (1 - theta / 180) ** k) ** L

/** The S-curve of random-hyperplane LSH: AND over k bits, OR over L tables. */
export function LshSCurve() {
  const state = useFigureState({
    k: int(10, { min: 1, max: 30, step: 1, label: 'bits per key k', format: (v) => String(v) }),
    L: int(20, { min: 1, max: 100, step: 1, label: 'tables L', format: (v) => String(v) }),
    near: slider(0, 180, 30, { step: 1, label: 'near angle (°)', format: (v) => String(v) }),
    far: slider(0, 180, 90, { step: 1, label: 'far angle (°)', format: (v) => String(v) }),
  })

  const series = useMemo(
    () =>
      [
        { name: 'one hash', x: ANGLES, y: ANGLES.map((t) => 1 - t / 180), muted: true, dashed: true },
        {
          name: `k = ${state.k}, L = 1`,
          x: ANGLES,
          y: ANGLES.map((t) => retrieve(t, state.k, 1)),
          slot: 1,
        },
        {
          name: `k = ${state.k}, L = ${state.L}`,
          x: ANGLES,
          y: ANGLES.map((t) => retrieve(t, state.k, state.L)),
          slot: 0,
        },
      ] as const,
    [state.k, state.L],
  )
  const p1 = 1 - state.near / 180
  const p2 = 1 - state.far / 180
  const rho = p1 > p2 && p2 > 0 && p1 < 1 ? Math.log(1 / p1) / Math.log(1 / p2) : NaN
  // Where p^k = 1/L the retrieval probability is 1 - (1 - 1/L)^L, at least 1 - 1/e: the knee of the S-curve.
  const threshold = 180 * (1 - (1 / state.L) ** (1 / state.k))

  const xAxis = useAxis({ label: 'angle θ between query and point (degrees)', range: X_RANGE })
  const yAxis = useAxis({ label: 'probability of retrieval', range: Y_RANGE })
  return (
    <Figure
      title="Locality-sensitive hashing: amplification by AND and OR"
      state={state}
      caption="Random-hyperplane hashing: one bit collides for a pair at angle θ with probability 1 − θ/180°. A key of k bits (AND) collides with probability (1 − θ/180°)^k; L independent tables (OR) retrieve the pair with probability 1 − (1 − (1 − θ/180°)^k)^L. Raising k moves the S-curve left and makes it steeper; raising L moves it right. Drag the near and far lines to set the angles a query must separate."

      readouts={
        <>
          <Readout label="P(retrieve near)" value={formatNumber(retrieve(state.near, state.k, state.L))} />
          <Readout label="P(retrieve far)" value={formatNumber(retrieve(state.far, state.k, state.L))} />
          <Readout label="ρ = ln(1/p₁)/ln(1/p₂)" value={Number.isFinite(rho) ? formatNumber(rho) : '–'} />
          <Readout label="knee where p^k = 1/L (°)" value={formatNumber(threshold)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={340}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
        <Handle {...state.handle('near', { label: 'near' })} />
        <Handle {...state.handle('far', { label: 'far' })} />
      </Plot>
    </Figure>
  )
}
