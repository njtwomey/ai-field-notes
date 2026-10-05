import { useMemo } from 'react'
import { Area, Curve, Figure, float, formatNumber, Handle, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { normalCdf } from 'aifn-compute/numerics/special'

const Z = toFlat(linspace(-4, 4, 401))
const SIGMAS = toFlat(linspace(0.1, 3, 146))
const pdf = (z: number) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI)

/** CRPS of N(0, σ²) at observation y, in closed form. */
function crpsGaussian(sigma: number, y: number): number {
  const z = y / sigma
  return sigma * (z * (2 * normalCdf(z) - 1) + 2 * pdf(z) - 1 / Math.sqrt(Math.PI))
}
const logScore = (sigma: number, y: number) =>
  0.5 * Math.log(2 * Math.PI * sigma * sigma) + (y * y) / (2 * sigma * sigma)

/**
 * Left: the predictive cdf F and the observation's step function; the CRPS is the area of their squared gap. Right: CRPS
 * and log score against the forecast's spread for the current observation.
 */
export function CrpsExplorer() {
  const state = useFigureState({
    sigma: float(1, { min: 0.1, max: 3, step: 0.05, label: 'forecast spread σ' }),
    obs: float(0.8, { min: -3.5, max: 3.5, step: 0.05, label: 'observation y' }),
  })
  const s = state.sigma
  const y = state.obs

  const left = useMemo(() => {
    const cdf = Z.map((z) => normalCdf(z / s))
    return [
      {
        name: 'squared gap (area = CRPS)',
        x: Z,
        y: Z.map((z, i) => (cdf[i] - (z >= y ? 1 : 0)) ** 2),
        slot: 2,
      },
      { name: 'forecast cdf F(z)', x: Z, y: cdf, slot: 0 },
      { name: 'observation step', x: [-4, y, y, 4], y: [0, 0, 1, 1], slot: 1, dashed: true },
    ] as const
  }, [s, y])

  const right = useMemo(
    () =>
      [
        { name: 'CRPS', x: SIGMAS, y: SIGMAS.map((v) => crpsGaussian(v, y)), slot: 0 },
        { name: 'log score', x: SIGMAS, y: SIGMAS.map((v) => logScore(v, y)), slot: 1 },
      ] as const,
    [y],
  )

  const xAxis = useAxis({ label: 'z', range: [-4, 4] })
  const yAxis = useAxis({ label: 'probability', range: [0, 1.05] })
  const xAxis2 = useAxis({ label: 'forecast spread σ', range: [0.1, 3] })
  const yAxis2 = useAxis({ label: 'score (lower is better)', range: [0, 6] })
  return (
    <Figure
      title="The continuous ranked probability score"
      state={state}
      caption="The forecast is a normal distribution centred at 0. Left: its cdf, and the step function that jumps at the observed value; the CRPS is the area under their squared difference. Drag the observation. Right: both scores as the forecast's spread changes, for the current observation. The log score punishes a forecast too narrow for the observation far more sharply; the CRPS changes gently and stays in the units of the data."

      readouts={
        <>
          <Readout label="CRPS" value={formatNumber(crpsGaussian(s, y))} />
          <Readout label="log score" value={formatNumber(logScore(s, y))} />
          <Readout label="absolute error of the mean" value={formatNumber(Math.abs(y))} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={300}>
          <Area {...left[0]} />
          <Curve {...left[1]} />
          <Curve {...left[2]} />
          <Handle
            kind="x"
            at={y}
            label="observation"
            onDrag={(x) => state.set('obs', Math.min(3.5, Math.max(-3.5, x)))}
          />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={300}>
          <Curve {...right[0]} />
          <Curve {...right[1]} />
          <Handle
            kind="x"
            at={s}
            label="forecast spread"
            onDrag={(x) => state.set('sigma', Math.min(3, Math.max(0.1, x)))}
          />
        </Plot>
      </div>
    </Figure>
  )
}
