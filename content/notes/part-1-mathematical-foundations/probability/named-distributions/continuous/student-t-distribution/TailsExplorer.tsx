import { useMemo } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Readout,
  setting,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { Normal, StudentT } from 'aifn-compute/probability/distributions'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

const GAUSSIAN = Normal(0, 1)

/** Student t with ν degrees of freedom against the standard normal, with an optional log scale to show the tails. */
export function TailsExplorer() {
  const state = useFigureState({
    nu: slider(0.5, 50, 2, { step: 0.5, label: 'degrees of freedom ν' }),
    cut: slider(1, 6, 3, { step: 0.5, label: 'threshold c' }),
    log: setting(true, 'log density'),
  })

  const t = useMemo(() => StudentT(state.nu), [state.nu])
  const series = useMemo(() => {
    const xs = toFlat(linspace(-8, 8, 401))
    return [
      { name: `t, ν = ${state.nu}`, x: xs, y: xs.map((x) => t.prob(x)), slot: 0 },
      { name: 'standard normal', x: xs, y: xs.map((x) => GAUSSIAN.prob(x)), slot: 1 },
    ] as const
  }, [state.nu, t])

  const tailT = 2 * (1 - t.cdf(state.cut))
  const tailZ = 2 * (1 - GAUSSIAN.cdf(state.cut))

  const xAxis = useAxis({ label: 'x', hold: 'union' })
  const yAxis = useAxis({
    label: state.log ? 'density (log scale)' : 'density',
    log: state.log,
    range: state.log ? [1e-16, 1] : [0, undefined],
  })
  return (
    <Figure
      title="Heavy tails"
      state={state}
      caption="On a log scale the Gaussian density is a downward parabola and the t density falls far more slowly, because its tails decay as a power of x. Small ν gives far more mass beyond a few units. As ν grows the two curves merge. Drag the vertical line to move the threshold c."
      readouts={
        <>
          <Readout label="P(|T| > c)" value={formatNumber(tailT)} />
          <Readout label="P(|Z| > c)" value={formatNumber(tailZ)} />
          <Readout label="ratio" value={formatNumber(tailT / tailZ)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Handle {...state.handle('cut', { label: 'threshold c' })} />
      </Plot>
    </Figure>
  )
}
