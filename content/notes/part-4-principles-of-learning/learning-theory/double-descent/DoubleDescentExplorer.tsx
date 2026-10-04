import { useMemo, useState } from 'react'
import { doubleDescent, type DoubleDescentResult } from 'aifn-methods/theory/double-descent'
import { stream } from 'aifn/foundation/random'
import {
  Figure,
  ControlGroup,
  Select,
  NumberSelector,
  Player,
  Plots,
  Plot,
  Curve,
  Points,
  Handle,
  Readout,
  formatNumber,
  useAxis,
} from 'aifn-render'

const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')

const FEATURE_KINDS = [
  { value: 'relu', label: 'Random ReLU features' },
  { value: 'fourier', label: 'Random Fourier features' },
]

export function DoubleDescentExplorer() {
  const [n, setN] = useState(40)
  const [kind, setKind] = useState<'relu' | 'fourier'>('relu')
  const [ridge, setRidge] = useState(0)
  const [featureIdx, setFeatureIdx] = useState(12)

  const seed = 6
  const dimension = 8
  const noise = 0.3
  const repeats = 6

  const res: DoubleDescentResult = useMemo(
    () =>
      doubleDescent(stream(`content/double-descent/${seed}`), {
        n,
        dimension,
        noise,
        kind,
        ridge,
        repeats,
        test: 600,
        slice: 101,
      }),
    [n, kind, ridge],
  )

  const at = Math.min(featureIdx, res.features.length - 1)
  const currentP = res.features[at]

  const sliceX = Array.from(res.slice.t)
  const truthY = Array.from(res.slice.truth)
  const fitY = Array.from(res.slice.fits.subarray(at * 101, (at + 1) * 101))

  const pAxis = useAxis({ label: 'parameters p (number of features)', hold: 'initial' })
  const errAxis = useAxis({ label: 'mean squared error', range: [0, Math.min(2.5, Math.max(...res.testError) * 1.1)] })

  const sliceAxisX = useAxis({ label: 'slice direction t' })
  const sliceAxisY = useAxis({ label: 'f(x)', range: [-3, 3] })

  return (
    <Figure
      title="Double descent in random features regression"
      purpose="As the number of random features p crosses the sample size n, test error spikes at the interpolation threshold (p = n) where the model is forced to fit training noise. Past the threshold, minimum-norm interpolation becomes smoother, and test error descends again."
      defaultSize="XL"
      controls={
        <ControlGroup>
          <NumberSelector
            label="Sample size n"
            value={n}
            onChange={(val) => {
              setN(val)
              setFeatureIdx(10)
            }}
            min={20}
            max={80}
            step={10}
            suggestions={[30, 40, 60]}
          />
          <Select
            label="Feature type"
            options={FEATURE_KINDS}
            value={kind}
            onChange={(v) => setKind(v as 'relu' | 'fourier')}
          />
          <NumberSelector
            label="Ridge penalty λ"
            value={ridge}
            onChange={setRidge}
            min={0}
            max={0.1}
            step={0.001}
            suggestions={[0, 0.001, 0.01, 0.05]}
          />
          <NumberSelector
            label="Current p"
            value={currentP}
            onChange={(val) => {
              const closest = res.features.reduce(
                (prev, curr, idx) => (Math.abs(curr - val) < Math.abs(res.features[prev] - val) ? idx : prev),
                0,
              )
              setFeatureIdx(closest)
            }}
            min={res.features[0]}
            max={res.features[res.features.length - 1]}
            step={1}
            suggestions={[n / 2, n, n * 2, n * 4]}
          />
          <Player label="Sweep features p" value={at} onChange={setFeatureIdx} count={res.features.length} />
        </ControlGroup>
      }
      readouts={
        <>
          <Readout label="Parameters p" value={currentP} />
          <Readout
            label="Regime"
            value={
              currentP < n ? 'Under-parameterised' : currentP === n ? 'Interpolation threshold' : 'Over-parameterised'
            }
          />
          <Readout label="Test MSE" value={fmt(res.testError[at])} />
          <Readout label="Train MSE" value={fmt(res.trainError[at])} />
          <Readout label="Weight norm ‖w‖" value={fmt(res.weightNorm[at])} />
        </>
      }
      caption="Left: test error (solid) and training error (dashed) across model capacity p, with a vertical guide at p = n. Right: cross-section of the true function (black) and the fitted function (colored) along a 1-D slice. Near p = n, weights explode to interpolate noise; for p ≫ n, the minimum-norm solution stabilizes into a smooth interpolant."
    >
      <Plots cols={2} widths={[1.1, 1]}>
        <Plot x={pAxis} y={errAxis}>
          <Curve name="test error" x={res.features} y={Array.from(res.testError)} slot={0} />
          <Curve name="train error" x={res.features} y={Array.from(res.trainError)} slot={1} dashed />
          <Curve name="interpolation threshold (p=n)" x={[n, n]} y={[0, 3]} emphasis dashed thin />
          <Points name="selected p" x={[currentP]} y={[res.testError[at]]} slot={0} size={8} />
          <Handle
            kind="x"
            at={currentP}
            label="p"
            onDrag={(val) => {
              const closest = res.features.reduce(
                (prev, curr, idx) => (Math.abs(curr - val) < Math.abs(res.features[prev] - val) ? idx : prev),
                0,
              )
              setFeatureIdx(closest)
            }}
          />
        </Plot>
        <Plot x={sliceAxisX} y={sliceAxisY}>
          <Curve name="true function" x={sliceX} y={truthY} muted width={1.5} />
          <Curve name="fitted slice" x={sliceX} y={fitY} slot={0} width={2} />
        </Plot>
      </Plots>
    </Figure>
  )
}
