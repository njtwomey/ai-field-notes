import { useMemo } from 'react'
import {
  Bars,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  MathText,
  Plot,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { useClassColors } from '../_shared/classColor'

type Distance = 'absolute' | 'squared'
const DISTANCES = [
  { value: 'absolute' as const, label: '|k − y|' },
  { value: 'squared' as const, label: '(k − y)²' },
]

/** SORD target: softmax of −α φ(k, y) over the classes. */
function sord(k: number, y: number, alpha: number, distance: Distance): number[] {
  const logits = Array.from(
    { length: k },
    (_, c) => -alpha * (distance === 'absolute' ? Math.abs(c - y) : (c - y) ** 2),
  )
  const top = Math.max(...logits)
  const e = logits.map((l) => Math.exp(l - top))
  const z = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / z)
}

/** Cross-entropy of a target against a confident prediction: 0.9 on class c, the rest spread evenly. */
function crossEntropy(target: number[], c: number): number {
  const k = target.length
  return -target.reduce((s, t, j) => s + t * Math.log(j === c ? 0.9 : 0.1 / (k - 1)), 0)
}

/**
 * Soft ordinal labels (SORD): the target puts mass on neighbouring classes in proportion to exp(−α φ). The right panel
 * shows why this matters: against a one-hot target, cross-entropy charges every wrong class the same.
 */
export function SoftLabels() {
  const state = useFigureState({
    k: int(7, { min: 3, max: 10, step: 1, label: 'classes K' }),
    alpha: float(1, { min: 0.1, max: 4, step: 0.05, label: 'sharpness α' }),
    distance: choice<Distance>(DISTANCES, 'absolute', { label: 'distance φ' }),
    y: int(3, { min: 1, max: 10, label: 'true class y (at most K)' }),
  })
  const truth = Math.min(state.y, state.k) - 1
  const colors = useClassColors(state.k)

  const { target, left, right, meanDistance } = useMemo(() => {
    const target = sord(state.k, truth, state.alpha, state.distance)
    const oneHot = target.map((_, j) => (j === truth ? 1 : 0))
    const smooth = target.map((_, j) => (j === truth ? 0.9 : 0) + 0.1 / state.k)
    const classes = target.map((_, j) => j + 1)
    const left = { classes, target, smooth }
    const right = [
      { name: 'one-hot target', x: classes, y: classes.map((_, c) => crossEntropy(oneHot, c)), slot: 2 },
      { name: 'SORD target', x: classes, y: classes.map((_, c) => crossEntropy(target, c)), slot: 0 },
    ] as const
    const meanDistance = target.reduce((s, t, j) => s + t * Math.abs(j - truth), 0)
    return { target, left, right, meanDistance }
  }, [state.k, truth, state.alpha, state.distance])

  const xAxis = useAxis({ label: 'predicted class c', range: [1, state.k] })
  const yAxis = useAxis({ label: 'cross-entropy', range: [0, undefined], hold: 'union' })
  const kAxis = useAxis({ label: 'class k', range: [0.5, state.k + 0.5], integer: true })
  const tAxis = useAxis({ label: 'target probability', range: [0, 1] })
  return (
    <Figure
      title="Soft ordinal labels"
      state={state}
      caption={
        <MathText text="Left: the SORD target for the true class $y$, $t_k \propto \exp(-\alpha\,\phi(k, y))$, beside uniform label smoothing, which spreads its mass evenly whatever the distance. Right: the cross-entropy of each target against a confident prediction that puts 0.9 on class $c$. With a one-hot target every wrong $c$ costs the same; with the SORD target the cost grows with the distance from $y$. Larger $\alpha$ sharpens the target towards one-hot." />
      }
      readouts={
        <>
          <Readout label="target mass on y" value={formatNumber(target[truth])} />
          <Readout label="expected |k − y| under the target" value={formatNumber(meanDistance)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Plot x={kAxis} y={tAxis} height={280} ariaLabel="Soft ordinal target distribution">
          <Bars name="SORD target" x={left.classes} y={left.target} colors={colors} />
          <Curve name="uniform label smoothing (ε = 0.1)" x={left.classes} y={left.smooth} slot={1} dashed />
        </Plot>
        <Plot x={xAxis} y={yAxis} height={280} ariaLabel={'Cross-entropy against a confident prediction'}>
          <Curve {...right[0]} />
          <Curve {...right[1]} />
        </Plot>
      </div>
    </Figure>
  )
}
