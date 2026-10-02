import { useMemo, useState } from 'react'
import { MathText } from '@/components/content/MathText'
import { Interactive, ParamChoice, ParamSlider, Readout, XYChart, formatNumber, type XYSeries } from 'aifn-render'
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
  const [k, setK] = useState(7)
  const [y, setY] = useState(3)
  const [alpha, setAlpha] = useState(1)
  const [distance, setDistance] = useState<Distance>('absolute')
  const truth = Math.min(y, k) - 1
  const colors = useClassColors(k)

  const { target, left, right, meanDistance } = useMemo(() => {
    const target = sord(k, truth, alpha, distance)
    const oneHot = target.map((_, j) => (j === truth ? 1 : 0))
    const smooth = target.map((_, j) => (j === truth ? 0.9 : 0) + 0.1 / k)
    const classes = target.map((_, j) => j + 1)
    const left: XYSeries[] = [
      { name: 'SORD target', type: 'bar', x: classes, y: target, pointColors: colors },
      { name: 'uniform label smoothing (ε = 0.1)', type: 'line', x: classes, y: smooth, slot: 1, dashed: true },
    ]
    const right: XYSeries[] = [
      { name: 'one-hot target', type: 'line', x: classes, y: classes.map((_, c) => crossEntropy(oneHot, c)), slot: 2 },
      { name: 'SORD target', type: 'line', x: classes, y: classes.map((_, c) => crossEntropy(target, c)), slot: 0 },
    ]
    const meanDistance = target.reduce((s, t, j) => s + t * Math.abs(j - truth), 0)
    return { target, left, right, meanDistance }
  }, [k, truth, alpha, distance, colors])

  return (
    <Interactive
      title="Soft ordinal labels"
      caption={
        <MathText text="Left: the SORD target for the true class $y$, $t_k \propto \exp(-\alpha\,\phi(k, y))$, beside uniform label smoothing, which spreads its mass evenly whatever the distance. Right: the cross-entropy of each target against a confident prediction that puts 0.9 on class $c$. With a one-hot target every wrong $c$ costs the same; with the SORD target the cost grows with the distance from $y$. Larger $\alpha$ sharpens the target towards one-hot." />
      }
      controls={
        <>
          <ParamSlider label="classes K" value={k} onChange={setK} min={3} max={10} step={1} />
          <ParamSlider label="true class y" value={Math.min(y, k)} onChange={setY} min={1} max={k} step={1} />
          <ParamSlider label="sharpness α" value={alpha} onChange={setAlpha} min={0.1} max={4} step={0.05} />
          <ParamChoice label="distance φ" value={distance} onChange={setDistance} options={DISTANCES} />
        </>
      }
      readout={
        <>
          <Readout label="target mass on y" value={formatNumber(target[truth])} />
          <Readout label="expected |k − y| under the target" value={formatNumber(meanDistance)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <XYChart
          series={left}
          xRange={[0.5, k + 0.5]}
          integerX
          yRange={[0, 1]}
          xLabel="class k"
          yLabel="target probability"
          height={280}
          ariaLabel="Soft ordinal target distribution"
        />
        <XYChart
          series={right}
          xRange={[1, k]}
          yRange={[0, undefined]}
          xLabel="predicted class c"
          yLabel="cross-entropy"
          height={280}
          ariaLabel="Cross-entropy against a confident prediction"
        />
      </div>
    </Interactive>
  )
}
