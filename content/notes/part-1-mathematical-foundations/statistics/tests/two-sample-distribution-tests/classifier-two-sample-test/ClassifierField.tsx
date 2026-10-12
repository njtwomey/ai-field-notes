import { useMemo } from 'react'
import {
  Bars,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Raster,
  Readout,
  Segments,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { classifierTest, twoDimensionalPair } from '../_shared/twoSample'

const BOX = 4
const G = toFlat(linspace(-BOX, BOX, 41))
const GRID = G.flatMap((y) => G.map((x) => [x, y]))
const clamp = (v: number) => Math.min(BOX - 0.5, Math.max(-BOX + 0.5, v))

/**
 * The classifier two-sample test on the two-dimensional problem: k-nearest neighbours learns to tell P's points from
 * Q's on half of each sample, and its accuracy on the other half is tested against the Binomial(n_test, ½) law that
 * chance would give.
 */
export function ClassifierField() {
  const state = useFigureState({
    mx: float(0.8, { min: -3, max: 3, step: 0.05, label: 'mean of Q, x₁', onChart: true }),
    my: float(0, { min: -3, max: 3, step: 0.05, label: 'mean of Q, x₂', onChart: true }),
    rho: float(0, { min: -0.95, max: 0.95, step: 0.05, label: 'correlation of Q, ρ' }),
    k: int(15, { min: 1, max: 75, label: 'neighbours k', suggestions: [1, 15, 75] }),
    n: int(150, { min: 20, max: 400, label: 'points in each sample', suggestions: [40, 150, 400] }),
    seed: int(1, { min: 0, max: 999, label: 'seed' }),
  })

  const data = useMemo(
    () => twoDimensionalPair(state.seed, state.n, state.n, [state.mx, state.my], state.rho),
    [state.seed, state.n, state.mx, state.my, state.rho],
  )
  const fit = useComputed(() => classifierTest(data.x, data.y, state.k, GRID), [data, state.k], { mode: 'release' })
  const t = fit.value
  // Signed around ½, so the regions that favour each sample take the two ends of the diverging scale.
  const field = G.map((_, r) => t.field.slice(r * G.length, (r + 1) * G.length).map((p) => p - 0.5))

  // The null law of the number correct, drawn as a law of the accuracy.
  const nullBars = useMemo(() => {
    const ks = Array.from({ length: t.nTest + 1 }, (_, i) => i)
    return { x: ks.map((i) => i / t.nTest), y: ks.map((i) => t.test.null.prob(i) as number) }
  }, [t])
  const top = Math.max(...nullBars.y)

  const split = (s: number[][], odd: 0 | 1) => s.filter((_, i) => i % 2 === odd)
  const xa = useAxis({ label: 'x₁', range: [-BOX, BOX] })
  const ya = useAxis({ label: 'x₂', range: [-BOX, BOX], equal: xa })
  const xb = useAxis({ label: 'held-out accuracy', range: [0.2, 1] })
  const yb = useAxis({ label: 'probability under the null', range: [0, 1.1 * top] })
  return (
    <Figure
      title="A classifier as a two-sample test"
      state={state}
      caption="Left: the classifier's probability that a point came from Q, minus ½, learned from the training half of each sample (small marks); the larger marks are the held-out halves it is scored on. Drag the marker to move Q's mean. Right: the law of the held-out accuracy if P and Q were the same, Binomial(n_test, ½) divided by n_test, and the accuracy reached (black line). An accuracy far into the right tail rejects. With equal means and ρ near ±0.9 the classifier must learn a diagonal pattern; few neighbours make it noisy, too many wash the pattern out."
      readouts={
        <>
          <Readout label="held-out accuracy" value={formatNumber(t.accuracy)} />
          <Readout label="held-out points" value={t.nTest} />
          <Readout label="binomial p-value" value={formatNumber(t.pValue)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={xa} y={ya}>
          <Raster
            x={G}
            y={G}
            z={field}
            scale="diverging"
            range={[-0.5, 0.5]}
            valueLabel="P(from Q | x) − ½"
            boundary
            stale={fit.stale}
            fillOpacity={0.6}
          />
          <Points
            name="P, training"
            x={split(data.x, 0).map((p) => p[0])}
            y={split(data.x, 0).map((p) => p[1])}
            slot={0}
            thin
          />
          <Points
            name="Q, training"
            x={split(data.y, 0).map((p) => p[0])}
            y={split(data.y, 0).map((p) => p[1])}
            slot={1}
            thin
          />
          <Points
            name="P, held out"
            x={split(data.x, 1).map((p) => p[0])}
            y={split(data.x, 1).map((p) => p[1])}
            slot={0}
          />
          <Points
            name="Q, held out"
            x={split(data.y, 1).map((p) => p[0])}
            y={split(data.y, 1).map((p) => p[1])}
            slot={1}
          />
          <Handle
            kind="point"
            at={[state.mx, state.my]}
            label="mean of Q"
            onDrag={([x, y]) => {
              state.set('mx', clamp(x))
              state.set('my', clamp(y))
            }}
          />
        </Plot>
        <Plot x={xb} y={yb}>
          <Bars name="null law" x={nullBars.x} y={nullBars.y} slot={2} opacity={0.6} />
          <Segments
            name="accuracy reached"
            segments={[{ from: [t.accuracy, 0], to: [t.accuracy, 1.1 * top] }]}
            emphasis
            width={2}
          />
        </Plot>
      </div>
    </Figure>
  )
}
