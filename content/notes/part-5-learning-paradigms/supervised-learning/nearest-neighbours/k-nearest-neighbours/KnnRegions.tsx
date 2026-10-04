import { useMemo } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Raster,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normal, stream, type Stream, uniform } from 'aifn/foundation/random'

type P = [number, number]

const PER_CLASS = 100
const N_TEST = 2000
const CENTRES = 6
const SPREAD = 0.45
const K_MAX = 150
const GRID = toFlat(linspace(-3.5, 3.5, 61))
const RANGE: [number, number] = [0, 1]

/** Each class is a mixture of six Gaussian bumps whose centres are themselves drawn around a class centre. */
function makeWorld(seed: number) {
  const g = stream(seed)
  const centres: P[][] = [
    [-0.6, 0.6],
    [0.6, -0.6],
  ].map(([cx, cy]) => Array.from({ length: CENTRES }, (): P => [cx + 1.1 * normal(g), cy + 1.1 * normal(g)]))
  const draw = (label: number, n: number, h: Stream): P[] =>
    Array.from({ length: n }, () => {
      const c = centres[label][Math.floor(uniform(h) * CENTRES)]
      return [c[0] + SPREAD * normal(h), c[1] + SPREAD * normal(h)]
    })
  // Class-conditional density up to a constant shared by both classes.
  const density = (label: number, p: P) =>
    centres[label].reduce((a, c) => a + Math.exp(-((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2) / (2 * SPREAD * SPREAD)), 0)
  const train = [...draw(0, PER_CLASS, g), ...draw(1, PER_CLASS, g)]
  const labels = train.map((_, i) => (i < PER_CLASS ? 0 : 1))
  const h = stream(seed + 500)
  const testLabels = Array.from({ length: N_TEST }, (_, i) => (i < N_TEST / 2 ? 0 : 1))
  const test = testLabels.map((l) => draw(l, 1, h)[0])
  return { train, labels, test, testLabels, density }
}

/** cum[q][r] = number of class-1 points among the r nearest training points to query q (r = 0 … n). */
function neighbourCounts(queries: P[], train: P[], labels: number[]): Uint16Array[] {
  return queries.map((q) => {
    const order = train.map((p, i) => [(p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2, i] as const).sort((a, b) => a[0] - b[0])
    const cum = new Uint16Array(order.length + 1)
    order.forEach(([, i], r) => (cum[r + 1] = cum[r] + labels[i]))
    return cum
  })
}

/** k-nearest-neighbour classification on two classes: decision regions and error against k. */
export function KnnRegions() {
  const state = useFigureState({
    k: int(15, { min: 1, max: K_MAX, step: 1, label: 'neighbours k', format: (v) => String(v) }),
    seed: int(2, { min: 1, max: 20, step: 1, label: 'data seed', format: (v) => String(v) }),
  })

  const world = useMemo(() => makeWorld(state.seed), [state.seed])
  const pre = useMemo(() => {
    const { train, labels, test, testLabels, density } = world
    const gridPts = GRID.flatMap((y) => GRID.map((x): P => [x, y]))
    const gridCum = neighbourCounts(gridPts, train, labels)
    const trainCum = neighbourCounts(train, train, labels)
    const testCum = neighbourCounts(test, train, labels)
    // A tie (exactly half the votes) counts as half an error, as if broken by a coin.
    const errorAt = (cum: Uint16Array[], truth: number[], kk: number) =>
      cum.reduce((a, c, i) => {
        const f = c[kk] / kk
        return a + (f === 0.5 ? 0.5 : (f > 0.5 ? 1 : 0) === truth[i] ? 0 : 1)
      }, 0) / truth.length
    const ks = Array.from({ length: K_MAX }, (_, i) => i + 1)
    const trainErr = ks.map((kk) => errorAt(trainCum, labels, kk))
    const testErr = ks.map((kk) => errorAt(testCum, testLabels, kk))
    const bayes =
      test.reduce((a, p, i) => a + ((density(1, p) > density(0, p) ? 1 : 0) === testLabels[i] ? 0 : 1), 0) / N_TEST
    return { gridCum, ks, trainErr, testErr, bayes }
  }, [world])

  const z = useMemo(
    () => GRID.map((_, row) => GRID.map((__, col) => pre.gridCum[row * GRID.length + col][state.k] / state.k)),
    [pre, state.k],
  )
  const overlay = useMemo(
    () =>
      [
        {
          name: 'points',
          x: world.train.map((p) => p[0]),
          y: world.train.map((p) => p[1]),
          group: world.labels,
          groupNames: ['class 0', 'class 1'],
        },
      ] as const,
    [world],
  )

  const best = pre.testErr.indexOf(Math.min(...pre.testErr)) + 1
  const errorSeries = [
    { name: 'training error', x: pre.ks, y: pre.trainErr, slot: 0 },
    { name: 'test error', x: pre.ks, y: pre.testErr, slot: 1 },
    { name: 'Bayes error', x: [1, K_MAX], y: [pre.bayes, pre.bayes], muted: true, dashed: true },
  ] as const

  const xAxis = useAxis({ label: 'x₁' })
  const yAxis = useAxis({ label: 'x₂' })
  const xAxis2 = useAxis({ label: 'k', hold: 'union' })
  const yAxis2 = useAxis({ label: 'error rate', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="k-nearest neighbours: decision regions and the choice of k"
      state={state}
      caption="Each class is a mixture of six Gaussian bumps; 100 training points per class. Left: shading is the fraction of the k nearest training points in class 1, and the classifier predicts class 1 where it exceeds one half. With k = 1 every training point owns a cell and training error is zero. Large k averages over a wide neighbourhood and smooths away real structure. Right: training and test error against k, with the Bayes error of the true mixture. Drag the line labelled k, or use the slider."

      readouts={
        <>
          <Readout label="training error" value={formatNumber(pre.trainErr[state.k - 1])} />
          <Readout label="test error" value={formatNumber(pre.testErr[state.k - 1])} />
          <Readout label="best k (test)" value={String(best)} />
          <Readout label="Bayes error" value={formatNumber(pre.bayes)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={360}>
          <Raster
            x={GRID}
            y={GRID}
            z={z}
            scale={'diverging'}
            range={RANGE}
            valueLabel={'fraction of neighbours in class 1'}
          />
          <Points {...overlay[0]} live />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={360}>
          <Curve {...errorSeries[0]} />
          <Curve {...errorSeries[1]} />
          <Curve {...errorSeries[2]} />
          <Handle kind="x" at={state.k} label="k" onDrag={(x) => state.set('k', Math.round(x))} />
        </Plot>
      </div>
    </Figure>
  )
}
