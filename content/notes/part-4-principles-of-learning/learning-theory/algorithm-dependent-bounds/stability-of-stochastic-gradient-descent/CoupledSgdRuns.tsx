import { type ReactNode, useContext, useMemo } from 'react'
import { drawTrainingSet } from 'aifn-methods/theory'
import { linearSgdSteps } from 'aifn-methods/learning/linear'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  FrameContext,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'

const NOISE = 0.3
const PANEL = 320
const fmt = (v: number) => formatNumber(Number(v.toPrecision(3)))

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

/** One SGD run on a 1-d least-squares problem, epoch by epoch, from the given root stream. */
function run(x: Float64Array, y: Float64Array, eta: number, alpha: number, epochs: number, seed: string) {
  const problem = {
    x: tensor(Array.from(x, (v) => [v])),
    y: tensor(Array.from(y)),
    alpha,
    schedule: 'constant' as const,
    eta0: eta,
  }
  const tr = trace(linearSgdSteps(problem), {}, epochs, { stream: stream(seed), keep: 'all' })
  const states = tr.steps
  return {
    slope: states.map((s) => toFlat(s.weights)[0]),
    intercept: states.map((s) => s.intercept),
  }
}

/**
 * The coupling in the proof of stability for SGD: two runs on samples that differ in one example, visiting the
 * examples in the same order. Their parameters separate only when the differing example is visited.
 */
export function CoupledSgdRuns() {
  const state = useFigureState({
    m: int(20, { ge: 5, le: 200, suggestions: [10, 20, 50, 200], label: 'training points m' }),
    epochs: int(50, { ge: 1, le: 200, suggestions: [1, 5, 20, 50, 200], label: 'epochs' }),
    eta: float(0.02, { gt: 0, le: 1, scale: 'log10', suggestions: [0.01, 0.02, 0.1, 0.3], label: 'step size η' }),
    alpha: float(0, { ge: 0, le: 10, suggestions: [0, 0.1, 1], label: 'penalty λ on (λ/2)w²' }),
    zx: slider(-1, 1, 0.8, { step: 0.01, onChart: true }),
    zy: slider(-2, 2, -1.8, { step: 0.01, onChart: true }),
    seed: int(2, { ge: 0, le: 999, label: 'seed' }),
  })
  const { m, epochs, eta, alpha, zx, zy, seed } = state
  const data = useMemo(() => drawTrainingSet(stream(`sgd-stability/data/${seed}`), 'sine', m, NOISE), [m, seed])
  const runs = useMemo(() => {
    const x2 = Float64Array.from(data.x)
    const y2 = Float64Array.from(data.y)
    x2[0] = zx
    y2[0] = zy
    // The same root stream gives both runs the same order of examples in every epoch.
    const order = `sgd-stability/order/${seed}`
    return { a: run(data.x, data.y, eta, alpha, epochs, order), b: run(x2, y2, eta, alpha, epochs, order) }
  }, [data, zx, zy, eta, alpha, epochs, seed])
  const last = runs.a.slope.length - 1
  const line = (r: { slope: number[]; intercept: number[] }) => [
    r.intercept[last] - r.slope[last],
    r.intercept[last] + r.slope[last],
  ]
  const distance = Math.hypot(runs.a.slope[last] - runs.b.slope[last], runs.a.intercept[last] - runs.b.intercept[last])

  const dataX = useAxis({ label: 'x', range: [-1, 1] })
  const dataY = useAxis({ label: 'y', range: [-2, 2] })
  const bAxis = useAxis({ label: 'intercept b' })
  const wAxis = useAxis({ label: 'slope w', equal: bAxis })
  return (
    <Figure
      title="Two SGD runs that differ in one example"
      state={state}
      defaultSize="L"
      readouts={
        <>
          <Readout label="distance between final parameters" value={fmt(distance)} />
          <Readout label="examples visited per run" value={formatNumber(epochs * m)} />
        </>
      }
      caption={`Stochastic gradient descent fits a line y = wx + b by least squares (loss ½(y − wx − b)² plus (λ/2)w²) with a constant step size η, starting from w = b = 0. Left: the training points, drawn from y = sin(πx) + ε with noise standard deviation ${NOISE}. The grey point is the first example z₁; drag the large point to choose its replacement z₁′. Right: the parameters (b, w) after each epoch for the run on S (dashed) and the run on S⁽¹⁾ (solid). Both runs visit the examples in the same random order, as in the coupling argument of the stability proof, so they take identical steps except when the replaced example is visited. With the opening small step size the gap between the runs grows steadily with the number of epochs, as the bound 2L²ηT/m predicts, until it nears the distance between the two least-squares solutions. A larger η separates the runs within the first epoch, after which they jitter around those solutions. A penalty λ > 0 pulls both runs towards w = 0 and shrinks the gap. More training points make each visit to the replaced example a smaller share of the steps, so the runs stay closer.`}
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <FixedHeight height={PANEL}>
          <Plot x={dataX} y={dataY}>
            <Curve name="final fit on S" x={[-1, 1]} y={line(runs.a)} muted dashed />
            <Curve name="final fit on S⁽¹⁾" x={[-1, 1]} y={line(runs.b)} slot={0} />
            <Points name="shared training points" x={data.x.subarray(1)} y={data.y.subarray(1)} slot={1} size={5} />
            <Points name="replaced point z₁" x={[data.x[0]]} y={[data.y[0]]} muted size={8} />
            <Points name="replacement z₁′" x={[zx]} y={[zy]} emphasis size={9} />
            <Handle {...state.handle(['zx', 'zy'], { label: 'z₁′' })} />
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={bAxis} y={wAxis}>
            <Curve name="run on S" x={runs.a.intercept} y={runs.a.slope} muted dashed showPoints />
            <Curve name="run on S⁽¹⁾" x={runs.b.intercept} y={runs.b.slope} slot={0} showPoints />
            <Points
              name="final parameters"
              x={[runs.a.intercept[last], runs.b.intercept[last]]}
              y={[runs.a.slope[last], runs.b.slope[last]]}
              emphasis
              size={8}
            />
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
