import { useMemo, useState } from 'react'
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
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

type Vec = [number, number]
type Move = 'start' | 'reflect' | 'expand' | 'contract outside' | 'contract inside' | 'shrink'

/** Rosenbrock's banana function: minimum 0 at (1, 1), at the end of a long curved valley. */
const rosenbrock = ([x, y]: Vec) => (1 - x) ** 2 + 100 * (y - x * x) ** 2
const X_RANGE: Vec = [-2, 2]
const Y_RANGE: Vec = [-1, 3]
const GRID = 81
const ITERATIONS = 200
/** Floor for f on a log axis once the simplex has converged to machine precision. */
const FLOOR = 1e-14

const add = (a: Vec, b: Vec, t: number): Vec => [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]

/** Standard Nelder–Mead coefficients: reflection 1, expansion 2, contraction ½, shrink ½. */
function nelderMead(start: Vec, size: number) {
  let simplex: Vec[] = [start, [start[0] + size, start[1]], [start[0], start[1] + size]]
  const history: { simplex: Vec[]; move: Move }[] = [{ simplex, move: 'start' }]
  for (let k = 0; k < ITERATIONS; k++) {
    const [best, second, worst] = [...simplex].sort((a, b) => rosenbrock(a) - rosenbrock(b))
    const fb = rosenbrock(best)
    const fs = rosenbrock(second)
    const fw = rosenbrock(worst)
    const c: Vec = [(best[0] + second[0]) / 2, (best[1] + second[1]) / 2]
    const r = add(c, worst, -1)
    const fr = rosenbrock(r)
    let move: Move
    if (fr < fb) {
      const e = add(c, worst, -2)
      move = rosenbrock(e) < fr ? 'expand' : 'reflect'
      simplex = [best, second, move === 'expand' ? e : r]
    } else if (fr < fs) {
      move = 'reflect'
      simplex = [best, second, r]
    } else {
      const outside = fr < fw
      const t = outside ? add(c, r, 0.5) : add(c, worst, 0.5)
      if (rosenbrock(t) < (outside ? fr : fw)) {
        move = outside ? 'contract outside' : 'contract inside'
        simplex = [best, second, t]
      } else {
        move = 'shrink'
        simplex = [best, add(best, second, 0.5), add(best, worst, 0.5)]
      }
    }
    history.push({ simplex, move })
  }
  return history
}

const clamp = (v: number, [lo, hi]: Vec) => Math.round(Math.min(Math.max(v, lo), hi) * 100) / 100

export function NelderMeadExplorer() {
  const [start, setStart] = useState<Vec>([-1.2, 1])
  const state = useFigureState({
    iteration: int(10, { min: 0, max: ITERATIONS, step: 1, label: 'iteration', format: (v) => String(v) }),
  })

  const grid = useMemo(() => {
    const x = toFlat(linspace(X_RANGE[0], X_RANGE[1], GRID))
    const y = toFlat(linspace(Y_RANGE[0], Y_RANGE[1], GRID))
    return { x, y, z: y.map((yv) => x.map((xv) => Math.log10(rosenbrock([xv, yv]) + 0.1))) }
  }, [])

  const history = useMemo(() => nelderMead(start, 0.4), [start])
  const bestPath = useMemo(
    () => history.map((h) => h.simplex.reduce((a, b) => (rosenbrock(b) < rosenbrock(a) ? b : a))),
    [history],
  )
  const current = history[state.iteration]

  const overlay = useMemo(() => {
    const trail = bestPath.slice(0, state.iteration + 1)
    const s = current.simplex
    return [
      { name: 'best vertex so far', x: trail.map((p) => p[0]), y: trail.map((p) => p[1]), slot: 1 },
      {
        name: 'simplex',
        x: [...s.map((p) => p[0]), s[0][0]],
        y: [...s.map((p) => p[1]), s[0][1]],
        slot: 2,
        showPoints: true,
      },
      { name: 'minimum', x: [1], y: [1], emphasis: true },
    ] as const
  }, [bestPath, current, state.iteration])

  const curve = useMemo(
    () =>
      [
        {
          name: 'best f',
          x: bestPath.map((_, k) => k),
          y: bestPath.map((p) => Math.max(rosenbrock(p), FLOOR)),
          slot: 1,
        },
      ] as const,
    [bestPath],
  )

  const best = bestPath[state.iteration]

  const xAxis = useAxis({ label: 'x' })
  const yAxis = useAxis({ label: 'y' })
  const xAxis2 = useAxis({ label: 'iteration', hold: 'union' })
  const yAxis2 = useAxis({ label: 'best f', hold: 'union', log: true })
  return (
    <Figure
      title="Nelder–Mead on the Rosenbrock function"
      state={state}
      caption="Left: log₁₀(f + 0.1) for f(x, y) = (1 − x)² + 100(y − x²)², the simplex at the chosen iteration, and the path of its best vertex. Drag the start point; the first simplex has sides 0.4 along the axes. Right: the best function value per iteration on a log scale; drag the vertical line, or step the slider, to replay. Watch the simplex stretch along the valley with expansions and shrink across it with contractions."

      readouts={
        <>
          <Readout label="move" value={current.move} />
          <Readout label="best vertex" value={`(${formatNumber(best[0])}, ${formatNumber(best[1])})`} />
          <Readout label="best f" value={formatNumber(rosenbrock(best))} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={340}>
          <Raster x={grid.x} y={grid.y} z={grid.z} valueLabel={'log₁₀(f + 0.1)'} />
          <Curve {...overlay[0]} live />
          <Curve {...overlay[1]} live />
          <Points {...overlay[2]} live />
          <Handle
            kind="point"
            at={start}
            label="start"
            onDrag={([x, y]) => setStart([clamp(x, X_RANGE), clamp(y, Y_RANGE)])}
          />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={340}>
          <Curve {...curve[0]} />
          <Handle {...state.handle('iteration', { label: 'iteration' })} />
        </Plot>
      </div>
    </Figure>
  )
}
