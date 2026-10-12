import { circles, moons, xor } from 'aifn-methods/data/synthetic'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { useMemo } from 'react'
import { Player } from 'aifn-render/controls'
import { Dashboard, DashboardCell, DashboardRow, Equation, Figure, tex } from 'aifn-render/layout'
import { choice, float, int, row, toggle, useFigureState } from 'aifn-render/state'
import { Curve, Plot, Plots, Points, Raster, Readout, Segments, useAxis, useScaleColor } from 'aifn-render/viz'
import { f3, usePlayed } from '../_gaussian-processes/incremental'
import { predict, splineVmFit, splineVmProblem } from './fit'

const CELLS = 72

const DATASETS = {
  moons: {
    label: 'two moons',
    make: (seed: number, n: number, noise: number) => moons(stream(`svm2-${seed}`), { n, noise }),
  },
  xor: { label: 'XOR (uniform)', make: (seed: number, n: number) => xor(stream(`svm2-${seed}`), { n }) },
  circles: {
    label: 'circles',
    make: (seed: number, n: number, noise: number) => circles(stream(`svm2-${seed}`), { n, noise: noise / 2 }),
  },
}
type DatasetKey = keyof typeof DATASETS

const SCHEMA = {
  data: row('1 · data', {
    dataset: choice(
      (Object.keys(DATASETS) as DatasetKey[]).map((k) => ({ value: k, label: DATASETS[k].label })),
      'moons',
      { label: 'data' },
    ),
    n: int(300, { ge: 40, le: 800, label: 'n' }),
    noise: float(0.15, { ge: 0.01, le: 0.5, label: 'noise sd (not XOR)' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  }),
  model: row('2 · spline', {
    degree: choice(
      [
        { value: 1, label: '1 (linear)' },
        { value: 2, label: '2 (quadratic)' },
        { value: 3, label: '3 (cubic)' },
      ],
      3,
      { label: 'degree p' },
    ),
    candidates: int(6, { ge: 2, le: 12, label: 'candidate knots per axis' }),
    interactions: toggle(true, 'corner terms'),
    penalty: float(0, { ge: 0, le: 20, suggestions: [0, 1, 2, 3, 5], label: 'knot cost λ (nats)' }),
    moveKnots: toggle(true, 'move knots'),
  }),
  run: row('fit', {
    iterations: int(400, { ge: 10, le: 5000, suggestions: [100, 400, 1000, 3000], label: 'iterations (at most)' }),
  }),
}

export function SplineVm2dShowcase() {
  const state = useFigureState(SCHEMA)
  const { n, noise, seed } = state.data
  const { candidates, interactions, penalty, moveKnots } = state.model
  const degree = Number(state.model.degree)
  const maxIter = state.run.iterations
  const key = state.data.dataset as DatasetKey

  const data = useMemo(() => {
    const d = DATASETS[key].make(seed, n, noise)
    const rows = toRows(d.x as Tensor) as number[][]
    const ys = Array.from(toFlat(d.y as Tensor), (v) => (v > 0.5 ? 1 : 0))
    const pad = (k: number) => {
      const v = rows.map((r) => r[k])
      const lo = Math.min(...v)
      const hi = Math.max(...v)
      return [lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo)] as [number, number]
    }
    return { rows, ys, rx: pad(0), ry: pad(1) }
  }, [key, n, noise, seed])

  // ── The fit: every iteration, computed once per setting ──
  const problem = useMemo(() => splineVmProblem(data.rows, data.ys, degree, 'bernoulli'), [data, degree])
  const states = useMemo(
    () => splineVmFit(problem, { candidates, interactions, moveKnots, penalty, maxIter, tolerance: 1e-3, window: 20 }),
    [problem, candidates, interactions, moveKnots, penalty, maxIter],
  )
  const runKey = JSON.stringify([key, n, noise, seed, degree, candidates, interactions, moveKnots, penalty, maxIter])
  const [pos, setPos] = usePlayed(runKey, states.length)
  const s = states[pos]
  const toX = useMemo(() => (d: number, t: number) => problem.lo[d] + t * problem.span[d], [problem])

  // ── The surface: the moderated P(y = 1) on a grid, and the training accuracy ──
  const grid = useMemo(() => {
    const gx = Array.from({ length: CELLS }, (_, j) => data.rx[0] + ((j + 0.5) * (data.rx[1] - data.rx[0])) / CELLS)
    const gy = Array.from({ length: CELLS }, (_, i) => data.ry[0] + ((i + 0.5) * (data.ry[1] - data.ry[0])) / CELLS)
    return { gx, gy, points: gy.flatMap((y) => gx.map((x) => [x, y])) }
  }, [data])
  const surface = useMemo(() => {
    const prob = predict(problem, s, grid.points, false).probability
    const z = grid.gy.map((_, i) => Array.from(prob.subarray(i * CELLS, (i + 1) * CELLS)))
    const atData = predict(problem, s, data.rows, false).probability
    const accuracy = data.ys.filter((y, i) => (atData[i] > 0.5 ? 1 : 0) === y).length / data.ys.length
    return { z, accuracy }
  }, [problem, s, grid, data])

  // ── Knots: lines for one-axis knots, dots for corners, both coloured by weight ──
  const diverging = useScaleColor('diverging')
  const wFull = useMemo(() => {
    const all = states.flatMap((st) => st.weights.map(Math.abs)).sort((a, b) => a - b)
    return all.length ? Math.max(all[Math.floor(0.95 * (all.length - 1))], 1e-12) : 1
  }, [states])
  const colourOf = useMemo(
    () => (w: number) => diverging(0.5 + 0.5 * Math.sign(w) * Math.sqrt(Math.min(1, Math.abs(w) / wFull))),
    [diverging, wFull],
  )
  const knots = useMemo(() => {
    const lines: { from: [number, number]; to: [number, number] }[] = []
    const edge = { x: [] as number[], y: [] as number[], colors: [] as string[] }
    const corners = { x: [] as number[], y: [] as number[], colors: [] as string[] }
    s.terms.forEach((t, j) => {
      const c = colourOf(s.weights[j])
      if (t.dims.length === 2) {
        corners.x.push(toX(0, t.at[0]))
        corners.y.push(toX(1, t.at[1]))
        corners.colors.push(c)
      } else if (t.dims[0] === 0) {
        const x = toX(0, t.at[0])
        lines.push({ from: [x, data.ry[0]], to: [x, data.ry[1]] })
        edge.x.push(x)
        edge.y.push(data.ry[0])
        edge.colors.push(c)
      } else {
        const y = toX(1, t.at[0])
        lines.push({ from: [data.rx[0], y], to: [data.rx[1], y] })
        edge.x.push(data.rx[0])
        edge.y.push(y)
        edge.colors.push(c)
      }
    })
    return { lines, edge, corners }
  }, [s, data, toX, colourOf])
  const series = useMemo(
    () => ({
      x: states.map((_, i) => i),
      objective: states.map((st) => st.objective),
      main: states.map((st) => st.terms.filter((t) => t.dims.length === 1).length),
      corners: states.map((st) => st.terms.filter((t) => t.dims.length === 2).length),
    }),
    [states],
  )

  const xa = useAxis({ label: 'x₁', range: data.rx, key: runKey })
  const ya = useAxis({ label: 'x₂', range: data.ry, key: runKey, equal: xa })
  const iterAxis = useAxis({ label: 'iteration', range: [0, Math.max(1, states.length - 1)], key: runKey })
  const evAxis = useAxis({
    label: `log p(y | α, t)${penalty > 0 ? ' − λK' : ''}`,
    hold: 'initial',
    key: runKey,
  })
  const countAxis = useAxis({ label: 'terms', range: [0, undefined], key: runKey })
  const xs = data.rows.map((r) => r[0])
  const ys2 = data.rows.map((r) => r[1])
  const nMain = s.terms.filter((t) => t.dims.length === 1).length
  const nCorner = s.terms.length - nMain

  return (
    <Figure
      title="SplineVM in two dimensions: the prediction surface"
      purpose="The same sparse Bayesian spline on two inputs: knots on each axis and corner terms (products of a knot on each axis), selected by the evidence and moved by its gradient. The surface is the predicted class probability."
      state={state}
      defaultSize="XL"
      equation={
        <Equation>
          {tex`P(y = 1) = \sigma\Big(\sum_{e} c_e u_1^{e_1} u_2^{e_2} + \sum_{j} w_j\,(u_{d_j} - t_j)_+^{p} + \sum_{k} v_k\,(u_1 - a_k)_+^{p}\,(u_2 - b_k)_+^{p}\Big)`}
        </Equation>
      }
      controls={
        <Player
          label="3 · play"
          value={pos}
          onChange={setPos}
          count={states.length}
          format={(p) => {
            const st = states[p]
            if (!st) return `iteration ${p}`
            const removed = st.pruned.length ? `, ${st.pruned.length} removed` : ''
            return `iteration ${p}: ${st.terms.length} terms${removed}${st.converged ? ' (converged)' : ''}`
          }}
        />
      }
      readouts={{
        'at this iteration': (
          <>
            <Readout label="axis knots" value={nMain} />
            <Readout label="corner terms" value={nCorner} />
            <Readout label="log evidence (Laplace)" value={f3(s.logEvidence)} />
            {penalty > 0 && <Readout label="objective (− λ per term)" value={f3(s.objective)} />}
            <Readout label="training accuracy" value={f3(surface.accuracy)} />
            <Readout label="|w| at full colour" value={f3(wFull)} />
            <Readout
              label="iterations"
              value={`${states.length - 1}${states.at(-1)?.converged ? ' (converged)' : ''}`}
            />
          </>
        ),
      }}
      caption={`${n} labelled points (aifn’s ${key === 'moons' ? 'moons' : key === 'xor' ? 'xor' : 'circles'}). Each input is rescaled to [0, 1]. The latent function f has a tensor polynomial part (every u₁^a u₂^b with a, b ≤ ${degree}) with a fixed prior, a truncated power per axis knot (a ridge: the ${degree === 1 ? 'slope' : 'p-th derivative'} along that axis jumps at the knot, everywhere along the other axis)${interactions ? ', and a corner term per pair of knots, (u₁ − a)₊^p (u₂ − b)₊^p, which is non-zero only in the quadrant above and to the right of its corner (as in MARS)' : ''}. Each term weight has its own ARD precision. The fit starts from ${candidates} candidate knots per axis${interactions ? ` and the ${candidates * candidates} corners they make` : ''}, and repeats: the best αⱼ for every term given the others (Tipping and Faul's sⱼ, qⱼ from the Laplace approximation; a term is removed when keeping it gains no more than λ = ${penalty} nats of evidence), then ${moveKnots ? 'one gradient step on every knot location and corner' : 'nothing more (knot moves are off)'}. The run stops after ${maxIter} iterations or once 20 iterations together gain under 10⁻³ nats. Left: the predicted P(y = 1) (moderated by the uncertainty in f) with its 0.5 contour, the data by class, axis knots as dashed lines with a dot at the edge, and corner terms as dots, each dot coloured by its weight (diverging: red positive, blue negative; square-root scale, full at the readout’s |w|). Right: the log evidence${penalty > 0 ? ' minus λ per term' : ''} and the numbers of axis knots and corner terms per iteration. Prototype: local numerics in the lab, not yet in the engine.`}
    >
      <Dashboard>
        <DashboardRow minHeight={420}>
          <DashboardCell ratio={1.6}>
            <Plot x={xa} y={ya}>
              <Raster
                x={grid.gx}
                y={grid.gy}
                z={surface.z}
                scale="sequential"
                range={[0, 1]}
                boundary
                valueLabel="P(y = 1)"
                fillOpacity={0.85}
              />
              <Points name="data" x={xs} y={ys2} group={data.ys} groupNames={['y = 0', 'y = 1']} thin />
              {knots.lines.length > 0 && <Segments name="axis knots" segments={knots.lines} dashed />}
              {knots.edge.x.length > 0 && (
                <Points
                  name="axis knot weight"
                  x={knots.edge.x}
                  y={knots.edge.y}
                  colors={knots.edge.colors}
                  size={11}
                />
              )}
              {knots.corners.x.length > 0 && (
                <Points
                  name="corner weight"
                  x={knots.corners.x}
                  y={knots.corners.y}
                  colors={knots.corners.colors}
                  size={12}
                  shape={3}
                />
              )}
            </Plot>
          </DashboardCell>
          <DashboardCell>
            <Plots rows={2} tight>
              <Plot x={iterAxis} y={evAxis}>
                <Curve name={penalty > 0 ? 'objective' : 'log evidence'} x={series.x} y={series.objective} slot={0} />
                <Points name="shown" x={[pos]} y={[series.objective[pos]]} emphasis live />
              </Plot>
              <Plot x={iterAxis} y={countAxis}>
                <Curve name="axis knots" x={series.x} y={series.main} slot={1} />
                <Curve name="corner terms" x={series.x} y={series.corners} slot={2} />
                <Points name="shown" x={[pos, pos]} y={[series.main[pos], series.corners[pos]]} emphasis live />
              </Plot>
            </Plots>
          </DashboardCell>
        </DashboardRow>
      </Dashboard>
    </Figure>
  )
}
