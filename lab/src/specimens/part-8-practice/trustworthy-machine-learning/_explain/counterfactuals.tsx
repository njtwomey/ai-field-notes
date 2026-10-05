/**
 * Counterfactual explanations (`aifn/learning/explain`): an MLP trained on the feasibility task (two classes joined by
 * a corridor of data, empty space between them) and a pinned point explained by Wachter et al.'s gradient search, DiCE
 * and FACE. Wachter and DiCE take the shortest changes, which cross the boundary where no data lie; FACE walks from
 * data point to data point through dense regions. A second figure shows Growing Spheres' shells.
 */
import { useMemo, useState } from 'react'
import { feasibilityTask } from 'aifn-methods/data/synthetic'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import {
  denseFunction,
  denseOutput,
  diverseCounterfactuals,
  faceGraph,
  faceSearch,
  growingSpheres,
  medianAbsoluteDeviation,
  wachterCounterfactual,
  type Actionability,
} from 'aifn-compute/learning/explain'
import { sigmoid } from 'aifn-compute/numerics/special'
import { multivariateKde } from 'aifn-compute/probability/stats'
import { Player } from 'aifn-render/controls'
import { ControlRow, Figure } from 'aifn-render/layout'
import {
  call,
  choice,
  int,
  pinField,
  row,
  setting,
  slider,
  toggle,
  useComputed,
  useFigureState,
  usePinned,
} from 'aifn-render/state'
import { Contours, Curve, Plot, Points, Raster, Readout, Segments, useAxis } from 'aifn-render/viz'
import { TrainRow } from './training'
import { fmt, useTrainedMlp, type MlpSetup } from './mlp'

const XR: [number, number] = [-3, 3]
const YR: [number, number] = [-2.2, 2.4]
const GX = Array.from({ length: 61 }, (_, i) => XR[0] + ((XR[1] - XR[0]) * i) / 60)
const GY = Array.from({ length: 47 }, (_, i) => YR[0] + ((YR[1] - YR[0]) * i) / 46)
const GRID = fromData(Float64Array.from(GY.flatMap((y) => GX.map((x) => [x, y])).flat()), [GX.length * GY.length, 2])

type Setup = { n: number; seed: number; width: number; steps: number }

const sigmoidOf = (z: Float64Array) => Float64Array.from(z, (v) => sigmoid(v) as number)

export function CounterfactualShowcase() {
  const state = useFigureState({
    data: row('1 · data and model', {
      n: int(300, { ge: 60, le: 2000, suggestions: [200, 300, 600], label: 'points n' }),
      seed: slider(0, 20, 0, { step: 1, label: 'seed' }),
      width: int(16, { ge: 2, le: 64, suggestions: [8, 16, 32], label: 'hidden units' }),
      steps: int(600, { ge: 1, le: 5000, suggestions: [300, 600, 1500], label: 'Adam steps' }),
    }),
    face: row('2 · FACE graph', {
      graph: choice(
        [
          { value: 'kde', label: 'ε-graph, KDE weights' },
          { value: 'knn', label: 'kNN graph, k-NN weights' },
          { value: 'epsilon', label: 'ε-graph, length weights' },
        ],
        'kde',
        { label: 'graph' },
      ),
      epsilon: slider(0.15, 1.5, 0.45, { step: 0.05, label: 'ε (longest step)' }),
      density: slider(0, 0.9, 0.25, { step: 0.01, label: 'density threshold t_d' }),
      prediction: slider(0.5, 0.99, 0.75, { step: 0.01, label: 'target probability t_p' }),
    }),
    act: row('3 · actionability', {
      fixX1: setting(false, 'x₁ immutable'),
      fixX2: setting(false, 'x₂ immutable'),
      upX2: setting(false, 'x₂ may only rise'),
    }),
    reveal: row('4 · show', {
      wachter: toggle(true, 'Wachter'),
      dice: toggle(true, 'DiCE'),
      face: toggle(true, 'FACE'),
      edges: toggle(false, 'FACE graph'),
    }),
    pin: pinField(),
  })
  const current: Setup = { n: state.data.n, seed: state.data.seed, width: state.data.width, steps: state.data.steps }
  const [trained, setTrained] = useState<Setup | null>(null)
  const stale = trained !== null && JSON.stringify(trained) !== JSON.stringify(current)
  const shown = trained ?? current
  const data = useMemo(() => feasibilityTask(stream(`cf-${shown.seed}`), { n: shown.n }), [shown.seed, shown.n])
  const X = useMemo(() => Float64Array.from(toFlat(data.x)), [data])
  const y = useMemo(() => Array.from(toFlat(data.y!)), [data])
  const n = y.length
  const setup = useMemo(
    (): MlpSetup | null =>
      trained && {
        data: call('data/synthetic/feasibilityTask', call('foundation/random/stream', `cf-${trained.seed}`), {
          n: trained.n,
        }),
        inputs: 2,
        width: trained.width,
        depth: 2,
        activation: 'tanh',
        steps: trained.steps,
        rate: 0.02,
        l2: 1e-3,
        seed: trained.seed,
      },
    [trained],
  )
  const mlp = useTrainedMlp(setup)
  const net = mlp.net
  // The query: a pinned row, else the class-0 row nearest the left blob's centre.
  const pin = usePinned(state.pin, (v) => state.set('pin', v), { valid: (i) => i < n })
  const fallback = useMemo(() => {
    let best = 0
    let bd = Infinity
    for (let i = 0; i < n; i++) {
      const dd = (X[2 * i] + 1.6) ** 2 + (X[2 * i + 1] - 0.6) ** 2
      if (y[i] === 0 && dd < bd) {
        bd = dd
        best = i
      }
    }
    return best
  }, [X, y, n])
  const focus = pin.pinned ?? fallback
  const query = useMemo(() => [X[2 * focus], X[2 * focus + 1]], [X, focus])
  const constraints = useMemo((): Actionability => {
    const immutable = [state.act.fixX1 ? 0 : -1, state.act.fixX2 ? 1 : -1].filter((i) => i >= 0)
    return { immutable, increasing: state.act.upX2 ? [1] : [] }
  }, [state.act.fixX1, state.act.fixX2, state.act.upX2])
  const scale = useMemo(() => medianAbsoluteDeviation(data.x), [data])

  // The model's P(y = 1) over the plane and at the data.
  const field = useMemo(() => {
    if (!net) return null
    const p = sigmoidOf(denseOutput(net, GRID))
    return GY.map((_, r) => Array.from(p.subarray(r * GX.length, (r + 1) * GX.length)))
  }, [net])
  const probability = useMemo(() => (net ? sigmoidOf(denseOutput(net, data.x)) : null), [net, data])
  // Relative KDE density (as FACE's p̃): over the plane for the t_d contour, and at any points.
  const kde = useMemo(() => {
    const top = Math.max(...Array.from(toFlat(multivariateKde(data.x, data.x).logDensity)))
    const at = (P: Tensor) =>
      Float64Array.from(toFlat(multivariateKde(data.x, P).logDensity), (l) => Math.min(1, Math.exp(l - top)))
    const g = at(GRID)
    return { at, grid: GY.map((_, r) => Array.from(g.subarray(r * GX.length, (r + 1) * GX.length))) }
  }, [data])

  const graph = useComputed(
    () =>
      faceGraph(data.x, query, {
        graph: state.face.graph as 'kde' | 'knn' | 'epsilon',
        epsilon: state.face.epsilon,
        k: 8,
        constraints,
      }),
    [data, query, state.face.graph, state.face.epsilon, constraints],
    { mode: 'release' },
  )
  const faceResult = useMemo(
    () =>
      graph.value && probability
        ? faceSearch(graph.value, probability, {
            predictionThreshold: state.face.prediction,
            densityThreshold: state.face.density,
          })
        : null,
    [graph.value, probability, state.face.prediction, state.face.density],
  )
  const gradientMethods = useComputed(
    () => {
      if (!net) return null
      const logit = denseFunction(net)
      // In logit space, where a saturated network still has gradients: target logit(t_p).
      const tp = state.face.prediction
      const wachter = wachterCounterfactual(logit, query, {
        target: Math.log(tp / (1 - tp)),
        tolerance: 0.2,
        scale,
        constraints,
        steps: 80,
        rate: 0.05,
      })
      const dice = diverseCounterfactuals(logit, query, stream(`dice-${focus}`), {
        count: 3,
        steps: 300,
        scale,
        constraints,
      })
      return { wachter, dice }
    },
    [net, query, state.face.prediction, scale, constraints, focus],
    { mode: 'release' },
  )
  const gm = gradientMethods.value
  const pts = (rows: number[]) => rows.map((i) => (i === n ? query : [X[2 * i], X[2 * i + 1]]))
  const facePath = faceResult && faceResult.index >= 0 ? pts(faceResult.path) : null
  const wachterPath = gm
    ? Array.from({ length: gm.wachter.path.shape[0] }, (_, t) => toFlat(gm.wachter.path).slice(2 * t, 2 * t + 2))
    : null
  const diceRows = gm
    ? Array.from({ length: 3 }, (_, k) => Array.from(toFlat(gm.dice.counterfactuals).slice(2 * k, 2 * k + 2)))
    : []
  // Densities and probabilities at each method's end point.
  const ends = useMemo(() => {
    if (!probability || !net) return null
    const list: { name: string; at: number[] }[] = []
    if (gm) list.push({ name: 'Wachter', at: Array.from(gm.wachter.counterfactual) })
    if (gm) diceRows.forEach((r, k) => list.push({ name: `DiCE ${k + 1}`, at: r }))
    if (faceResult?.counterfactual) list.push({ name: 'FACE', at: Array.from(faceResult.counterfactual) })
    if (list.length === 0) return []
    const P = fromData(Float64Array.from(list.flatMap((e) => e.at)), [list.length, 2])
    const dens = kde.at(P)
    const prob = sigmoidOf(denseOutput(net, P))
    return list.map((e, k) => ({
      ...e,
      density: dens[k],
      probability: prob[k],
      distance: Math.hypot(e.at[0] - query[0], e.at[1] - query[1]),
    }))
  }, [gm, faceResult, probability, net, kde, query]) // eslint-disable-line react-hooks/exhaustive-deps

  const ax = useAxis({ label: 'x₁', range: XR, nice: false })
  const ay = useAxis({ label: 'x₂', range: YR, nice: false, equal: ax })
  const select = (p: [number, number]) => {
    let best = 0
    let bd = Infinity
    for (let i = 0; i < n; i++) {
      const dd = (X[2 * i] - p[0]) ** 2 + (X[2 * i + 1] - p[1]) ** 2
      if (dd < bd) {
        bd = dd
        best = i
      }
    }
    pin.toggle(best)
  }
  const edgeSegments = useMemo(() => {
    const g = graph.value
    if (!g || !state.reveal.edges) return []
    const P = (i: number) => (i === n ? query : [X[2 * i], X[2 * i + 1]])
    return g.edges
      .filter(([a, b]) => a < b || g.edges.length < 4000)
      .map(([a, b]) => ({ from: P(a) as [number, number], to: P(b) as [number, number] }))
  }, [graph.value, state.reveal.edges, X, n, query])
  const show = state.reveal
  return (
    <>
      <Figure
        title="Shortest change against feasible path"
        purpose="Wachter et al. and DiCE find the nearest points the model classifies otherwise, which can lie where no data are; FACE only moves along short edges between data points, weighted to prefer dense regions, so its counterfactual is a real, reachable example."
        state={state}
        defaultSize="XL"
        controls={
          <TrainRow
            label="5 · train"
            trained={trained !== null}
            stale={stale}
            mlp={mlp}
            onTrain={() => setTrained({ ...current })}
          />
        }
        readouts={{
          [`query: row ${focus}${pin.pinned === null ? ' (click a point to pin another)' : ''}`]: (
            <>
              <Readout label="x" value={`(${fmt(query[0])}, ${fmt(query[1])})`} />
              <Readout label="P(y = 1) at x" value={probability ? fmt(probability[focus]) : '—'} />
              <Readout label="FACE candidates" value={faceResult ? String(faceResult.candidates.length) : '—'} />
              <Readout label="FACE path cost" value={faceResult ? fmt(faceResult.cost) : '—'} />
            </>
          ),
          'counterfactuals: P(y = 1) · distance · relative density': (
            <>
              {(ends ?? []).map((e) => (
                <Readout
                  key={e.name}
                  label={e.name}
                  value={`${fmt(e.probability, 2)} · ${fmt(e.distance, 2)} · ${fmt(e.density, 2)}`}
                />
              ))}
              {faceResult && faceResult.index < 0 && <Readout label="FACE" value="no reachable candidate" />}
            </>
          ),
        }}
        caption={`Data: aifn feasibilityTask (seeded), ${n} points: class 0 (left blob) and class 1 (right blob) joined by a corridor of data along a lower arc; the space between the blobs is empty. Press Train to fit a 2 → ${shown.width} → ${shown.width} → 1 tanh MLP by Adam in the worker; the background is its P(y = 1) with the 0.5 boundary in ink, and the thin contour is the relative KDE density at t_d (FACE's candidates lie inside it). Click a point to make it the query (again or Escape to unpin). Wachter's search (its iterates) minimises λ(logit − logit t_p)² plus the L1 distance in units of each feature's median absolute deviation, raising λ until the logit is within 0.2 of its target (on the logit, so a saturated network still has a gradient). DiCE finds three counterfactuals at once, trading validity, proximity and a determinantal diversity term. FACE builds the chosen graph over the data (an edge only where a step is no longer than ε and allowed by the actionability switches), weighs each edge by −log(relative density) × length, and takes the cheapest path by Dijkstra's algorithm to a point with P ≥ t_p and density ≥ t_d. Drag t_d to push FACE's end point into denser data; turn on "x₂ may only rise" to forbid the corridor's dip. The readouts give each end point's P(y = 1), its distance from x and its relative density: the gradient methods stop in the empty gap.`}
      >
        <Plot x={ax} y={ay} onPlotClick={select}>
          {field && <Raster x={GX} y={GY} z={field} range={[0, 1]} valueLabel="P(y = 1)" fillOpacity={0.35} boundary />}
          <Contours x={GX} y={GY} z={kde.grid} levels={[Math.max(0.01, state.face.density)]} labels={false} muted />
          {edgeSegments.length > 0 && <Segments segments={edgeSegments} muted width={0.6} />}
          <Points
            name="data"
            x={Array.from({ length: n }, (_, i) => X[2 * i])}
            y={Array.from({ length: n }, (_, i) => X[2 * i + 1])}
            group={y}
            groupNames={['y = 0', 'y = 1']}
            thin
          />
          {show.wachter && wachterPath && (
            <Curve
              name="Wachter"
              x={wachterPath.map((p) => p[0])}
              y={wachterPath.map((p) => p[1])}
              slot={3}
              width={2}
              stale={gradientMethods.stale}
            />
          )}
          {show.wachter && gm && (
            <Points
              name="Wachter"
              x={[gm.wachter.counterfactual[0]]}
              y={[gm.wachter.counterfactual[1]]}
              slot={3}
              size={11}
            />
          )}
          {show.dice && gm && (
            <Segments
              name="DiCE"
              segments={diceRows.map((r) => ({ from: query as [number, number], to: r as [number, number] }))}
              slot={4}
              dashed
              width={1.5}
            />
          )}
          {show.dice && gm && (
            <Points
              name="DiCE"
              x={diceRows.map((r) => r[0])}
              y={diceRows.map((r) => r[1])}
              slot={4}
              size={10}
              shape={2}
            />
          )}
          {show.face && facePath && (
            <Curve
              name="FACE"
              x={facePath.map((p) => p[0])}
              y={facePath.map((p) => p[1])}
              slot={5}
              width={2.5}
              showPoints
              stale={graph.stale}
            />
          )}
          <Points name="query x" x={[query[0]]} y={[query[1]]} emphasis size={13} />
        </Plot>
      </Figure>
      <GrowingSpheresFigure net={net} query={query} focus={focus} X={X} />
    </>
  )
}

// ── Growing Spheres ──────────────────────────────────────────────────────────────────────────────────────────────────

function GrowingSpheresFigure({
  net,
  query,
  focus,
  X,
}: {
  net: ReturnType<typeof useTrainedMlp>['net']
  query: number[]
  focus: number
  X: Float64Array
}) {
  const state = useFigureState({
    search: row('1 · search', {
      radius: slider(0.1, 3, 0.4, { step: 0.05, label: 'first radius η' }),
      samples: int(300, { ge: 20, le: 5000, suggestions: [100, 300, 1000], label: 'samples per shell' }),
    }),
  })
  const { radius, samples } = state.search
  const result = useComputed(
    () => {
      if (!net) return null
      const predict = (Z: Tensor) => Float64Array.from(denseOutput(net, Z), (v) => (v > 0 ? 1 : 0))
      return growingSpheres(predict, query, stream(`gs-${focus}`), { radius, samples })
    },
    [net, query, focus, radius, samples],
    { mode: 'release' },
  )
  const r = result.value
  const [at, setAt] = useState(0)
  const layers = r?.layers ?? []
  const step = Math.min(at, Math.max(0, layers.length - 1))
  const circle = (rad: number) => {
    const t = Array.from({ length: 121 }, (_, k) => (2 * Math.PI * k) / 120)
    return { x: t.map((a) => query[0] + rad * Math.cos(a)), y: t.map((a) => query[1] + rad * Math.sin(a)) }
  }
  const ax = useAxis({ label: 'x₁', range: XR, nice: false })
  const ay = useAxis({ label: 'x₂', range: YR, nice: false, equal: ax })
  const last = layers[step]
  const finalStep = step === layers.length - 1
  const S = r ? toFlat(r.samples) : []
  return (
    <Figure
      title="Growing Spheres: shells until an enemy"
      purpose="Growing Spheres samples uniformly in a ball around x, halving it while it already holds points of the other class, then in ever larger spherical shells until one does; the nearest such enemy, with its smallest changes undone while the class stays changed, is the counterfactual."
      state={state}
      defaultSize="L"
      controls={
        <ControlRow label="2 · shells">
          <Player
            label="shell"
            value={step}
            onChange={setAt}
            count={Math.max(1, layers.length)}
            format={(p) => `shell ${p + 1}`}
          />
        </ControlRow>
      }
      readouts={{
        search: (
          <>
            <Readout label="shells" value={String(layers.length)} />
            <Readout label="model evaluations" value={r ? String(r.evaluations) : '—'} />
            <Readout label="this shell" value={last ? `${fmt(last.inner, 2)} < r ≤ ${fmt(last.outer, 2)}` : '—'} />
            <Readout label="enemies in it" value={last ? String(last.enemies) : '—'} />
          </>
        ),
        result: (
          <>
            <Readout label="enemy" value={r?.enemy ? `(${fmt(r.enemy[0])}, ${fmt(r.enemy[1])})` : '—'} />
            <Readout label="sparse enemy" value={r?.sparse ? `(${fmt(r.sparse[0])}, ${fmt(r.sparse[1])})` : '—'} />
          </>
        ),
      }}
      caption={`Uses the model and the query pinned above (train it there first). Each shell's outer circle is drawn up to the shell the player shows; the first rounds halve the ball while it still holds enemies (none here unless x sits near the boundary), then shells of width η grow until one holds an enemy. The final shell's ${samples} samples are shown, coloured by the model's class; the ink cross is the nearest enemy and the diamond its sparse version, which resets a feature to x's value whenever the class stays changed. Like Wachter's search, it ignores where the data (grey) lie: its enemy sits just past the boundary, often in empty space.`}
    >
      <Plot x={ax} y={ay}>
        <Points
          name="data"
          x={Array.from({ length: X.length / 2 }, (_, i) => X[2 * i])}
          y={Array.from({ length: X.length / 2 }, (_, i) => X[2 * i + 1])}
          muted
          thin
        />
        {layers.slice(0, step + 1).map((l, k) => {
          const c = circle(l.outer)
          return <Curve key={k} name="shells" x={c.x} y={c.y} muted thin silent />
        })}
        {finalStep && r && (
          <Points
            name="last shell's samples"
            x={Array.from({ length: S.length / 2 }, (_, i) => S[2 * i])}
            y={Array.from({ length: S.length / 2 }, (_, i) => S[2 * i + 1])}
            group={net ? Array.from(denseOutput(net, r.samples), (v) => (v > 0 ? 1 : 0)) : null}
            groupNames={['y = 0', 'y = 1']}
            thin
          />
        )}
        <Points name="query x" x={[query[0]]} y={[query[1]]} emphasis size={13} />
        {finalStep && r?.enemy && (
          <Points name="enemy" x={[r.enemy[0]]} y={[r.enemy[1]]} emphasis shape={3} size={12} />
        )}
        {finalStep && r?.sparse && (
          <Points name="sparse enemy" x={[r.sparse[0]]} y={[r.sparse[1]]} emphasis shape={2} size={12} />
        )}
      </Plot>
    </Figure>
  )
}
