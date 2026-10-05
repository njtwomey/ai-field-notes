/**
 * Showcase: label propagation for learning with label proportions (Poyiadzi, Santos-Rodriguez and Twomey 2018). Points
 * are split into bags that reveal only their class proportions; LP-LLP starts every point at its bag's proportion,
 * propagates the scores over a similarity graph and projects them back onto the bags' class masses, step by step. The
 * graph, the steps, the bags, the baselines and the comparison are `aifn-methods/learning/weak-supervision`'s; the lab
 * draws them.
 */
import { useMemo, useState } from 'react'
import { stream } from 'aifn/foundation/random'
import { dense, toFlat, toRows, type Tensor } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import { kNearestNeighbourGraph } from 'aifn/graph'
import { normaliseScores } from 'aifn/graph/propagation'
import { accuracy } from 'aifn/learning/metrics'
import { convexHull } from 'aifn/numerics/geometry'
import { halfKernel, xor } from 'aifn-methods/data/synthetic'
import {
  bagProportionsOf,
  bagsByProportion,
  lpllpGammaSearch,
  lpllpSteps,
  type LlpComparison,
  type LlpMethod,
} from 'aifn-methods/learning/weak-supervision'
import { Player } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { call, choice, float, int, pinField, row, toggle, useFigureState, usePinned, when, type Task } from '@lab/state'
import { formatValue, TrainControls, useTrainedRun } from '@lab/views'
import { Area, Bars, Curve, Plot, Plots, Points, Readout, Segments, useAxis, useScaleColor } from '@lab/viz'

const f3 = (v: number) => (Number.isFinite(v) ? formatValue(Number(v.toPrecision(3))) : '—')
const CLASSES = ['class 0', 'class 1']
/** The paper's three-bag configurations (proportion of class 1 per bag). */
const CONFIGS: Record<string, number[]> = { A: [0.6, 0.4, 0.5], B: [0.85, 0.25, 0.4] }

type Dataset = 'xor' | 'half-kernel'

function makeData(kind: Dataset, n: number, seed: number) {
  const s = stream(`lab/lpllp/${kind}/${seed}`)
  return kind === 'xor'
    ? xor(s, { kind: 'gaussian', sd: 0.2, n: [Math.floor(n / 2), n - Math.floor(n / 2)] })
    : halfKernel(s, { n })
}

/** Each bag's requested proportions: a paper configuration, or bags alternating purity p and 1 − p. */
function requested(mode: string, bags: number, purity: number): number[][] {
  const pi = CONFIGS[mode] ?? Array.from({ length: bags }, (_, k) => (k % 2 === 0 ? purity : 1 - purity))
  return pi.map((p) => [1 - p, p])
}

/** No points moved: one constant, so the bags derived from it keep their identity across renders. */
const NO_MOVES: Record<number, number> = {}

export function LpllpShowcase() {
  const state = useFigureState({
    data: row('1 · data', {
      dataset: choice(
        [
          { value: 'xor', label: 'Gaussian XOR (paper Table 1)' },
          { value: 'half-kernel', label: 'half-kernel (Table 2)' },
        ],
        'xor',
        { label: 'dataset' },
      ),
      n: int(150, { ge: 12, le: 400, suggestions: [120, 180, 300], label: 'points' }),
      seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
    }),
    bags: row('2 · bags', {
      mode: choice(
        [
          { value: 'B', label: 'paper B: 0.85, 0.25, 0.40' },
          { value: 'A', label: 'paper A: 0.60, 0.40, 0.50' },
          { value: 'purity', label: 'purity p, alternating' },
        ],
        'B',
        { label: 'proportions' },
      ),
      count: int(6, { ge: 2, le: 40, suggestions: [3, 6, 12], label: 'bags', when: when('mode', 'purity') }),
      purity: float(0.8, {
        ge: 0.5,
        le: 1,
        suggestions: [0.6, 0.75, 0.9, 1],
        label: 'purity p',
        when: when('mode', 'purity'),
      }),
    }),
    graph: row('3 · graph and propagation', {
      gamma: float(4, { gt: 0, scale: 'log10', suggestions: [0.5, 2, 4, 16, 64], label: 'γ in exp(−γ‖xᵢ − xⱼ‖²)' }),
      neighbours: int(0, { ge: 0, le: 50, suggestions: [0, 5, 10], label: 'neighbours k (0: full graph)' }),
      alpha: float(0.5, { gt: 0, lt: 1, suggestions: [0.25, 0.5, 0.9], label: 'α' }),
      steps: int(60, { ge: 1, le: 2000, suggestions: [30, 60, 200], label: 'steps' }),
    }),
    view: row('4 · show', {
      colour: choice(
        [
          { value: 'label', label: 'inferred label' },
          { value: 'score', label: 'score f = F_{i,1}' },
        ],
        'label',
        { label: 'colour by' },
      ),
      edges: toggle(true, 'graph edges'),
    }),
    pin: pinField(),
  })
  const { dataset, n, seed } = state.data
  const { mode, count, purity } = state.bags
  const { gamma, neighbours, alpha, steps } = state.graph
  const data = useMemo(() => makeData(dataset as Dataset, n, seed), [dataset, n, seed])
  const truth = useMemo(() => Int32Array.from(toFlat(data.y as Tensor)), [data])
  const rows = useMemo(() => toRows(data.x as Tensor) as number[][], [data])
  const wanted = useMemo(() => requested(mode, count, purity), [mode, count, purity])
  const made = useMemo(() => bagsByProportion(stream(`lab/lpllp/bags/${seed}`), truth, wanted), [truth, wanted, seed])
  const B = wanted.length
  // Points moved between bags by clicking (point → bag), reset when the bags are remade.
  const bagKey = `${dataset}/${n}/${seed}/${mode}/${count}/${purity}`
  const [moves, setMoves] = useState<{ key: string; to: Record<number, number> }>({ key: bagKey, to: {} })
  const moved = moves.key === bagKey ? moves.to : NO_MOVES
  const bags = useMemo(() => {
    const b = Int32Array.from(made.bags)
    for (const [i, k] of Object.entries(moved)) b[Number(i)] = k
    return b
  }, [made, moved])
  // The oracle reveals each bag's class proportions for whatever it holds.
  const proportions = useMemo(() => bagProportionsOf(bags, truth, B, 2), [bags, truth, B])
  const P = dense.data(proportions)
  const sizes = useMemo(() => Array.from({ length: B }, (_, k) => bags.filter((b) => b === k).length), [bags, B])

  const run = useMemo(
    () =>
      trace(
        lpllpSteps(data.x as Tensor, bags, proportions, { gamma, neighbours, alpha, tolerance: 0 }),
        undefined,
        steps,
        {
          keep: 'all',
        },
      ),
    [data, bags, proportions, gamma, neighbours, alpha, steps],
  )
  const T = run.steps.length - 1
  const [position, setPosition] = useState(0)
  const at = Math.min(position, 2 * T)
  const t = Math.ceil(at / 2)
  const phase = at === 0 ? 'start: each point at its bag’s proportion' : at % 2 === 1 ? 'propagated' : 'projected'
  const s = run.steps[t]
  const F = at % 2 === 1 ? s.propagated : s.scores
  const f = useMemo(() => {
    const v = dense.data(F)
    return Float64Array.from({ length: n }, (_, i) => v[2 * i + 1])
  }, [F, n])
  const labels = useMemo(() => Array.from(toFlat(normaliseScores(F).labels)), [F])
  const acc = useMemo(
    () => run.steps.map((st) => accuracy(Array.from(truth), Array.from(toFlat(st.labels)))),
    [run, truth],
  )
  const viol = run.steps.map((st, k) => (k === 0 ? 0 : st.violation))
  const wrong = labels.flatMap((l, i) => (l !== truth[i] ? [i] : []))

  // Bag pin: a click on a point pins its bag; with a bag pinned, a click on a point of another bag moves it in.
  const pins = usePinned(state.pin, (v) => state.set('pin', v), { valid: (k) => k < B })
  const pinned = pins.pinned
  const span = useMemo(() => {
    const xs = rows.map((r) => r[0])
    const ys = rows.map((r) => r[1])
    return Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys))
  }, [rows])
  const click = ([x, y]: [number, number]) => {
    let best = -1
    let bestD = Infinity
    rows.forEach((r, i) => {
      const d = (r[0] - x) ** 2 + (r[1] - y) ** 2
      if (d < bestD) {
        bestD = d
        best = i
      }
    })
    if (best < 0 || bestD > (0.04 * span) ** 2) return pins.clear()
    if (pinned === null || bags[best] === pinned) return pins.toggle(bags[best])
    setMoves({ key: bagKey, to: { ...moved, [best]: pinned } })
  }

  // Display helpers: a k-NN graph's edges (k = 3 on the full graph) and the pinned bag's hull.
  const edges = useMemo(() => {
    const g = kNearestNeighbourGraph(data.x as Tensor, Math.min(n - 1, neighbours > 0 ? neighbours : 3))
    return g.edges.map((e) => ({
      from: rows[e.from] as unknown as readonly [number, number],
      to: rows[e.to] as unknown as readonly [number, number],
    }))
  }, [data, rows, n, neighbours])
  const hull = useMemo(() => {
    if (pinned === null) return []
    const members = rows.filter((_, i) => bags[i] === pinned)
    if (members.length < 3) return []
    const h = toRows(convexHull(members).points) as number[][]
    return h.map((p, k) => ({
      from: p as unknown as readonly [number, number],
      to: h[(k + 1) % h.length] as unknown as readonly [number, number],
    }))
  }, [pinned, rows, bags])

  const scoreColour = useScaleColor('diverging')
  const BINS = 10
  const binColours = useMemo(() => Array.from({ length: BINS }, (_, b) => scoreColour((b + 0.5) / BINS)), [scoreColour])

  const ax = useAxis({ label: 'x₁', key: dataset })
  const ay = useAxis({ label: 'x₂', equal: ax, key: dataset })
  const bx = useAxis({ label: 'bag k', range: [-0.6, B - 0.4], key: B, integer: true })
  const by = useAxis({ label: 'share of class 1', range: [0, 1.02] })
  const sx = useAxis({ label: 'step t', range: [0, steps], key: steps })
  const sy = useAxis({ label: 'accuracy · violation', range: [0, 1.02] })
  const massShown = dense.data(at % 2 === 1 ? s.propagatedMass : s.mass)
  const meanScore = Array.from({ length: B }, (_, k) => (sizes[k] > 0 ? massShown[2 * k + 1] / sizes[k] : NaN))
  const hardShare = Array.from({ length: B }, (_, k) => {
    const members = labels.filter((_, i) => bags[i] === k)
    return members.length ? members.reduce((a, l) => a + l, 0) / members.length : NaN
  })
  const bagIdx = Array.from({ length: B }, (_, k) => k)
  const pk = pinned ?? -1
  const pointsShown = colourBy(state.view.colour as string)
  function colourBy(c: string) {
    return c === 'score'
      ? {
          group: Array.from(f, (v) => Math.min(BINS - 1, Math.max(0, Math.floor(v * BINS)))),
          names: binColours.map((_, b) => `f ≈ ${((b + 0.5) / BINS).toFixed(2)}`),
          colors: binColours,
        }
      : { group: labels, names: CLASSES.map((c) => `inferred ${c}`), colors: undefined }
  }

  return (
    <Figure
      title="Label propagation for label proportions, step by step"
      purpose="Bags reveal only their class proportions. LP-LLP starts every point at its bag's proportion, spreads the scores over a similarity graph, and projects them back so every bag keeps its class mass; the labels sort themselves out where the graph separates the classes."
      state={state}
      defaultSize="XL"
      controls={
        <ControlRow label="5 · steps">
          <Player
            value={at}
            onChange={setPosition}
            count={2 * T + 1}
            label="half-step"
            format={(p) => (p === 0 ? 't = 0' : `t = ${Math.ceil(p / 2)} ${p % 2 === 1 ? 'propagate' : 'project'}`)}
          />
        </ControlRow>
      }
      readouts={{
        now: (
          <>
            <Readout label="step" value={`${t} of ${T} (${phase})`} />
            <Readout label="accuracy" value={f3(accuracy(Array.from(truth), labels))} />
            <Readout label="misclassified" value={`${wrong.length} of ${n}`} />
            <Readout label="largest bag violation before projection" value={f3(s.violation)} />
            <Readout label="projection cycles" value={s.projections} />
          </>
        ),
        'pinned bag': (
          <>
            <Readout
              label="bag"
              value={pinned === null ? 'click a point to pin its bag' : `${pinned} (${sizes[pinned]} points)`}
            />
            <Readout label="constraint π_k (class 1)" value={pinned === null ? '—' : f3(P[2 * pinned + 1])} />
            <Readout label="mean score Σf/n_k now" value={pinned === null ? '—' : f3(meanScore[pinned])} />
            <Readout label="share labelled 1" value={pinned === null ? '—' : f3(hardShare[pinned])} />
            <Readout label="points moved" value={Object.keys(moved).length} />
          </>
        ),
      }}
      caption={
        <>
          aifn <code>{dataset === 'xor' ? 'xor' : 'halfKernel'}</code> ({n} points) split by{' '}
          <code>bagsByProportion</code> into {B} bags; <code>lpllpSteps</code> with W = exp(−γ‖xᵢ − xⱼ‖²)
          {neighbours > 0 ? ` on a ${neighbours}-NN graph` : ''}, S = D⁻¹W and the propagation (1 − α)(I − αS)⁻¹ (α ={' '}
          {alpha}). Each step has two halves on the player: propagate (scores average over walks on the graph, so bags
          drift off their proportions) and project (alternating projections between each row on the simplex and each
          bag&apos;s class mass n_k π_k). Left: colour is the inferred label (or the score), marker shape the bag, red
          halos the points labelled wrongly against the hidden truth; edges are a 3-NN view of the graph. Middle: each
          bag&apos;s constraint π_k (bars), its mean score now (dots: off the bar after propagating, on it after
          projecting) and the share of its points labelled 1 (ink ticks). Right: accuracy and the bag violation before
          each projection, by step. Click a point to pin its bag (its hull is drawn); with a bag pinned, click points of
          other bags to move them in; Escape unpins.
        </>
      }
    >
      <Plots cols={3} widths={[3, 2, 2]}>
        <Plot x={ax} y={ay} onPlotClick={click} title={phase}>
          {state.view.edges && <Segments name="graph (3-NN view)" segments={edges} />}
          <Points
            name="labelled wrongly"
            x={wrong.map((i) => rows[i][0])}
            y={wrong.map((i) => rows[i][1])}
            tone="destructive"
            size={16}
          />
          <Points
            name="points"
            x={rows.map((r) => r[0])}
            y={rows.map((r) => r[1])}
            group={pointsShown.group}
            groupNames={pointsShown.names}
            colors={pointsShown.colors}
            shape={Array.from(bags)}
            shapeNames={bagIdx.map((k) => `bag ${k}`)}
            size={9}
          />
          {hull.length > 0 && <Segments name={`bag ${pk} hull`} segments={hull} width={2} />}
        </Plot>
        <Plot
          x={bx}
          y={by}
          title="bags: constraint and estimate"
          onPlotClick={([x]) => pins.toggle(Math.max(0, Math.min(B - 1, Math.round(x))))}
        >
          <Bars name="constraint π_k" x={bagIdx} y={bagIdx.map((k) => P[2 * k + 1])} slot={1} width={0.6} />
          <Points
            name={`mean score (${at % 2 === 1 ? 'propagated' : 'projected'})`}
            x={bagIdx}
            y={meanScore}
            slot={0}
            size={10}
          />
          <Points name="share labelled 1" x={bagIdx} y={hardShare} emphasis size={8} />
          {pinned !== null && <Points name="pinned bag" x={[pinned]} y={[1.0]} emphasis size={12} />}
        </Plot>
        <Plot x={sx} y={sy} title="by step">
          <Curve name="accuracy" x={run.index ? Array.from(run.index) : []} y={acc} slot={2} />
          <Curve
            name="violation before projection"
            x={Array.from(run.index)}
            y={viol.map((v) => Math.min(1, v / Math.max(1, ...sizes)))}
            slot={3}
          />
          <Points name="now" x={[t]} y={[acc[t]]} emphasis />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── γ by smoothness ──────────────────────────────────────────────────────────────────────────────────────────────────

const GAMMAS = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]

export function GammaHeuristicFigure() {
  const state = useFigureState({
    data: row('1 · data', {
      dataset: choice(
        [
          { value: 'xor', label: 'Gaussian XOR' },
          { value: 'half-kernel', label: 'half-kernel' },
        ],
        'half-kernel',
        { label: 'dataset' },
      ),
      n: int(150, { ge: 12, le: 300, suggestions: [120, 180], label: 'points' }),
      config: choice(['A', 'B'], 'B', { label: 'paper bag configuration' }),
      seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
    }),
  })
  const { dataset, n, config, seed } = state.data
  const data = useMemo(() => makeData(dataset as Dataset, n, seed), [dataset, n, seed])
  const truth = useMemo(() => Int32Array.from(toFlat(data.y as Tensor)), [data])
  const bags = useMemo(
    () => bagsByProportion(stream(`lab/lpllp/bags/${seed}`), truth, requested(config, 3, 0)),
    [truth, config, seed],
  )
  const search = useMemo(() => {
    const r = lpllpGammaSearch(data.x as Tensor, bags.bags, bags.proportions, GAMMAS, { maxSteps: 120 })
    return r
  }, [data, bags])
  const accs = useMemo(
    () =>
      GAMMAS.map((gamma) => {
        const st = trace(lpllpSteps(data.x as Tensor, bags.bags, bags.proportions, { gamma }), undefined, 120, {
          keep: 'none',
        }).final
        return accuracy(Array.from(truth), Array.from(toFlat(st.labels)))
      }),
    [data, bags, truth],
  )
  const top = Math.max(...search.scores)
  const gx = useAxis({ label: 'γ', range: [0.2, 80], log: true })
  const gy = useAxis({ label: 'accuracy · smoothness (scaled)', range: [0, 1.02] })
  const chosen = GAMMAS.indexOf(search.gamma)
  return (
    <Figure
      title="Choosing γ by smoothness, without labels"
      purpose="The paper picks the graph's γ with no labels at all: the γ whose converged scores are smoothest on its own graph, Σ_c f̄_cᵀ S f̄_c. Its choice lands where the hidden accuracy is high."
      state={state}
      defaultSize="M"
      readouts={
        <>
          <Readout label="chosen γ" value={f3(search.gamma)} />
          <Readout label="accuracy at chosen γ" value={f3(accs[chosen])} />
          <Readout label="best accuracy over the grid" value={f3(Math.max(...accs))} />
        </>
      }
      caption={
        <>
          aifn <code>lpllpGammaSearch</code> over γ ∈ {'{'}
          {GAMMAS.join(', ')}
          {'}'} on the paper&apos;s configuration {config} (three bags). The smoothness score (scaled to its maximum)
          uses only the bags; the accuracy uses the hidden labels and is shown for judging the choice. Too small a γ
          blurs the classes together (every score drifts to its bag&apos;s average); too large a γ leaves points
          isolated.
        </>
      }
    >
      <Plot x={gx} y={gy}>
        <Curve
          name="smoothness Σ f̄ᵀ S f̄ (scaled)"
          x={GAMMAS}
          y={search.scores.map((v) => (top > 0 ? Math.max(0, v) / top : 0))}
          slot={0}
          showPoints
        />
        <Curve name="accuracy (hidden labels)" x={GAMMAS} y={accs} slot={2} showPoints />
        <Points name="chosen γ" x={[search.gamma]} y={[accs[chosen]]} emphasis size={12} />
      </Plot>
    </Figure>
  )
}

// ── Comparison ───────────────────────────────────────────────────────────────────────────────────────────────────────

const METHOD_NAMES: Record<LlpMethod, string> = {
  lpllp: 'LP-LLP',
  invcal: 'InvCal',
  'alter-svm': 'alter-∝SVM',
  meanmap: 'MeanMap',
  'proportion-loss': 'proportion loss (MLP)',
}
const METHOD_SLOT: Record<LlpMethod, number> = { lpllp: 0, invcal: 1, 'alter-svm': 2, meanmap: 3, 'proportion-loss': 4 }

type Settings = {
  dataset: Dataset
  n: number
  repeats: number
  bagSizes: number[]
  purities: number[]
  methods: LlpMethod[]
  gamma: number
  seed: number
}

const comparisonTask = (s: Settings): Task<LlpComparison> =>
  call<LlpComparison>('applied/learning/weak-supervision/llpComparison', {
    datasets: Array.from({ length: s.repeats }, (_, r) =>
      s.dataset === 'xor'
        ? call('applied/data/synthetic/xor', call('foundation/random/stream', `lab/llp/${s.seed}/${r}`), {
            kind: 'gaussian',
            sd: 0.2,
            n: [Math.floor(s.n / 2), s.n - Math.floor(s.n / 2)],
          })
        : call('applied/data/synthetic/halfKernel', call('foundation/random/stream', `lab/llp/${s.seed}/${r}`), {
            n: s.n,
          }),
    ),
    bagSizes: s.bagSizes,
    purities: s.purities,
    methods: s.methods,
    gamma: s.gamma,
    seed: s.seed,
  })

export function LlpComparisonFigure() {
  const state = useFigureState({
    data: row('1 · data', {
      dataset: choice(
        [
          { value: 'xor', label: 'Gaussian XOR' },
          { value: 'half-kernel', label: 'half-kernel' },
        ],
        'xor',
        { label: 'dataset' },
      ),
      n: int(120, { ge: 30, le: 300, suggestions: [90, 120, 180], label: 'points per dataset' }),
      repeats: int(2, { ge: 1, le: 20, suggestions: [1, 2, 5], label: 'datasets per cell' }),
      seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
    }),
    grid: row('2 · bags', {
      sizes: choice(
        [
          { value: '60,20,8', label: 'bag sizes 60, 20, 8' },
          { value: '40,10,4', label: 'bag sizes 40, 10, 4' },
          { value: '30', label: 'bag size 30' },
        ],
        '60,20,8',
        { label: 'bag sizes' },
      ),
      purities: choice(
        [
          { value: '0.6,0.75,0.9', label: 'purity 0.6, 0.75, 0.9' },
          { value: '0.55,0.65,0.75,0.85,0.95', label: 'purity 0.55 … 0.95' },
        ],
        '0.6,0.75,0.9',
        { label: 'purities' },
      ),
    }),
    methods: row('3 · methods', {
      gamma: float(4, { gt: 0, scale: 'log10', suggestions: [1, 4, 16], label: 'γ (graph and RBF kernels)' }),
      svm: toggle(true, 'alter-∝SVM (slowest)'),
      meanmap: toggle(true, 'MeanMap'),
      loss: toggle(true, 'proportion loss'),
    }),
  })
  const settings: Settings = {
    dataset: state.data.dataset as Dataset,
    n: state.data.n,
    repeats: state.data.repeats,
    bagSizes: (state.grid.sizes as string).split(',').map(Number),
    purities: (state.grid.purities as string).split(',').map(Number),
    methods: [
      'lpllp',
      'invcal',
      ...(state.methods.svm ? (['alter-svm'] as const) : []),
      ...(state.methods.meanmap ? (['meanmap'] as const) : []),
      ...(state.methods.loss ? (['proportion-loss'] as const) : []),
    ],
    gamma: state.methods.gamma,
    seed: state.data.seed,
  }
  const trained = useTrainedRun(settings, comparisonTask)
  const r = trained.run.value
  const shown = trained.trained ?? settings
  const px = useAxis({ label: 'bag purity p', range: [0.5, 1] })
  const py = useAxis({ label: 'accuracy', range: [0.3, 1.02] })
  const done = r?.done ?? 0
  const total = r?.total ?? shown.bagSizes.length * shown.purities.length * shown.repeats
  return (
    <Figure
      title="Accuracy as bags grow and purity falls: LP-LLP against the baselines"
      purpose="Smaller and purer bags carry more information. LP-LLP uses the graph as well as the bags, so it keeps its accuracy where bags are large and mixed; InvCal and MeanMap see the bags only through their means."
      state={state}
      defaultSize="XL"
      controls={
        <TrainControls
          run={trained as never}
          progress={done / Math.max(1, total)}
          progressText={`${done} / ${total} cells`}
          label="run"
        />
      }
      readouts={
        r ? (
          <>
            {r.methods.map((m, mi) => (
              <Readout
                key={m}
                label={`${METHOD_NAMES[m]} mean accuracy`}
                value={f3(
                  r.mean[mi]
                    .flat()
                    .filter(Number.isFinite)
                    .reduce((a, v, _, all) => a + v / all.length, 0),
                )}
              />
            ))}
          </>
        ) : null
      }
      caption={
        <>
          aifn <code>llpComparison</code>: for each bag size and purity p, {shown.repeats} fresh{' '}
          {shown.dataset === 'xor' ? 'Gaussian XOR' : 'half-kernel'} datasets of {shown.n} points are split into bags of
          that size whose class-1 share alternates p and 1 − p (<code>bagsByProportion</code>); every method sees the
          same bags and labels every point (transductive, as the paper). LP-LLP (<code>lpllp</code>, γ = {shown.gamma})
          against InvCal (<code>inverseCalibration</code>, Rüping 2010), alter-∝SVM (<code>alterProportionSvm</code>, Yu
          et al. 2013), MeanMap (<code>meanMap</code>, Quadrianto et al. 2009, random Fourier features) and the
          proportion loss (<code>proportionClassifier</code>, a 16-unit MLP). One panel per bag size; the band is ± one
          sd over datasets. Accuracy 0.5 is chance; a method can also learn the classes swapped when bags are near 0.5.
        </>
      }
    >
      <Plots cols={shown.bagSizes.length}>
        {shown.bagSizes.map((size, b) => (
          <Plot
            key={size}
            x={px}
            y={py}
            legend={b === 0}
            title={!trained.trained ? 'press Train to run' : `bags of ${size}`}
          >
            {r &&
              r.methods.map((m, mi) => {
                const mean = r.mean[mi][b]
                const sd = r.sd[mi][b]
                const ok = shown.purities.map((_, p) => Number.isFinite(mean[p]))
                const xs = shown.purities.filter((_, p) => ok[p])
                return [
                  <Area
                    key={`${m}-band`}
                    name={METHOD_NAMES[m]}
                    slot={METHOD_SLOT[m]}
                    x={xs}
                    y={xs.map((_, k) =>
                      Math.min(1, mean[shown.purities.indexOf(xs[k])] + (sd[shown.purities.indexOf(xs[k])] || 0)),
                    )}
                    base={xs.map((_, k) =>
                      Math.max(0, mean[shown.purities.indexOf(xs[k])] - (sd[shown.purities.indexOf(xs[k])] || 0)),
                    )}
                    opacity={0.1}
                    line={false}
                  />,
                  <Curve
                    key={m}
                    name={METHOD_NAMES[m]}
                    slot={METHOD_SLOT[m]}
                    x={xs}
                    y={xs.map((x) => mean[shown.purities.indexOf(x)])}
                    showPoints
                  />,
                ]
              })}
            <Curve name="chance" x={[0.5, 1]} y={[0.5, 0.5]} muted dashed thin />
          </Plot>
        ))}
      </Plots>
    </Figure>
  )
}
