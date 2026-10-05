/**
 * Showcase: active learning with label proportions (Poyiadzis, Santos-Rodriguez and Twomey 2019). The learner holds two
 * small bags with known proportions and asks an LLP-oracle about bags it builds from a pool; the oracle answers with
 * the bag's class proportion only. Query strategies, the runs and the curves are
 * `aifn-methods/learning/weak-supervision`'s (`activeProportionsRun`, `activeProportionsCurves`); the lab draws them.
 */
import { useMemo, useState } from 'react'
import { stream } from 'aifn/foundation/random'
import { toFlat, toRows, type Tensor } from 'aifn/foundation/tensor'
import { halfKernel, xor } from 'aifn-methods/data/synthetic'
import {
  activeProportionsProblem,
  type ActiveCurves,
  type ActiveProportionsState,
  type ActiveStrategy,
} from 'aifn-methods/learning/weak-supervision'
import { Player } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { call, choice, float, int, row, useFigureState, type Task } from '@lab/state'
import { formatValue, TrainControls, useTrainedRun } from '@lab/views'
import { Area, Curve, Plot, Plots, Points, Readout, useAxis } from '@lab/viz'

const f3 = (v: number) => (Number.isFinite(v) ? formatValue(Number(v.toPrecision(3))) : '—')
const STRATEGY_NAMES: Record<ActiveStrategy, string> = {
  'us-lp': 'US-LP (k most uncertain, proportion)',
  'us-mass': 'US-Mass (most uncertain + its neighbours)',
  'us-exact': 'US-Exact (k most uncertain, true labels)',
  random: 'Random bag (proportion)',
}
const STRATEGY_SLOT: Record<ActiveStrategy, number> = { 'us-lp': 0, 'us-mass': 1, 'us-exact': 2, random: 3 }
const ROLES = ['starting bag (0.75 class 1)', 'starting bag (0.25 class 1)', 'queried', 'pool', 'test']

type Dataset = 'xor' | 'half-kernel'
const dataOf = (kind: Dataset, n: number, key: string) =>
  kind === 'xor'
    ? xor(stream(key), { kind: 'gaussian', sd: 0.5, n: [Math.floor(n / 2), n - Math.floor(n / 2)] })
    : halfKernel(stream(key), { n })
const dataTask = (kind: Dataset, n: number, key: string) =>
  kind === 'xor'
    ? call('applied/data/synthetic/xor', call('foundation/random/stream', key), {
        kind: 'gaussian',
        sd: 0.5,
        n: [Math.floor(n / 2), n - Math.floor(n / 2)],
      })
    : call('applied/data/synthetic/halfKernel', call('foundation/random/stream', key), { n })

const dataFields = () =>
  row('1 · data', {
    dataset: choice(
      [
        { value: 'half-kernel', label: 'half-kernel' },
        { value: 'xor', label: 'Gaussian XOR (overlapping, sd 0.5)' },
      ],
      'half-kernel',
      { label: 'dataset' },
    ),
    n: int(200, { ge: 60, le: 400, suggestions: [150, 200, 300], label: 'points' }),
    gamma: float(4, { gt: 0, scale: 'log10', suggestions: [1, 4, 16], label: 'γ of the graph' }),
    seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
  })

type RunSettings = {
  dataset: Dataset
  n: number
  gamma: number
  seed: number
  strategy: ActiveStrategy
  bagSize: number
  queries: number
}

/** The data and the active-learning problem that a run's dataset, size and seed fix. */
function makeProblem(dataset: Dataset, n: number, seed: number) {
  const data = dataOf(dataset, n, `lab/active/${dataset}/${seed}`)
  const problem = activeProportionsProblem(stream(`lab/active/split/${seed}`), {
    x: data.x as Tensor,
    y: data.y as Tensor,
  })
  return { data, problem }
}

export function ActiveRunFigure() {
  const state = useFigureState({
    data: dataFields(),
    query: row('2 · queries', {
      strategy: choice(
        (Object.keys(STRATEGY_NAMES) as ActiveStrategy[]).map((value) => ({ value, label: STRATEGY_NAMES[value] })),
        'us-lp',
        { label: 'strategy' },
      ),
      bagSize: int(10, { ge: 1, le: 40, suggestions: [1, 5, 10, 20], label: 'points per query k' }),
      queries: int(6, { ge: 1, le: 30, suggestions: [4, 6, 10], label: 'queries' }),
    }),
  })
  const settings: RunSettings = {
    dataset: state.data.dataset as Dataset,
    n: state.data.n,
    gamma: state.data.gamma,
    seed: state.data.seed,
    strategy: state.query.strategy as ActiveStrategy,
    bagSize: state.query.bagSize,
    queries: state.query.queries,
  }
  const trained = useTrainedRun(settings, (s): Task<ActiveProportionsState[]> => {
    const { problem } = makeProblem(s.dataset, s.n, s.seed)
    return call<ActiveProportionsState[]>('applied/learning/weak-supervision/activeProportionsRun', problem, {
      strategy: s.strategy,
      bagSize: s.bagSize,
      queries: s.queries,
      gamma: s.gamma,
      seed: s.seed,
    })
  })
  const shown = trained.trained ?? settings
  const { dataset: shownDataset, n: shownN, seed: shownSeed } = shown
  const { data, problem } = useMemo(
    () => makeProblem(shownDataset, shownN, shownSeed),
    [shownDataset, shownN, shownSeed],
  )
  const rows = useMemo(() => toRows(data.x as Tensor) as number[][], [data])
  const truth = useMemo(() => Array.from(toFlat(data.y as Tensor)), [data])
  const states = trained.run.value ?? []
  const [picked, setPicked] = useState<{ run: unknown; q: number } | null>(null)
  const q = Math.min(picked && picked.run === trained.trained ? picked.q : 0, Math.max(0, states.length - 1))
  const s = states[q]
  const test = useMemo(() => new Set(Array.from(problem.test)), [problem])
  const role = (i: number) => {
    const b = s ? s.bags[i] : problem.bags[i]
    if (b === 0 || b === 1) return b
    if (b >= 2) return 2
    return test.has(i) ? 4 : 3
  }
  const labels = s ? Array.from(s.labels) : truth.map(() => 0)
  const all = rows.map((_, i) => i)
  const current = s?.query ?? []
  const ax = useAxis({ label: 'x₁', key: shown.dataset })
  const ay = useAxis({ label: 'x₂', equal: ax, key: shown.dataset })
  const qx = useAxis({ label: 'queries', range: [0, shown.queries], key: shown.queries, integer: true })
  const qy = useAxis({ label: 'test accuracy', range: [0.4, 1.02] })
  const answerTruth = current.map((i) => truth[i])
  return (
    <Figure
      title="Querying bags from an LLP-oracle, one query at a time"
      purpose="The learner builds a bag of k pool points and the oracle names only its class proportion. Querying the most uncertain points works even though the answer is a proportion: LP-LLP turns it into labels through the graph."
      state={state}
      defaultSize="XL"
      controls={
        <>
          <TrainControls
            run={trained as never}
            progress={states.length / (shown.queries + 1)}
            progressText={`${Math.max(0, states.length - 1)} / ${shown.queries} queries`}
          />
          <ControlRow label="3 · queries so far">
            <Player
              value={q}
              onChange={(v) => setPicked({ run: trained.trained, q: v })}
              count={Math.max(1, states.length)}
              label="query"
            />
          </ControlRow>
        </>
      }
      readouts={
        s ? (
          <>
            <Readout label="queries" value={`${s.t} of ${shown.queries}`} />
            <Readout label="test accuracy" value={f3(s.accuracy)} />
            <Readout label="this query" value={current.length ? `${current.length} points` : 'none yet (step 0)'} />
            <Readout label="oracle’s answer (share of class 1)" value={current.length ? f3(s.answer) : '—'} />
            <Readout label="true labels in the query" value={answerTruth.length ? answerTruth.join(' ') : '—'} />
            {shown.strategy === 'us-mass' && (
              <Readout
                label="seed point uncertainty |f − ½|"
                value={s.seed >= 0 ? f3(states[q - 1]?.uncertainty[s.seed]) : '—'}
              />
            )}
          </>
        ) : null
      }
      caption={
        <>
          aifn <code>activeProportionsRun</code> on {shown.dataset === 'xor' ? 'Gaussian XOR' : 'the half-kernel'} (
          {shown.n} points): <code>activeProportionsProblem</code> holds out 30% for testing and starts from two bags of
          16 with class-1 shares 0.75 and 0.25 (shapes); every other point is in the pool. Each query builds a bag of k
          = {shown.bagSize} pool points by {STRATEGY_NAMES[shown.strategy]}, adds the oracle&apos;s answer as a new bag
          and refits LP-LLP (γ = {shown.gamma}). Colour is LP-LLP&apos;s label; the large ink marks are the points of
          the query shown. Play the queries from the start; the right chart is the test accuracy after each.
        </>
      }
    >
      <Plots cols={2} widths={[3, 2]}>
        <Plot x={ax} y={ay} title={!trained.trained ? 'press Train to query' : `after ${s?.t ?? 0} queries`}>
          <Points
            name="points"
            x={all.map((i) => rows[i][0])}
            y={all.map((i) => rows[i][1])}
            group={labels}
            groupNames={['labelled class 0', 'labelled class 1']}
            shape={all.map(role)}
            shapeNames={ROLES}
            size={8}
          />
          <Points
            name="this query"
            x={current.map((i) => rows[i][0])}
            y={current.map((i) => rows[i][1])}
            emphasis
            size={14}
          />
        </Plot>
        <Plot x={qx} y={qy}>
          <Curve
            name="test accuracy"
            x={states.map((st) => st.t)}
            y={states.map((st) => st.accuracy)}
            slot={STRATEGY_SLOT[shown.strategy]}
            showPoints
          />
          {s && <Points name="shown" x={[s.t]} y={[s.accuracy]} emphasis size={12} />}
        </Plot>
      </Plots>
    </Figure>
  )
}

type CurveSettings = {
  dataset: Dataset
  n: number
  gamma: number
  seed: number
  bagSize: number
  queries: number
  repeats: number
}

export function ActiveCurvesFigure() {
  const state = useFigureState({
    data: dataFields(),
    query: row('2 · experiment', {
      bagSize: int(10, { ge: 1, le: 40, suggestions: [1, 5, 10, 20], label: 'points per query k' }),
      queries: int(4, { ge: 1, le: 20, suggestions: [4, 8], label: 'queries' }),
      repeats: int(5, { ge: 1, le: 40, suggestions: [3, 5, 10], label: 'datasets' }),
    }),
  })
  const settings: CurveSettings = {
    dataset: state.data.dataset as Dataset,
    n: state.data.n,
    gamma: state.data.gamma,
    seed: state.data.seed,
    bagSize: state.query.bagSize,
    queries: state.query.queries,
    repeats: state.query.repeats,
  }
  const trained = useTrainedRun(settings, (s): Task<ActiveCurves> =>
    call<ActiveCurves>('applied/learning/weak-supervision/activeProportionsCurves', {
      datasets: Array.from({ length: s.repeats }, (_, r) =>
        dataTask(s.dataset, s.n, `lab/active-curves/${s.seed}/${r}`),
      ),
      bagSize: s.bagSize,
      queries: s.queries,
      gamma: s.gamma,
      seed: s.seed,
    }),
  )
  const r = trained.run.value
  const shown = trained.trained ?? settings
  const qx = useAxis({ label: 'queries', range: [0, shown.queries], key: shown.queries, integer: true })
  const qy = useAxis({ label: 'test accuracy (mean over datasets)', hold: 'union', key: trained.trained })
  const xs = Array.from({ length: shown.queries + 1 }, (_, i) => i)
  return (
    <Figure
      title="Accuracy against queries for every query strategy"
      purpose="The paper's experiment: the same start, four ways to build each queried bag. The exact oracle's labels set the ceiling; how close the proportion answers of US-LP and US-Mass come to it, and whether they beat random bags, depends on the data and on k."
      state={state}
      defaultSize="L"
      controls={
        <TrainControls
          run={trained as never}
          progress={(r?.done ?? 0) / Math.max(1, r?.total ?? 1)}
          progressText={`${r?.done ?? 0} / ${r?.total ?? shown.repeats * 4} runs`}
        />
      }
      readouts={
        r ? (
          <>
            {r.strategies.map((st, k) => (
              <Readout
                key={st}
                label={`${st} after ${shown.queries}`}
                value={`${f3(r.mean[k][shown.queries])} ± ${f3(r.sd[k][shown.queries])}`}
              />
            ))}
          </>
        ) : null
      }
      caption={
        <>
          aifn <code>activeProportionsCurves</code>: {shown.repeats} datasets, each with the paper&apos;s start (two
          bags of 16 at 0.75 and 0.25, 30% test), every strategy run for {shown.queries} queries of k = {shown.bagSize}{' '}
          points from the same start. Lines are the mean test accuracy, bands ± one sd over datasets. With k = 1 every
          strategy that ranks by uncertainty is the same and the oracle&apos;s answer is the label itself.
        </>
      }
    >
      <Plot x={qx} y={qy} title={!trained.trained ? 'press Train to run the experiment' : undefined}>
        {r &&
          r.strategies.flatMap((st, k) => [
            <Area
              key={`${st}-band`}
              name={STRATEGY_NAMES[st]}
              slot={STRATEGY_SLOT[st]}
              x={xs}
              y={xs.map((i) => r.mean[k][i] + r.sd[k][i])}
              base={xs.map((i) => r.mean[k][i] - r.sd[k][i])}
              opacity={0.1}
              line={false}
            />,
            <Curve key={st} name={STRATEGY_NAMES[st]} slot={STRATEGY_SLOT[st]} x={xs} y={r.mean[k]} showPoints />,
          ])}
      </Plot>
    </Figure>
  )
}
