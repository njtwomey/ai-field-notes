/**
 * "Neural ODEs with stochastic vector field mixtures: efficiency": the forward-evaluation analysis of §4.3 (fig. 11),
 * computed here. VF, VF with TVLoss, SVFM and SVFM with TVLoss are trained in the worker on one dataset; every instance's
 * realised path is then solved alone by Dormand–Prince at several tolerances, and its function evaluations (NFE) are
 * set beside the NFE of the whole set solved as one system and beside the variance of its VF along the path.
 */
import { useMemo, useState } from 'react'
import type { NfeStudy, NfeStudyModel, NfeStudyOptions, SvfmRunOptions } from 'aifn-applied/neural/ode-mixtures'
import { Select } from '@lab/controls'
import { Dashboard, DashboardCell, DashboardRow, Figure } from '@lab/layout'
import { call, choice, float, int, row, useFigureState, type Task } from '@lab/state'
import { TrainControls, useTrainedRun } from '@lab/views'
import { Annotation, Bars, Curve, Histogram, Plot, Points, Raster, Readout, useAxis } from '@lab/viz'
import { architecture, f3, fieldRow, meanOf, piRow, SELECTIONS, solverRow } from './shared'

const DATA = [
  { value: 'moons', label: 'moons' },
  { value: 'circles', label: 'nested circles' },
  { value: 'xor', label: 'XOR' },
] as const

type Settings = {
  data: string
  n: number
  components: number
  selection: string
  lambdaT: number
  lambdaV: number
  steps: number
  seed: number
  arch: Partial<SvfmRunOptions>
}

const TOLERANCES = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

function taskOf(s: Settings): Task<NfeStudy> {
  const random = call('foundation/random/stream', s.seed + 1)
  const raw =
    s.data === 'moons'
      ? call('applied/data/synthetic/moons', random, { n: s.n, noise: 0.1 })
      : s.data === 'circles'
        ? call('applied/data/synthetic/circles', random, { n: s.n, noise: 0.05, factor: 0.5 })
        : call('applied/data/synthetic/xor', random, { n: s.n, kind: 'gaussian', sd: 0.35 })
  const tv = { transport: true, variance: true, transportWeight: s.lambdaT, varianceWeight: s.lambdaV }
  const mix = { components: s.components, stochastic: true, selection: s.selection as NfeStudyModel['selection'] }
  const models: NfeStudyModel[] = [
    { label: 'VF' },
    { label: 'VF + TVLoss', losses: tv },
    { label: 'SVFM', ...mix },
    { label: 'SVFM + TVLoss', ...mix, losses: tv },
  ]
  const options: NfeStudyOptions = {
    models,
    training: { ...s.arch, steps: s.steps },
    tolerances: TOLERANCES,
    seed: s.seed,
  }
  return call<NfeStudy>(
    'applied/neural/ode-mixtures/nfeStudy',
    call('applied/neural/ode-mixtures/classificationTask', raw),
    options,
  )
}

export function SvfmEfficiencyShowcase() {
  const state = useFigureState({
    setup: row('1 · data and models', {
      data: choice(DATA, 'moons', { label: 'data' }),
      n: choice([300, 1000], 1000, { label: 'points' }),
      components: choice([2, 4, 8], 4, { label: 'SVFM components K' }),
      selection: choice(SELECTIONS, 'pick-and-stick', { label: 'component selection' }),
      lambdaT: float(0.1, { gt: 0, scale: 'log10', suggestions: [0.01, 0.1, 1], label: 'TVLoss: λ of TLoss' }),
      lambdaV: float(0.1, { gt: 0, scale: 'log10', suggestions: [0.01, 0.1, 1], label: 'TVLoss: λ of VLoss' }),
    }),
    pi: piRow(),
    fields: fieldRow(),
    solver: solverRow(5, null),
    train: row('2 · training', {
      steps: int(300, { ge: 1, suggestions: [150, 300, 500, 1000], label: 'iterations per model' }),
      seed: int(0, { label: 'seed', ge: 0, le: 9999 }),
    }),
  })
  const settings: Settings = {
    data: String(state.setup.data),
    n: Number(state.setup.n),
    components: Number(state.setup.components),
    selection: String(state.setup.selection),
    lambdaT: Number(state.setup.lambdaT),
    lambdaV: Number(state.setup.lambdaV),
    steps: Number(state.train.steps),
    seed: state.train.seed,
    arch: architecture(state.pi, state.fields, state.solver),
  }
  const trained = useTrainedRun(settings, taskOf)
  const study = trained.trained ? trained.run.value : null
  const runKey = trained.trained
  const [tolIndex, setTolIndex] = useState(2)
  const tol = Math.min(tolIndex, TOLERANCES.length - 1)
  const results = useMemo(() => study?.results ?? [], [study])
  const M = study?.labels.length ?? 4
  const progress = study ? (study.current + (study.phase === 'training' ? study.step / study.steps : 1)) / M : 0
  const status = !study
    ? '…'
    : study.phase === 'done'
      ? `${M} models trained and measured`
      : `${study.phase} ${study.labels[study.current]} (${study.phase === 'training' ? `${study.step} / ${study.steps}` : 'NFE'})`
  const pending = !trained.trained ? 'press Run to start' : results.length === 0 ? 'training…' : undefined

  const maxNfe = useMemo(() => {
    let m = 10
    for (const r of results) for (const v of r.perInstance[tol]) m = Math.max(m, v)
    return m
  }, [results, tol])
  const nfeX = useAxis({ label: 'NFE per instance', range: [0, undefined], key: [runKey, tol], hold: 'union' })
  const countY = useAxis({ label: 'instances', range: [0, undefined], key: [runKey, tol], hold: 'union' })
  const tolX = useAxis({ label: 'relative tolerance', log: true, range: [1e-6, 1e-2], key: runKey })
  const nfeY = useAxis({ label: 'NFE', log: true, hold: 'union', key: runKey })
  const varX = useAxis({ label: 'variance of the VF along the path', log: true, hold: 'union', key: runKey })
  const nfeY2 = useAxis({ label: 'NFE per instance', range: [0, undefined], hold: 'union', key: [runKey, tol] })
  const barX = useAxis({ label: 'model', categories: study?.labels ?? ['VF', 'VF + TVLoss', 'SVFM', 'SVFM + TVLoss'] })
  const barY = useAxis({ label: 'NFE', range: [0, undefined], hold: 'union', key: [runKey, tol] })
  const binWidth = Math.max(1, Math.ceil(maxNfe / 40))

  const sweep = useMemo(
    () =>
      results.map((r) => ({
        mean: r.perInstance.map((a) => meanOf(a)),
        batch: r.batch,
      })),
    [results],
  )
  const scatter = useMemo(
    () =>
      results.map((r) => ({
        x: Array.from(r.fieldVariance, (v) => Math.max(v, 1e-8)),
        y: Array.from(r.perInstance[tol]),
      })),
    [results, tol],
  )
  return (
    <>
      <Figure
        title="Work per instance"
        id="work"
        purpose="How many function evaluations each instance needs under each model, against the single number usually reported."
        state={state}
        defaultSize="XL"
        controls={<TrainControls run={trained as never} label="run" progress={progress} progressText={status} />}
        readouts={
          <>
            {results.map((r, i) => (
              <Readout
                key={i}
                label={r.label}
                value={`acc ${f3(r.accuracy)} · NFE mean ${f3(meanOf(r.perInstance[tol]))} · all as one ${r.batch[tol]}`}
              />
            ))}
          </>
        }
        caption={
          <>
            aifn nfeStudy (aifn-applied/neural/ode-mixtures): the four models are trained in turn in the worker (Adam,
            minibatches of 50, grid of 5 intervals; the fields as set below, π linear by default, since π only chooses
            which field and the fields carry the dynamics), then each instance&apos;s realised path (one component draw
            and one field sample, held for the solve) is solved alone by core dormandPrinceRows, which gives every
            instance its own steps, at rtol 10⁻² … 10⁻⁶ (atol = rtol/100). Top left: the NFE of each instance at the
            tolerance chosen below; the dashed lines mark the NFE of all instances solved as one system (one step size
            for every instance, error measured over every coordinate), the number usually reported for a model, which
            the hardest instances set. Top right: the mean NFE per instance (solid) and the NFE of the whole set
            (dashed) against the tolerance. Bottom left: each instance&apos;s NFE against the variance of its VF along
            its path (VLoss for that instance): a constant VF makes the embedded error estimate vanish (eq. 16), so a
            few long steps pass. Bottom right: the mean per instance and the whole-set NFE by model. The absolute counts
            include Dormand–Prince&apos;s start (two evaluations and a short first step), so even a constant field takes
            about 30.
            {study?.error ? ` Run stopped: ${study.error}` : ''}
          </>
        }
      >
        <Dashboard>
          <DashboardRow ratio={1.2}>
            <DashboardCell>
              <Plot x={nfeX} y={countY} title={pending ?? `NFE per instance at rtol ${TOLERANCES[tol]}`}>
                {results.map((r, i) => (
                  <Histogram
                    key={i}
                    name={r.label}
                    values={r.perInstance[tol]}
                    bins={{ width: binWidth }}
                    range={[0, maxNfe + binWidth]}
                    normalize="count"
                    slot={i}
                  />
                ))}
                {results.map((r, i) => (
                  <Annotation key={`b${i}`} x={r.batch[tol]} slot={i} dashed />
                ))}
              </Plot>
            </DashboardCell>
            <DashboardCell>
              <Plot x={tolX} y={nfeY} title="NFE against the tolerance">
                {sweep.map((s, i) => (
                  <Curve key={i} name={results[i].label} x={TOLERANCES} y={s.mean} slot={i} showPoints />
                ))}
                {sweep.map((s, i) => (
                  <Curve
                    key={`b${i}`}
                    name={`${results[i].label} (all as one)`}
                    x={TOLERANCES}
                    y={s.batch}
                    slot={i}
                    dashed
                  />
                ))}
                <Annotation x={TOLERANCES[tol]} dashed />
              </Plot>
            </DashboardCell>
          </DashboardRow>
          <DashboardRow ratio={1}>
            <DashboardCell>
              <Plot x={varX} y={nfeY2} title="NFE against the variance of the VF">
                {scatter.map((s, i) => (
                  <Points key={i} name={results[i].label} x={s.x} y={s.y} slot={i} thin />
                ))}
              </Plot>
            </DashboardCell>
            <DashboardCell>
              <Plot x={barX} y={barY} title="by model (loss setting)">
                {results.length > 0 && (
                  <Bars
                    name="mean per instance"
                    x={results.map((_, i) => i - 0.18)}
                    y={results.map((r) => meanOf(r.perInstance[tol]))}
                    width={0.34}
                    slot={5}
                  />
                )}
                {results.length > 0 && (
                  <Bars
                    name="all as one system"
                    x={results.map((_, i) => i + 0.18)}
                    y={results.map((r) => r.batch[tol])}
                    width={0.34}
                    slot={6}
                  />
                )}
              </Plot>
            </DashboardCell>
          </DashboardRow>
        </Dashboard>
        <Select
          label="tolerance shown (rtol)"
          value={String(tol)}
          onChange={(v) => setTolIndex(Number(v))}
          options={TOLERANCES.map((t, i) => ({ value: String(i), label: String(t) }))}
        />
      </Figure>
      <PairFigure study={study} runKey={runKey} tol={tol} />
    </>
  )
}

/** Fig. 11's right column: one model's NFE against another's, instance by instance, with the savings. */
function PairFigure({ study, runKey, tol }: { study: NfeStudy | null; runKey: unknown; tol: number }) {
  const results = useMemo(() => study?.results ?? [], [study])
  const [pair, setPair] = useState<[number, number]>([0, 3])
  const a = Math.min(pair[0], Math.max(0, results.length - 1))
  const b = Math.min(pair[1], Math.max(0, results.length - 1))
  const grid = useMemo(() => {
    if (results.length < 2 || a === b) return null
    const xa = results[a].perInstance[tol]
    const xb = results[b].perInstance[tol]
    let m = 1
    for (let i = 0; i < xa.length; i++) m = Math.max(m, xa[i], xb[i])
    const bins = 30
    const w = Math.ceil((m + 1) / bins)
    const z = Array.from({ length: bins }, () => new Array<number>(bins).fill(0))
    let saved = 0
    let fewer = 0
    for (let i = 0; i < xa.length; i++) {
      z[Math.min(bins - 1, Math.floor(xb[i] / w))][Math.min(bins - 1, Math.floor(xa[i] / w))]++
      saved += xa[i] - xb[i]
      if (xb[i] < xa[i]) fewer++
    }
    const centres = Array.from({ length: bins }, (_, i) => (i + 0.5) * w)
    return { z: z.map((r) => r.map((v) => (v > 0 ? v : NaN))), centres, top: bins * w, saved, share: fewer / xa.length }
  }, [results, a, b, tol])
  const x = useAxis({
    label: `NFE, ${results[a]?.label ?? 'model A'}`,
    range: [0, grid?.top],
    key: [runKey, a, b, tol],
  })
  const y = useAxis({
    label: `NFE, ${results[b]?.label ?? 'model B'}`,
    range: [0, grid?.top],
    key: [runKey, a, b, tol],
    equal: x,
  })
  const options = (study?.labels ?? []).map((l, i) => ({ value: String(i), label: l }))
  const title = grid
    ? `${results[b].label} saves ${grid.saved} NFEs (${Math.round(100 * grid.share)}% of instances need fewer)`
    : 'choose two trained models'
  return (
    <Figure
      title="Instance by instance"
      id="pairs"
      purpose="One model's work against another's for every instance: which instances get cheaper, and by how much."
      defaultSize="L"
      caption={
        <>
          Each cell counts the instances that took that many evaluations under the model across (x) and the model up
          (y), at the tolerance chosen above; the diagonal is equal work, and cells below it are instances the second
          model solves with fewer evaluations. The title tallies the evaluations saved over all instances and the share
          of instances that need fewer (as in fig. 11, right).
        </>
      }
    >
      <Plot x={x} y={y} title={title}>
        {grid && <Raster x={grid.centres} y={grid.centres} z={grid.z} scale="sequential" valueLabel="instances" />}
        {grid && <Curve name="equal work" x={[0, grid.top]} y={[0, grid.top]} emphasis dashed />}
      </Plot>
      <div className="flex flex-wrap gap-4">
        <Select label="across (x)" value={String(a)} onChange={(v) => setPair([Number(v), b])} options={options} />
        <Select label="up (y)" value={String(b)} onChange={(v) => setPair([a, Number(v)])} options={options} />
      </div>
    </Figure>
  )
}
