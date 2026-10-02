import { useCallback, useMemo, useState } from 'react'
import {
  classifier,
  logitShift,
  classifierLogits,
  type JemCheckpoint,
  type JemRun,
  type JemRunOptions,
  type LogitShift,
} from 'aifn-applied/generative/energy'
import { add, argmax, fromData, logsumexp, neg, toFlat, unwrap, type Tensor } from 'aifn/foundation/tensor'
import { softmax } from 'aifn/numerics/special'
import { Player } from '@lab/controls'
import { Figure } from '@lab/layout'
import { call, choice, float, int, row, slider, useFigureState, type Task } from '@lab/state'
import { formatValue, TrainControls, useTrainedRun } from '@lab/views'
import { Curve, Handle, Histogram, Plot, Plots, Points, Raster, Readout, useAxis } from '@lab/viz'

const f3 = (v: number) => (Number.isFinite(v) ? formatValue(Number(v.toPrecision(3))) : '—')

/** Labelled datasets of `aifn-applied/data/synthetic` with known class densities; the test set is a second draw. */
const DATASETS = {
  moons: { label: 'two moons', key: 'moons', knobs: { n: 400, noise: 0.1, spacing: 'random' } },
  blobs: {
    label: 'three blobs',
    key: 'blobs',
    knobs: { n: 450, centers: 3, sd: 0.35, layout: 'polygon', separation: 4 },
  },
  rings: { label: 'two rings', key: 'rings', knobs: { n: 400, radii: [0.6, 1.5], noise: 0.12 } },
  four: { label: 'four blobs', key: 'blobs', knobs: { n: 480, centers: 4, sd: 0.3, layout: 'polygon', separation: 4 } },
} as const
type DataKey = keyof typeof DATASETS

type Settings = { data: DataKey; seed: number; options: JemRunOptions }

const jemTask = ({ data, seed, options }: Settings): Task<JemRun> => {
  const d = DATASETS[data]
  const make = (s: number) => call(`applied/data/synthetic/${d.key}`, call('foundation/random/stream', s), d.knobs)
  return call<JemRun>(
    'applied/generative/energy/jemRun',
    make(seed),
    make(seed + 1000),
    call('applied/data/synthetic/annulus', call('foundation/random/stream', seed + 2000), {
      n: 200,
      inner: 3,
      outer: 4,
    }),
    options,
  )
}

const columns = (a: Float64Array) => ({
  x: Array.from({ length: a.length / 2 }, (_, i) => a[2 * i]),
  y: Array.from({ length: a.length / 2 }, (_, i) => a[2 * i + 1]),
})
const rows = (v: ArrayLike<number>, g: number) =>
  Array.from({ length: g }, (_, i) => Array.from({ length: g }, (_, j) => v[i * g + j]))

/** Fields of one model at a checkpoint after the logit shift: decision regions, energy, p(class 0 | x). */
function fieldsOf(shot: JemCheckpoint, K: number, shift: Float64Array) {
  const n = shift.length
  const logits = add(fromData(shot.logits, [n, K]), fromData(shift, [n, 1])) as Tensor
  return {
    regions: toFlat(argmax(logits, -1)),
    energy: toFlat(neg(logsumexp(logits, -1)) as Tensor),
  }
}

export function JemShowcase() {
  const state = useFigureState({
    setup: row('1 · data and model', {
      data: choice(
        (Object.keys(DATASETS) as DataKey[]).map((k) => ({ value: k, label: DATASETS[k].label })),
        'moons',
        { label: 'data' },
      ),
      hidden: choice([32, 64], 64, { label: 'hidden width (2 layers, SiLU)' }),
      steps: int(600, { ge: 1, suggestions: [300, 600, 1000], label: 'updates' }),
      seed: int(1, { label: 'seed', ge: 0, le: 9999 }),
    }),
    sampler: row('2 · Langevin negatives (JEM)', {
      langevinSteps: int(20, { label: 'steps per draw K', ge: 1, le: 100, suggestions: [5, 10, 20, 40] }),
      langevinStepSize: float(0.02, {
        label: 'step size α',
        gt: 0,
        le: 1,
        scale: 'log10',
        suggestions: [0.005, 0.01, 0.02, 0.05],
      }),
      reinitialise: float(0.05, { label: 'restart probability ρ', ge: 0, le: 1, suggestions: [0, 0.05, 0.2] }),
      regularisation: float(0.1, { label: 'energy penalty', ge: 0, le: 10, suggestions: [0, 0.1, 1] }),
    }),
    shift: row('3 · logit shift c(x), added to every logit after training', {
      kind: choice(
        [
          { value: 'none', label: 'none' },
          { value: 'radial', label: 'a‖x‖²' },
          { value: 'tilt', label: 'a·x₁' },
          { value: 'bump', label: 'a·exp(−‖x‖²/2)' },
        ],
        'radial',
        { label: 'c(x)' },
      ),
      amount: slider(-3, 3, 0, { label: 'a', step: 0.1 }),
    }),
    px: slider(-3, 3, 0.5, { step: 0.01, onChart: true }),
    py: slider(-3, 3, 0.25, { step: 0.01, onChart: true }),
  })
  const { setup, sampler } = state
  const settings: Settings = {
    data: setup.data as DataKey,
    seed: setup.seed,
    options: {
      steps: Number(setup.steps),
      hidden: [Number(setup.hidden), Number(setup.hidden)],
      stepSize: 1e-3,
      langevinSteps: sampler.langevinSteps,
      langevinStepSize: sampler.langevinStepSize,
      reinitialise: sampler.reinitialise,
      regularisation: sampler.regularisation,
      seed: setup.seed,
      samples: 150,
      sampleSteps: 40,
    },
  }
  const trained = useTrainedRun(settings, jemTask)
  const result = trained.run.value
  const K = result?.classes ?? 2
  const ceShots = useMemo(() => result?.crossEntropy.checkpoints ?? [], [result])
  const jemShots = useMemo(() => result?.jem.checkpoints ?? [], [result])
  const count = Math.min(ceShots.length, jemShots.length)
  const [picked, setPicked] = useState<{ run: Settings | null; index: number } | null>(null)
  const index = Math.min(picked && picked.run === trained.trained ? picked.index : 0, Math.max(0, count - 1))
  const pick = (i: number) => setPicked({ run: trained.trained, index: i })
  const ce = ceShots[index]
  const jem = jemShots[index]

  const shiftKind = state.shift.kind as LogitShift['kind']
  const shiftAmount = state.shift.amount
  const shift = useMemo((): LogitShift => ({ kind: shiftKind, amount: shiftAmount }), [shiftKind, shiftAmount])
  const g = result?.gridX.length ?? 0
  const grid = useMemo(() => {
    if (!result) return null
    const pts = new Float64Array(2 * g * g)
    for (let i = 0; i < g; i++)
      for (let j = 0; j < g; j++) {
        pts[2 * (i * g + j)] = result.gridX[j]
        pts[2 * (i * g + j) + 1] = result.gridY[i]
      }
    return fromData(pts, [g * g, 2])
  }, [result, g])
  const c = useMemo(() => (grid ? logitShift(shift, grid) : new Float64Array(0)), [grid, shift])
  const ceFields = useMemo(() => (ce && grid ? fieldsOf(ce, K, c) : null), [ce, grid, K, c])
  const jemFields = useMemo(() => (jem && grid ? fieldsOf(jem, K, c) : null), [jem, grid, K, c])
  // One colour range per model for the whole run (from the unshifted final energies), so checkpoints compare.
  const energyRange = useCallback(
    (shots: JemCheckpoint[]): [number, number] => {
      let lo = Infinity
      let hi = -Infinity
      const last = shots.at(-1)
      if (!last) return [0, 1]
      const e = fieldsOf(last, K, new Float64Array(last.logits.length / K)).energy
      for (const v of e) {
        lo = Math.min(lo, v)
        hi = Math.max(hi, v)
      }
      return [lo, hi]
    },
    [K],
  )
  const ceRange = useMemo(() => energyRange(ceShots), [energyRange, ceShots])
  const jemRange = useMemo(() => energyRange(jemShots), [energyRange, jemShots])

  const width = Number(setup.hidden)
  const net = useMemo(() => classifier(2, K, { hidden: [width, width] }), [K, width])
  const probeAt = (shot: JemCheckpoint | undefined) => {
    if (!shot) return null
    const at = fromData(Float64Array.of(state.px, state.py), [1, 2])
    const cx = logitShift(shift, at)[0]
    const logits = add(unwrap(classifierLogits(net, shot.params, at)) as Tensor, cx) as Tensor
    return { p: toFlat(softmax(logits) as Tensor), energy: toFlat(neg(logsumexp(logits, -1)) as Tensor)[0] }
  }
  const ceProbe = probeAt(ce)
  const jemProbe = probeAt(jem)

  const box = result?.box ?? 3
  const x1 = useAxis({ label: 'x₁', range: [-box, box], key: box })
  const x2 = useAxis({ label: 'x₂', range: [-box, box], key: box, equal: x1 })
  const stepAxis = useAxis({ label: 'update', range: [0, result?.steps ?? 1], key: result?.steps })
  const eceAxis = useAxis({ label: 'test ECE', range: [0, undefined], hold: 'union', key: result?.steps })
  const aucAxis = useAxis({ label: 'OOD AUROC of −E(x)', range: [0, 1] })
  const energyAxis = useAxis({ label: 'energy E(x)', hold: 'union', key: result?.steps })
  const densityAxis = useAxis({ label: 'density', range: [0, undefined], hold: 'union', key: result?.steps })

  const data = useMemo(() => (result ? { ...columns(result.data), group: Array.from(result.labels) } : null), [result])
  const names = result?.labelNames ?? []
  const metrics = useMemo(() => {
    const x = ceShots.slice(0, count).map((s) => s.step)
    return {
      x,
      ceEce: ceShots.slice(0, count).map((s) => s.ece),
      jemEce: jemShots.slice(0, count).map((s) => s.ece),
      ceAuc: ceShots.slice(0, count).map((s) => s.oodAuroc),
      jemAuc: jemShots.slice(0, count).map((s) => s.oodAuroc),
    }
  }, [ceShots, jemShots, count])
  // The true energy −log p(x), its colour range capped 12 nats above the minimum (it grows without bound off the data).
  const trueEnergy = useMemo(() => {
    if (!result?.trueLogDensity) return null
    const e = Array.from(result.trueLogDensity, (v) => -v)
    const lo = Math.min(...e)
    return { z: rows(e, g), range: [lo, lo + 12] as [number, number] }
  }, [result, g])
  const probeHandle = <Handle {...state.handle(['px', 'py'], { label: 'probe' })} />
  const marker = ce ? (
    <Handle
      kind="x"
      at={ce.step}
      onDrag={(s) => {
        let best = 0
        for (let i = 0; i < count; i++) if (Math.abs(ceShots[i].step - s) < Math.abs(ceShots[best].step - s)) best = i
        pick(best)
      }}
      label={`update ${ce.step}`}
    />
  ) : null
  const empty = !trained.trained ? 'press Train to start' : !ce ? 'training…' : undefined

  const regionPlot = (title: string, fields: typeof ceFields) => (
    <Plot x={x1} y={x2} title={empty ?? title}>
      {fields && result && (
        <Raster
          x={result.gridX}
          y={result.gridY}
          z={rows(fields.regions, g)}
          scale="categorical"
          fillOpacity={0.22}
          categoryNames={names}
        />
      )}
      {data && <Points name="training points" x={data.x} y={data.y} group={data.group} groupNames={names} thin />}
      {probeHandle}
    </Plot>
  )
  const energyPlot = (
    title: string,
    fields: typeof ceFields,
    shot: JemCheckpoint | undefined,
    range: [number, number],
  ) => (
    <Plot x={x1} y={x2} title={empty ?? title}>
      {fields && result && (
        <Raster
          x={result.gridX}
          y={result.gridY}
          z={rows(fields.energy, g)}
          scale="sequential"
          range={range}
          valueLabel="E(x)"
          fillOpacity={0.85}
        />
      )}
      {data && <Points name="data" x={data.x} y={data.y} muted thin />}
      {shot && <Points name="Langevin samples of p(x)" {...columns(shot.samples)} emphasis size={6} />}
    </Plot>
  )
  const conditionalPlot = (title: string, shot: JemCheckpoint | undefined) => (
    <Plot x={x1} y={x2} title={empty ?? title}>
      {data && <Points name="data" x={data.x} y={data.y} muted thin />}
      {shot?.conditional.map((s, k) => (
        <Points key={k} name={`samples of p(x | ${names[k] ?? k})`} {...columns(s)} slot={k} size={5} />
      ))}
    </Plot>
  )
  const oodPlot = (title: string, shot: JemCheckpoint | undefined) => (
    <Plot x={energyAxis} y={densityAxis} title={empty ?? title}>
      {shot && <Histogram name="test points" values={shot.testEnergy} slot={4} />}
      {shot && <Histogram name="out-of-distribution shell" values={shot.oodEnergy} slot={5} />}
    </Plot>
  )
  const done = result?.done ?? 0
  const total = trained.trained?.options.steps ?? settings.options.steps ?? 1
  return (
    <Figure
      title="Your classifier is secretly an energy-based model"
      purpose="Adding any c(x) to every logit leaves p(y | x) unchanged, so a softmax classifier leaves its density p(x) ∝ Σ_y exp f(x)[y] free; JEM trains that density too, and only then do its energy and Langevin samples follow the data."
      state={state}
      defaultSize="XL"
      controls={
        <TrainControls run={trained as never} progress={done / total} progressText={`${done} / ${total} updates`} />
      }
      readouts={{
        'at the probe': (
          <>
            <Readout label="p(y | x), cross-entropy" value={ceProbe ? Array.from(ceProbe.p, f3).join(' · ') : '—'} />
            <Readout label="p(y | x), JEM" value={jemProbe ? Array.from(jemProbe.p, f3).join(' · ') : '—'} />
            <Readout label="E(x) − c(x), cross-entropy" value={ceProbe ? f3(ceProbe.energy) : '—'} />
            <Readout label="E(x) − c(x), JEM" value={jemProbe ? f3(jemProbe.energy) : '—'} />
          </>
        ),
        [`update ${ce?.step ?? 0}`]: (
          <>
            <Readout label="accuracy CE / JEM" value={ce && jem ? `${f3(ce.accuracy)} / ${f3(jem.accuracy)}` : '—'} />
            <Readout label="test ECE CE / JEM" value={ce && jem ? `${f3(ce.ece)} / ${f3(jem.ece)}` : '—'} />
            <Readout label="OOD AUROC CE / JEM" value={ce && jem ? `${f3(ce.oodAuroc)} / ${f3(jem.oodAuroc)}` : '—'} />
          </>
        ),
      }}
      caption={
        <>
          aifn jemRun: two MLP classifiers (2 → {settings.options.hidden?.[0]} → {settings.options.hidden?.[0]} → K,
          SiLU), from the same initial parameters and minibatches; one trained by softmax cross-entropy, one by JEM
          (cross-entropy plus persistent contrastive divergence on E(x) = −logsumexp f(x), negatives by short-run
          Langevin from a replay buffer). Rows: decision regions; the energy with Langevin samples of p(x) started
          uniform on the box (ink); samples of each p(x | y) ∝ exp f(x)[y]; the energy at test points against points on
          a far shell (3 ≤ ‖x‖ ≤ 4). The logit shift adds c(x) to both models&apos; logits: the regions and p(y | x) at
          the probe do not move while the energy does. The true energy −log p(x) is drawn from the generator&apos;s
          known densities. Drag the probe on the region plots; play the checkpoints or drag the update marker.
        </>
      }
    >
      <Plots cols={3}>
        {regionPlot('cross-entropy: decision regions', ceFields)}
        {regionPlot('JEM: decision regions', jemFields)}
        <Plot x={x1} y={x2} title={empty ?? 'true energy −log p(x)'}>
          {trueEnergy && result && (
            <Raster
              x={result.gridX}
              y={result.gridY}
              z={trueEnergy.z}
              scale="sequential"
              range={trueEnergy.range}
              valueLabel="−log p(x)"
              fillOpacity={0.85}
            />
          )}
          {data && <Points name="data" x={data.x} y={data.y} muted thin />}
        </Plot>
      </Plots>
      <Plots cols={3}>
        {energyPlot('cross-entropy: energy and samples', ceFields, ce, ceRange)}
        {energyPlot('JEM: energy and samples', jemFields, jem, jemRange)}
        <Plot x={stepAxis} y={eceAxis} title="calibration">
          {metrics.x.length > 0 && <Curve name="cross-entropy" x={metrics.x} y={metrics.ceEce} slot={4} showPoints />}
          {metrics.x.length > 0 && <Curve name="JEM" x={metrics.x} y={metrics.jemEce} slot={5} showPoints />}
          {marker}
        </Plot>
      </Plots>
      <Plots cols={3}>
        {conditionalPlot('cross-entropy: samples of p(x | y)', ce)}
        {conditionalPlot('JEM: samples of p(x | y)', jem)}
        <Plot x={stepAxis} y={aucAxis} title="out-of-distribution detection">
          {metrics.x.length > 0 && <Curve name="cross-entropy" x={metrics.x} y={metrics.ceAuc} slot={4} showPoints />}
          {metrics.x.length > 0 && <Curve name="JEM" x={metrics.x} y={metrics.jemAuc} slot={5} showPoints />}
          {marker}
        </Plot>
      </Plots>
      <Plots cols={2}>
        {oodPlot('cross-entropy: E(x) on test and far points', ce)}
        {oodPlot('JEM: E(x) on test and far points', jem)}
      </Plots>
      <Player
        label="checkpoint"
        value={index}
        onChange={pick}
        count={Math.max(1, count)}
        format={(i) => `update ${ceShots[i]?.step ?? 0}`}
      />
    </Figure>
  )
}
