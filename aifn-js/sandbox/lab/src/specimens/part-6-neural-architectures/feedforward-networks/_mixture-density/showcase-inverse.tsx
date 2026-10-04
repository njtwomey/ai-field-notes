/**
 * Showcase: Bishop's inverse problem. A squared-error MLP and a mixture density network with K components, trained side
 * by side in the worker (`mixtureDensityRun`) on t ~ U(0, 1) observed through x = t + 0.3 sin(2πt) + ε
 * (`bishopInverse`). Every number drawn comes from the run (curves, checkpoints), from `mdnPredict` at a checkpoint, or
 * from the dataset's truth; the lab computes no model maths.
 */
import { useMemo, useState } from 'react'
import { stream } from 'aifn/foundation/random'
import { fromData, toFlat } from 'aifn/foundation/tensor'
import type { InverseTruth } from 'aifn-methods/data'
import { bishopInverse } from 'aifn-methods/data/synthetic'
import { mdnPredict, type MdnSnapshot } from 'aifn-methods/learning/mixture-density'
import { Player } from '@lab/controls'
import { Figure } from '@lab/layout'
import { choice, int, row, useFigureState } from '@lab/state'
import { formatValue, optimiserField, TrainControls, trainingMethodOf, useTrainedRun } from '@lab/views'
import { Annotation, Curve, Handle, Plot, Plots, Points, Raster, Readout, Rug, useAxis, useScaleColor } from '@lab/viz'
import { TrainingCurves } from './curves'
import { paramsOf, runTask, SLOT, useCheckpoint, useNetworks, type RunSettings } from './shared'

const f3 = (v: number) => (Number.isFinite(v) ? formatValue(Number(v.toPrecision(3))) : '—')
const DATA_SEED = 1
const N = 400
const X_RANGE: [number, number] = [-0.15, 1.15]
const T_RANGE: [number, number] = [-0.1, 1.1]
const G = 111
const grid = (lo: number, hi: number, n: number) => Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1))
const XS = grid(...X_RANGE, G)
const TS = grid(...T_RANGE, G)
const XS_TENSOR = fromData(Float64Array.from(XS), [G, 1])

export function InverseShowcase() {
  const state = useFigureState({
    data: row('1 · data', {
      noise: choice([0.02, 0.05, 0.1], 0.05, { label: 'noise σ on x' }),
    }),
    net: row('2 · networks (same body)', {
      components: int(3, { ge: 1, le: 10, suggestions: [1, 2, 3, 5, 8], label: 'MDN components K' }),
      width: int(20, { ge: 2, le: 64, suggestions: [5, 10, 20, 40], label: 'hidden width (one tanh layer)' }),
    }),
    optimiser: optimiserField({ label: '3 · optimiser (full batch)', stepSize: 0.01 }),
    run: row('4 · run', {
      steps: int(2000, { ge: 1, suggestions: [1000, 2000, 3000], label: 'steps' }),
      seed: int(1, { label: 'seed', ge: 0, le: 9999 }),
    }),
  })
  const settings: RunSettings = {
    generator: 'bishopInverse',
    knobs: { n: N, noise: state.data.noise },
    dataSeed: DATA_SEED,
    components: state.net.components,
    hidden: [state.net.width],
    method: trainingMethodOf(state.optimiser, { clipNorm: 10 }),
    steps: state.run.steps,
    seed: state.run.seed,
  }
  const trained = useTrainedRun(settings, runTask)
  const snap: MdnSnapshot | undefined = trained.run.value ?? undefined
  const shown = trained.trained ?? settings
  const nets = useNetworks(snap)
  const { shots, index, shot, pick, pickStep } = useCheckpoint(snap, trained.trained)

  // The data of the run shown (the same registered call as in the worker), and its truth.
  const noise = shown.knobs.noise
  const data = useMemo(() => bishopInverse(stream(DATA_SEED), { n: N, noise }), [noise])
  const truth = data.meta.truth as InverseTruth
  const points = useMemo(() => ({ x: Array.from(toFlat(data.x)), t: Array.from(toFlat(data.y!)) }), [data])
  const branches = useMemo(() => {
    const t = grid(0, 1, 301)
    return { x: t.map((v) => truth.forward([v])[0]), t }
  }, [truth])
  const trueMean = useMemo(() => Array.from(toFlat(truth.mean(XS_TENSOR))), [truth])

  // The probe: a draggable x.
  const [probe, setProbe] = useState(0.5)
  const x0 = Math.min(X_RANGE[1], Math.max(X_RANGE[0], probe))

  // Both networks on the grid of x at the checkpoint.
  const fit = useMemo(() => {
    if (!nets || !shot) return null
    const mdn = mdnPredict(nets.mdn, paramsOf(shot.mixture), XS_TENSOR)
    const mean = mdnPredict(nets.mean, paramsOf(shot.mean), XS_TENSOR)
    const mix = mdn.mixture!
    // p(t | x) per column, scaled by the column's maximum so every x shows its shape.
    const z = TS.map(() => new Array<number>(G).fill(0))
    for (let j = 0; j < G; j++) {
      const col = TS.map((t) => Math.exp(mix.logDensity(j, [t])))
      const top = Math.max(...col) || 1
      col.forEach((v, i) => (z[i][j] = v / top))
    }
    const K = mix.components
    const means = { x: [] as number[], t: [] as number[], w: [] as number[] }
    // Every third x, so each mark's fill (its weight) shows.
    for (let j = 0; j < G; j += 3)
      for (let k = 0; k < K; k++) {
        const w = mix.weights[j * K + k]
        if (w < 0.03) continue
        means.x.push(XS[j])
        means.t.push(mix.means[j * K + k])
        means.w.push(w)
      }
    return { z, means, mdnMean: Array.from(mdn.mean), netMean: Array.from(mean.mean) }
  }, [nets, shot])

  // The slice at the probe.
  const slice = useMemo(() => {
    const xt = fromData(Float64Array.of(x0), [1, 1])
    const truthDensity = Array.from(
      toFlat(
        truth.logLikelihood(
          fromData(new Float64Array(TS.length).fill(x0), [TS.length, 1]),
          fromData(Float64Array.from(TS)),
        ),
      ),
      Math.exp,
    )
    const solutions = truth.solutions([x0]).map((s) => s.value[0])
    if (!nets || !shot) return { truthDensity, solutions, mdn: null, modes: [], net: NaN, mixture: null }
    const p = mdnPredict(nets.mdn, paramsOf(shot.mixture), xt)
    const mix = p.mixture!
    const all = mix.modes(0)
    // Modes holding a visible share of the peak; a very narrow spurious bump is dropped.
    const modes = all.filter((m) => m.density > 0.05 * all[0].density)
    return {
      truthDensity,
      solutions,
      mdn: TS.map((t) => Math.exp(mix.logDensity(0, [t]))),
      modes,
      net: mdnPredict(nets.mean, paramsOf(shot.mean), xt).mean[0],
      mixture: mix.row(0),
    }
  }, [nets, shot, truth, x0])

  const color = useScaleColor('sequential')
  const meanColours = useMemo(() => fit?.means.w.map((w) => color(w)) ?? [], [fit, color])

  const ax = useAxis({ label: 'x (input)', range: X_RANGE })
  const at = useAxis({ label: 't (target)', range: T_RANGE })
  const tSlice = useAxis({ label: 't', range: T_RANGE })
  const dens = useAxis({ label: `p(t | x = ${x0.toFixed(2)})`, hold: 'union', key: trained.trained })

  const probeHandle = <Handle kind="x" at={x0} onDrag={setProbe} label={`x = ${x0.toFixed(2)}`} />
  const empty = !trained.trained ? 'press Train to start' : !shot ? 'training…' : undefined
  const done = snap?.step ?? 0
  const total = shown.steps
  const h = snap?.history
  const hi = h && shot ? h.step.indexOf(shot.step) : -1

  return (
    <Figure
      title="Bishop's inverse problem: the conditional mean against a mixture density network"
      purpose="Predict t from x where x = t + 0.3 sin 2πt + ε: in the fold three values of t share one x. The squared error fits the conditional mean, which cuts through the gaps; a mixture density network fits p(t | x) itself."
      state={state}
      defaultSize="XL"
      controls={
        <TrainControls run={trained as never} progress={done / total} progressText={`${done} / ${total} steps`} />
      }
      readouts={
        <>
          <Readout label="step" value={shot ? shot.step : '—'} />
          <Readout label="MDN NLL" value={hi >= 0 ? f3(h!.nll[hi]) : '—'} />
          <Readout label="squared-error NLL" value={hi >= 0 ? f3(h!.meanNll[hi]) : '—'} />
          <Readout label="probe x" value={f3(x0)} />
          <Readout label="true branches" value={slice.solutions.map(f3).join(', ') || 'none'} />
          <Readout label="MDN modes" value={slice.modes.map((m) => f3(m.value[0])).join(', ') || '—'} />
          <Readout label="squared-error t̂" value={f3(slice.net)} />
        </>
      }
      caption={
        <>
          aifn <code>mixtureDensityRun</code> on <code>bishopInverse</code> ({N} points, noise σ {noise}): two MLPs with
          one tanh layer of {shown.hidden[0]} units trained by the chosen full-batch optimiser (Adam or L-BFGS) from the
          same seed, one on the squared error (slot {SLOT.mean + 1}) and one on the mixture negative log-likelihood{' '}
          <code>mixtureDensityNll</code> with K = {shown.components} Gaussians (π by softmax, σ by exp plus a floor).
          Left: the data, the true branches of the inverse (ink) and its conditional mean E[t | x] (dashed), with the
          squared-error network, which follows the mean through the gaps between branches. Middle: the MDN&apos;s p(t |
          x), each column scaled to its maximum, with each component&apos;s mean drawn where its weight is above 3%,
          coloured by its weight (pale: switched off, dark: carrying the column) and the MDN&apos;s own mean. Right: the
          slice at the probe, the MDN&apos;s density with its modes against the true p(t | x) (dashed) and the true
          branches (ticks); the squared-error prediction is the vertical line. Drag the probe x on the left or middle
          chart; play the checkpoints from step 0, or drag the step marker on either curve.
        </>
      }
    >
      <Plots cols={3} scale={0.7}>
        <Plot x={ax} y={at} title={empty ?? 'squared error: the conditional mean'}>
          <Points name="data" x={points.x} y={points.t} muted thin />
          <Curve name="true branches" x={branches.x} y={branches.t} emphasis width={1} />
          <Curve name="true E[t | x]" x={XS} y={trueMean} emphasis dashed width={1} />
          {fit && <Curve name="squared-error network" x={XS} y={fit.netMean} slot={SLOT.mean} />}
          {probeHandle}
        </Plot>
        <Plot x={ax} y={at} title={empty ?? `MDN p(t | x), K = ${shown.components}`}>
          {fit && (
            <Raster
              x={XS}
              y={TS}
              z={fit.z}
              scale="sequential"
              range={[0, 1]}
              valueLabel="p(t | x) / max"
              fillOpacity={0.55}
            />
          )}
          <Points name="data" x={points.x} y={points.t} muted thin />
          {fit && (
            <Points
              name="component means (colour: weight)"
              x={fit.means.x}
              y={fit.means.t}
              colors={meanColours}
              size={6}
            />
          )}
          {fit && <Curve name="MDN mean" x={XS} y={fit.mdnMean} slot={SLOT.mdn} dashed />}
          {probeHandle}
        </Plot>
        <Plot x={tSlice} y={dens} title={`the slice at x = ${x0.toFixed(2)}`}>
          <Curve name="true p(t | x)" x={TS} y={slice.truthDensity} emphasis dashed width={1} />
          {slice.mdn && <Curve name="MDN p(t | x)" x={TS} y={slice.mdn} slot={SLOT.mdn} />}
          {slice.modes.length > 0 && (
            <Points
              name="MDN modes"
              x={slice.modes.map((m) => m.value[0])}
              y={slice.modes.map((m) => m.density)}
              slot={SLOT.mdn}
              size={7}
            />
          )}
          <Rug name="true branches" values={slice.solutions} emphasis length={12} />
          {Number.isFinite(slice.net) && <Annotation x={slice.net} slot={SLOT.mean} text="squared error" />}
        </Plot>
      </Plots>
      <TrainingCurves snap={snap} step={shot?.step} onStep={pickStep} runKey={trained.trained} />
      <Player
        label="checkpoint"
        value={index}
        onChange={pick}
        count={Math.max(1, shots.length)}
        format={(i) => `step ${shots[i]?.step ?? 0}`}
      />
    </Figure>
  )
}
