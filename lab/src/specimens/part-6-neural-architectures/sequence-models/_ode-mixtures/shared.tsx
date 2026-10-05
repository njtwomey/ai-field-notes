/**
 * Shared pieces of the stochastic vector field mixture pages: model settings and their run options, plain-array
 * geometry and the checkpoint picker (the training figure is in training-figure.tsx).
 */
import { useState } from 'react'
import type { SvfmLossSettings, SvfmRun, SvfmRunOptions } from 'aifn-methods/neural/ode-mixtures'
import { Slider } from 'aifn-render/controls'
import { choice, float, int, row, setting } from 'aifn-render/state'
import { formatValue } from '@lab/views'
import { Handle, type Vector } from 'aifn-render/viz'

export const f3 = (v: number | undefined) =>
  v !== undefined && Number.isFinite(v) ? formatValue(Number(v.toPrecision(3))) : '—'

/** The lattice of fig. 6 without augmentation: a VF, one stochastic VF, a mixture of VFs, a mixture of SVFs. */
export const MODELS = [
  { value: 'vf', label: 'VF (a neural ODE)' },
  { value: 'svf', label: 'SVF (one stochastic VF)' },
  { value: 'vfm', label: 'VFM (a mixture of VFs)' },
  { value: 'svfm', label: 'SVFM (a mixture of stochastic VFs)' },
] as const
export type ModelKind = (typeof MODELS)[number]['value']
export const isMixture = (m: unknown) => m === 'vfm' || m === 'svfm'

export const SELECTIONS = [
  { value: 'pick-and-stick', label: 'pick and stick' },
  { value: 'forward-filtering', label: 'forward filtering' },
] as const

/** TLoss, VLoss and both (TVLoss): the path regularisers of §2.3.1. */
export const PATH_LOSSES = [
  { value: 'none', label: 'none' },
  { value: 'T', label: 'TLoss' },
  { value: 'V', label: 'VLoss' },
  { value: 'TV', label: 'TVLoss (both)' },
] as const
export type PathLoss = (typeof PATH_LOSSES)[number]['value']

/** The model fields of a run's options. */
export function modelOptions(
  model: ModelKind,
  components: number,
  selection: string,
  augment: boolean,
): Pick<SvfmRunOptions, 'components' | 'stochastic' | 'selection' | 'augment'> {
  return {
    components: isMixture(model) ? components : 1,
    stochastic: model === 'svf' || model === 'svfm',
    selection: selection as SvfmRunOptions['selection'],
    augment: augment ? 1 : 0,
  }
}

/** The loss settings of a path regulariser (TLoss and VLoss are refused with FLoss by the library as well). */
export function pathLosses(path: PathLoss, lambdaT: number, lambdaV = lambdaT, forecast = false): SvfmLossSettings {
  if (forecast) return { forecast: true }
  return {
    transport: path === 'T' || path === 'TV',
    variance: path === 'V' || path === 'TV',
    transportWeight: lambdaT,
    varianceWeight: lambdaV,
  }
}

/** The TLoss / VLoss row: which, and the weight λ of each (§2.3.1). */
export const lossRow = (label = 'path losses') =>
  row(label, {
    path: choice(PATH_LOSSES, 'none', { label: 'TLoss / VLoss' }),
    lambdaT: float(0.1, {
      gt: 0,
      scale: 'log10',
      suggestions: [0.01, 0.1, 1],
      label: 'λ of TLoss',
      when: (v) => v.path === 'T' || v.path === 'TV',
    }),
    lambdaV: float(0.1, {
      gt: 0,
      scale: 'log10',
      suggestions: [0.01, 0.1, 1],
      label: 'λ of VLoss',
      when: (v) => v.path === 'V' || v.path === 'TV',
    }),
  })

const ACTIVATIONS = [
  { value: 'relu', label: 'rectified (the paper)' },
  { value: 'tanh', label: 'tanh' },
  { value: 'softplus', label: 'softplus' },
] as const

/**
 * The π networks (eqs. 2–4): f_{π_t0} for π(t₀), and f_ψ, f_Ψ for forward filtering. π chooses which field and the
 * fields carry the dynamics, so π is kept simple: a constant, linear, or one small hidden layer (≤ 16 units).
 */
export const piRow = () =>
  row('π network', {
    prior: choice(
      [
        { value: 'state', label: 'a function of (h(t₀), t₀) (eq. 2)' },
        { value: 'constant', label: 'a learned constant' },
      ],
      'state',
      { label: 'π(t₀)' },
    ),
    piLayers: int(0, { ge: 0, le: 1, suggestions: [0, 1], label: 'hidden layers (0: linear)' }),
    piHidden: int(8, {
      ge: 1,
      le: 16,
      suggestions: [4, 8, 16],
      label: 'hidden units',
      when: (v) => Number(v.piLayers) > 0,
    }),
    piActivation: choice(ACTIVATIONS, 'relu', { label: 'activation', when: (v) => Number(v.piLayers) > 0 }),
    temperature: float(1, { gt: 0, le: 10, scale: 'log10', suggestions: [0.5, 1, 2], label: 'π(t₀) temperature' }),
    emissions: choice(['learned', 'uniform'], 'learned', { label: 'emissions ψ (filtering, eq. 4)' }),
    transitions: choice(['learned', 'fixed'], 'learned', { label: 'transitions Ψ (filtering, eq. 3)' }),
    stickiness: float(0.9, { gt: 0, lt: 1, suggestions: [0.5, 0.9, 0.99], label: 'P(stay): fixed, or initial' }),
  })

/** The component VFs (figs. 2–3): their networks and the SVF variance heads. */
export const fieldRow = () =>
  row('vector fields', {
    hidden: int(32, { ge: 1, le: 256, suggestions: [16, 32, 64], label: 'hidden units' }),
    layers: int(1, { ge: 1, le: 4, suggestions: [1, 2], label: 'hidden layers' }),
    activation: choice(ACTIVATIONS, 'relu', { label: 'activation' }),
    sharedTrunk: setting(false, { label: 'shared trunk (one output layer per component)' }),
    timeDependent: setting(true, { label: 'time-dependent f(h, t)' }),
    maxVariance: float(0.5, { gt: 0, le: 2, suggestions: [0.1, 0.25, 0.5, 1], label: 'largest SVF variance τ' }),
    varianceBias: float(-5, { ge: -10, le: 5, suggestions: [-5, -2, 0], label: 'variance head initial bias' }),
    learnVariance: setting(true, { label: 'learn the variances (else held)' }),
  })

/** The solver (training: fixed steps through the grid; evaluation: Dormand–Prince) and the grid. */
export const solverRow = (grid: number, shown: number | null) =>
  row('solver', {
    method: choice(
      [
        { value: 'rk4', label: 'RK4' },
        { value: 'euler', label: 'Euler' },
        { value: 'dormand-prince', label: 'Dormand–Prince' },
      ],
      'rk4',
      { label: 'training solver' },
    ),
    stepSize: float(1 / grid, { gt: 0, le: 1, suggestions: [0.05, 0.1, 0.2], label: 'step (fixed-step solvers)' }),
    grid: int(grid, { ge: 1, le: 50, suggestions: [5, 10, 20], label: 'grid intervals T' }),
    rtol: float(1e-4, { gt: 0, le: 0.1, scale: 'log10', suggestions: [1e-3, 1e-4, 1e-6], label: 'NFE rtol' }),
    ...(shown === null
      ? {}
      : { shown: int(shown, { ge: 1, le: 1000, suggestions: [50, 100, 200, 400], label: 'sampled paths shown' }) }),
  })

type Values = Record<string, unknown>

/** Run options from the π, field and solver rows. */
export function architecture(pi: Values, fields: Values, solver: Values | null): Partial<SvfmRunOptions> {
  return {
    hidden: Number(fields.hidden),
    layers: Number(fields.layers),
    activation: fields.activation as SvfmRunOptions['activation'],
    maxVariance: Number(fields.maxVariance),
    ...(solver
      ? {
          grid: Number(solver.grid),
          stepSize: Number(solver.stepSize),
          rtol: Number(solver.rtol),
          ...(solver.shown === undefined ? {} : { shown: Number(solver.shown) }),
        }
      : {}),
    architecture: {
      sharedTrunk: Boolean(fields.sharedTrunk),
      timeDependent: Boolean(fields.timeDependent),
      varianceBias: Number(fields.varianceBias),
      learnVariance: Boolean(fields.learnVariance),
      prior: pi.prior as 'state' | 'constant',
      piHidden: Number(pi.piHidden),
      piLayers: Number(pi.piLayers),
      piActivation: pi.piActivation as SvfmRunOptions['activation'],
      temperature: Number(pi.temperature),
      emissions: pi.emissions as 'learned' | 'uniform',
      transitions: pi.transitions as 'learned' | 'fixed',
      stickiness: Number(pi.stickiness),
      ...(solver ? { method: solver.method as 'rk4' | 'euler' | 'dormand-prince' } : {}),
    },
  }
}

/** The name of a configuration, for titles: "SVFM (K = 4, pick and stick) + TVLoss". */
export function modelName(run: SvfmRun | null, path?: string): string {
  if (!run) return ''
  const kind = run.components > 1 ? (run.stochastic ? 'SVFM' : 'VFM') : run.stochastic ? 'SVF' : 'VF'
  const aug = run.stateDim > run.dim ? 'A-' : ''
  const k = run.components > 1 ? ` (K = ${run.components}, ${run.selection.replace(/-/g, ' ')})` : ''
  return `${aug}${kind}${k}${path && path !== 'none' ? ` + ${path === 'TV' ? 'TVLoss' : `${path}Loss`}` : ''}`
}

/** Arrows of a field on a plane grid (x inner, y outer), scaled to `scale` data units per unit of field, capped. */
export function arrows(field: ArrayLike<number>, axis: ArrayLike<number>, scale: number): Vector[] {
  const out: Vector[] = []
  const cell = Math.abs(axis[1] - axis[0])
  const g = axis.length
  for (let i = 0; i < g; i++)
    for (let j = 0; j < g; j++) {
      const k = i * g + j
      let u = field[2 * k] * scale
      let v = field[2 * k + 1] * scale
      const len = Math.hypot(u, v)
      if (len < 1e-9) continue
      if (len > 0.9 * cell) {
        u *= (0.9 * cell) / len
        v *= (0.9 * cell) / len
      }
      out.push({ from: [axis[j], axis[i]], to: [axis[j] + u, axis[i] + v], head: 5, width: 1, muted: true })
    }
  return out
}

/** The 95th percentile of the arrow lengths of fields [g² × 2]. */
export function typicalLength(fields: readonly ArrayLike<number>[]): number {
  const lengths: number[] = []
  for (const f of fields) for (let i = 0; i < f.length / 2; i++) lengths.push(Math.hypot(f[2 * i], f[2 * i + 1]))
  lengths.sort((p, q) => p - q)
  return lengths[Math.floor(0.95 * (lengths.length - 1))] || 1
}

/** A row-major g × g array as raster rows. */
export const rows = (v: ArrayLike<number>, g: number, map: (x: number) => number = (x) => x) =>
  Array.from({ length: g }, (_, i) => Array.from({ length: g }, (__, j) => map(v[i * g + j])))

/** Every k-th entry of a long curve, with 1-based iterations. */
export function thin(values: Float64Array, max = 600): { x: number[]; y: number[] } {
  const stride = Math.max(1, Math.floor(values.length / max))
  const x: number[] = []
  const y: number[] = []
  for (let i = 0; i < values.length; i += stride) {
    x.push(i)
    y.push(values[i])
  }
  return { x, y }
}

/** A choice that belongs to the run it was made on: a new run starts again from `fallback`. */
export function usePick(run: unknown, fallback = 0) {
  const [picked, setPicked] = useState<{ run: unknown; index: number } | null>(null)
  const index = picked && picked.run === run ? picked.index : fallback
  return [index, (i: number) => setPicked({ run, index: i })] as const
}

/** The checkpoint shown (the latest by default), the training figure's marker and a steppable slider over them. */
export function useCheckpoint(run: SvfmRun | null, runKey: unknown) {
  const shots = run?.checkpoints ?? []
  const [picked, pick] = usePick(runKey, Infinity)
  const index = Math.min(picked, Math.max(0, shots.length - 1))
  const shot = shots[index]
  const pickStep = (step: number) => {
    let best = 0
    shots.forEach((c, i) => {
      if (Math.abs(c.step - step) < Math.abs(shots[best].step - step)) best = i
    })
    pick(best)
  }
  const marker = shot ? <Handle kind="x" at={shot.step} onDrag={pickStep} label={`iteration ${shot.step}`} /> : null
  const slider = (
    <Slider
      label="model shown (training iteration)"
      value={index}
      onChange={(v) => pick(Math.round(v))}
      min={0}
      max={Math.max(1, shots.length - 1)}
      step={1}
      steppable
      disabled={shots.length < 2}
      format={(i) => `iteration ${shots[Math.round(i)]?.step ?? 0}`}
    />
  )
  return { shot, index, marker, slider }
}

const meanOf = (a: ArrayLike<number>) => {
  let s = 0
  for (let i = 0; i < a.length; i++) s += a[i]
  return a.length ? s / a.length : NaN
}
export { meanOf }
