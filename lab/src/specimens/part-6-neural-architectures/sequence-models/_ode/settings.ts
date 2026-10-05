/**
 * The showcase's settings: tasks, datasets, models and presets, and the worker task each setting builds.
 */
import type {
  CnfRun,
  CnfRunOptions,
  LatentOdeRun,
  LatentOdeRunOptions,
  OdeRun,
  OdeRunOptions,
} from 'aifn-methods/neural/ode'
import { call, type Task } from 'aifn-render/state'
import { datasetChoice, type DatasetValue } from '@lab/views'

export type TaskKind = 'classification' | 'regression' | 'density' | 'latent'

export const TASKS = [
  { value: 'classification', label: 'classify 2-d points' },
  { value: 'regression', label: 'regress g(x) = −x (1-d)' },
  { value: 'density', label: 'density (CNF / FFJORD)' },
  { value: 'latent', label: 'irregular trajectories (latent ODE)' },
] as const

/** Classification data: Dupont's disc inside a ring, and registered generators of `aifn-methods/data/synthetic`. */
export const CLASS_DATA = datasetChoice({
  disc: {
    label: 'disc inside a ring (Dupont)',
    n: 400,
    noise: false,
    custom: { task: (random, { n }) => call('applied/neural/ode/discInRing', random, n) },
  },
  circles: { label: 'nested circles', n: 300, noise: 0.04, knobs: { factor: 0.45 } },
  moons: { label: 'two moons', n: 300, noise: 0.08 },
  spirals: { label: 'two spirals', n: 300, noise: 0.03, knobs: { arms: 2 } },
})

/** Density data; the checkerboard keeps one colour of tiles. */
export const DENSITY_DATA = datasetChoice({
  moons: { label: 'two moons', n: 1000, noise: 0.06 },
  rings: { label: 'three rings', n: 1200, noise: 0.08 },
  checkerboard: { label: 'checkerboard', n: 2000, knobs: { tiles: 4 } },
})

export const LATENT_DATA = [
  { value: 'sine', label: 'sines (1-d)' },
  { value: 'spiral', label: 'spirals (2-d)' },
] as const

export const MODELS = [
  { value: 'node', label: 'neural ODE' },
  { value: 'anode', label: 'augmented NODE' },
  { value: 'sonode', label: 'second-order NODE' },
  { value: 'resnet', label: 'ResNet (Euler, untied)' },
] as const
export type ModelKind = (typeof MODELS)[number]['value']

export const GRADIENTS = [
  { value: 'backprop', label: 'backprop through the solver' },
  { value: 'adjoint', label: 'adjoint (solve backwards)' },
] as const

export const SOLVERS = [
  { value: 'euler', label: 'Euler' },
  { value: 'rk4', label: 'RK4' },
  { value: 'dormand-prince', label: 'Dormand–Prince (adaptive)' },
] as const

/** Everything a run depends on, as plain data. */
export type Settings = {
  task: TaskKind
  classData: DatasetValue
  densityData: DatasetValue
  latentData: 'sine' | 'spiral'
  model: ModelKind
  augment: number
  timeDependent: boolean
  gradient: 'backprop' | 'adjoint'
  method: 'euler' | 'rk4' | 'dormand-prince'
  stepSize: number
  rtol: number
  checkpoints: number
  depth: number
  estimator: 'exact' | 'hutchinson'
  probe: 'rademacher' | 'gaussian'
  steps: number
  learningRate: number
  kinetic: number
  jacobian: number
  hidden: number
  seed: number
}

export type Result =
  | { task: 'classification' | 'regression'; run: OdeRun }
  | { task: 'density'; run: CnfRun }
  | { task: 'latent'; run: LatentOdeRun }

const solverOf = (s: Settings) => ({
  method: s.method,
  stepSize: s.stepSize,
  rtol: s.rtol,
  atol: s.rtol * 1e-2,
  gradient: s.gradient,
  checkpoints: s.checkpoints,
  maxSteps: 2000,
})

/** The worker task of a setting. */
export function taskOf(s: Settings): Task<OdeRun | CnfRun | LatentOdeRun> {
  if (s.task === 'classification' || s.task === 'regression') {
    const data =
      s.task === 'classification' ? CLASS_DATA.task(s.classData, s.seed) : call('applied/neural/ode/reflectionData', 40)
    const options: OdeRunOptions = {
      kind: s.model,
      task: s.task,
      solver: solverOf(s),
      hidden: s.hidden,
      augment: s.augment,
      timeDependent: s.timeDependent,
      depth: s.depth,
      kinetic: s.kinetic,
      jacobian: s.jacobian,
      steps: s.steps,
      batchSize: 128,
      learningRate: s.learningRate,
      seed: s.seed,
    }
    return call<OdeRun>('applied/neural/ode/odeRun', data, options)
  }
  if (s.task === 'density') {
    const options: CnfRunOptions = {
      solver: solverOf(s),
      hidden: s.hidden,
      estimator: s.estimator,
      probe: s.probe,
      kinetic: s.kinetic,
      jacobian: s.jacobian,
      keepLabel: s.densityData.key === 'checkerboard' ? 0 : undefined,
      steps: s.steps,
      batchSize: 128,
      learningRate: s.learningRate,
      seed: s.seed,
    }
    return call<CnfRun>('applied/neural/ode/cnfRun', DENSITY_DATA.task(s.densityData, s.seed), options)
  }
  const options: LatentOdeRunOptions = {
    kind: s.latentData,
    solver: solverOf(s),
    steps: s.steps,
    learningRate: s.learningRate,
    seed: s.seed,
  }
  return call<LatentOdeRun>('applied/neural/ode/latentOdeRun', options)
}

/**
 * Presets: each sets the configuration (Train runs it). A disc inside a ring defeats a plain 2-d NODE (its flow is a
 * homeomorphism of the plane, so the disc cannot leave the ring; it can only squeeze the ring between points) and an extra dimension fixes it; the
 * reflection g(x) = −x is impossible in 1-d; an adaptive solver's work grows as the field stiffens and RNODE's
 * regularisers hold it down; a ResNet of N blocks is N Euler steps with untied weights.
 */
export const PRESETS: Record<string, { label: string; values: Record<string, string | number | boolean> }> = {
  nodeDisc: {
    label: 'NODE: disc inside a ring',
    values: { 'setup.task': 'classification', 'setup.classData': 'disc', 'setup.model': 'node', 'train.steps': 300 },
  },
  anodeDisc: {
    label: 'ANODE: disc inside a ring',
    values: { 'setup.task': 'classification', 'setup.classData': 'disc', 'setup.model': 'anode', 'train.steps': 300 },
  },
  reflection: {
    label: 'g(x) = −x: NODE cannot',
    values: { 'setup.task': 'regression', 'setup.model': 'node', 'train.steps': 200 },
  },
  nfe: {
    label: 'NFE growth (Dormand–Prince)',
    values: {
      'setup.task': 'classification',
      'setup.classData': 'spirals',
      'setup.model': 'node',
      'solver.method': 'dormand-prince',
      'solver.rtol': 1e-4,
      'train.steps': 250,
    },
  },
  rnode: {
    label: 'same, RNODE-regularised',
    values: {
      'setup.task': 'classification',
      'setup.classData': 'spirals',
      'setup.model': 'node',
      'solver.method': 'dormand-prince',
      'solver.rtol': 1e-4,
      'train.steps': 250,
      'train.kinetic': 0.01,
      'train.jacobian': 0.01,
    },
  },
  resnet: {
    label: 'ResNet = Euler',
    values: { 'setup.task': 'classification', 'setup.classData': 'moons', 'setup.model': 'resnet', 'setup.depth': 10 },
  },
  cnf: {
    label: 'FFJORD checkerboard',
    values: {
      'setup.task': 'density',
      'setup.densityData': 'checkerboard',
      'setup.estimator': 'hutchinson',
      'train.steps': 300,
    },
  },
  latent: {
    label: 'latent ODE spirals',
    values: { 'setup.task': 'latent', 'setup.latentData': 'spiral', 'train.steps': 300 },
  },
}
