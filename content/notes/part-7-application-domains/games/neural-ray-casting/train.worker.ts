/**
 * Trains the coordinate network off the main thread, so training runs at full speed while the page draws at the
 * display's rate. The worker owns the dataset, the model and the optimiser state; every `every` steps it posts the
 * weights as one flat vector (`ravel`), which the page rebuilds into its own copy of the network to draw the network's
 * view. Held-out errors cost about as much as fifteen training steps, so they are evaluated at most every 100 steps.
 */

import { child, stream } from 'aifn-compute/foundation/random'
import { ravel, type Params } from 'aifn-compute/foundation/pytree'
import type { TrainingState } from 'aifn-compute/nn/training'
import { rayOffsets, worldById, type World } from '../_shared/world'
import { buildModel, errors, type Model, type ModelOptions } from './neural'

export type ToWorker =
  | { type: 'init'; generation: number; world: string; options: ModelOptions }
  | { type: 'run'; on: boolean }
  | { type: 'every'; steps: number }

export type Checkpoint = { t: number; train: number; test: number; relative: number; within10: number }

export type FromWorker = {
  type: 'sync'
  generation: number
  t: number
  params: Float64Array
  checkpoint: Checkpoint | null
  stepsPerSecond: number
}

/** Milliseconds of training between looks at the message queue (pause, reset, a new setting). */
const SLICE = 40
const EVAL_MIN = 100
const ROOT = stream('neural-ray-casting/run')

let model: Model | null = null
let world: World | null = null
let options: ModelOptions | null = null
let state: TrainingState<Params[]> | null = null
let generation = 0
let running = false
let every = 100
let lastEval = 0
let timer: ReturnType<typeof setTimeout> | undefined
let rateFrom = { t: 0, time: 0 }

function post(evaluate: boolean) {
  if (!model || !state || !world || !options) return
  const checkpoint = evaluate
    ? {
        t: state.t,
        train: errors(model.net, world, state.params, model.trainCheck, options).rmse,
        ...(({ rmse, relative, within10 }) => ({ test: rmse, relative, within10 }))(
          errors(model.net, world, state.params, model.test, options),
        ),
      }
    : null
  const now = performance.now()
  const stepsPerSecond = now > rateFrom.time ? ((state.t - rateFrom.t) * 1000) / (now - rateFrom.time) : 0
  rateFrom = { t: state.t, time: now }
  const { vector } = ravel(state.params)
  const message: FromWorker = { type: 'sync', generation, t: state.t, params: vector, checkpoint, stepsPerSecond }
  postMessage(message, { transfer: [vector.buffer] })
}

function loop() {
  timer = undefined
  if (!running || !model || !state) return
  const start = performance.now()
  while (performance.now() - start < SLICE) {
    state = model.alg.step(state, { t: state.t, stream: child(ROOT, 'step', state.t) })
    if (state.t % every === 0) {
      const evaluate = state.t - lastEval >= Math.max(every, EVAL_MIN)
      if (evaluate) lastEval = state.t
      post(evaluate)
    }
  }
  timer = setTimeout(loop, 0)
}

onmessage = (e: MessageEvent<ToWorker>) => {
  const m = e.data
  if (m.type === 'init') {
    generation = m.generation
    world = worldById(m.world)
    options = m.options
    model = buildModel(world, rayOffsets(options.rays, options.fov), options)
    state = model.alg.init(model.start, child(ROOT, 'init'))
    lastEval = 0
    rateFrom = { t: 0, time: performance.now() }
    post(true)
  } else if (m.type === 'every') {
    every = Math.max(1, Math.round(m.steps))
  } else {
    running = m.on
    rateFrom = { t: state?.t ?? 0, time: performance.now() }
  }
  if (running && timer === undefined) timer = setTimeout(loop, 0)
}
