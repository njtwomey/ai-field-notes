/**
 * Models trained outside the page, run with ONNX Runtime Web. Every model follows one contract:
 *
 *   input  rays:     float32 [N, 3]  (x, y, φ): position in cells (x right, y down) and absolute ray angle in radians
 *   output distance: float32 [N]     distance along each ray to the first wall, in cells
 *
 * The runtime (about 10 MB of WebAssembly) is imported on first use, so it costs nothing until a reader picks a model.
 * It runs single-threaded: threads need cross-origin isolation, which GitHub Pages does not provide.
 */

import type { InferenceSession } from 'onnxruntime-web/wasm'
import wasm from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url'
import mjs from 'onnxruntime-web/ort-wasm-simd-threaded.mjs?url'
import manifestJson from './_models/manifest.json'

export type TrainedModel = {
  id: string
  file: string
  world: string
  /** The model class, e.g. `density`, and its description. */
  family: string
  familyLabel: string
  /** The trained model within its class, e.g. `density-128`, and its description. */
  name: string
  label: string
  params: number
  within10: number
  medianRelative: number
  rmse: number
}

const urls = import.meta.glob<string>('./_models/*.onnx', { query: '?url', import: 'default', eager: true })

/** The models exported by the training scripts, with their map and held-out scores. */
export const TRAINED: readonly TrainedModel[] = manifestJson as TrainedModel[]

export const modelUrl = (m: TrainedModel): string => urls[`./_models/${m.file}`]

let runtime: Promise<typeof import('onnxruntime-web/wasm')> | null = null

function ort() {
  runtime ??= import('onnxruntime-web/wasm').then((o) => {
    o.env.wasm.numThreads = 1
    o.env.wasm.wasmPaths = { wasm, mjs }
    return o
  })
  return runtime
}

/** Load a model from its URL. */
export async function loadModel(url: string): Promise<InferenceSession> {
  const o = await ort()
  const session = await o.InferenceSession.create(url)
  if (!session.inputNames.includes('rays') || !session.outputNames.includes('distance'))
    throw new Error(
      `expected input "rays" and output "distance", found ${session.inputNames.join(', ')} → ${session.outputNames.join(', ')}`,
    )
  return session
}

/** Distances for rays packed as (x, y, φ) triples. */
export async function runRays(session: InferenceSession, rays: Float32Array): Promise<Float32Array> {
  const o = await ort()
  const out = await session.run({ rays: new o.Tensor('float32', rays, [rays.length / 3, 3]) })
  return out.distance.data as Float32Array
}
