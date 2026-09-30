/**
 * Choosing an inference engine for a model (plan §7.2), and the custom engines registered against model shapes, with
 * collapsed Gibbs sampling for latent Dirichlet allocation as the example (Griffiths & Steyvers 2004, "Finding
 * scientific topics", PNAS 101 suppl. 1, eq. 5).
 *
 * `infer(model, bindings, engine?)` returns `{ engine, algorithm, options }`, ready for `run`, `trace` or `live`.
 * Without an engine name, a registered custom engine that matches the model is used first; then variable elimination
 * for models whose latent variables are all discrete, else Gibbs sampling.
 */

import { categorical, type Stream } from 'aifn/random'
import { logGamma } from 'aifn/special'
import { fromData, isTensor, toFlat, type Matrix, type Tensor } from 'aifn/tensor'
import type { Algorithm } from 'aifn/trace'
import { beliefPropagationSteps } from './bp'
import { enumerationSteps, variableEliminationSteps } from './exact'
import { gibbsSteps } from './gibbs'
import { argRefs, dist, model as describe, plateChain, type Arg, type Bindings, type Model, type Nested } from './model'
import { toDiscreteFactorGraph } from './structure'

/** A custom engine: a test of the model's shape and a constructor of the algorithm and its options. */
export interface EngineRegistration {
  name: string
  matches(model: Model): boolean
  create(model: Model, bindings: Bindings): { algorithm: Algorithm<unknown, unknown>; options: unknown }
}

const registry: EngineRegistration[] = []

/** Register a custom engine; later registrations take precedence. Returns a function that removes it. */
export function registerEngine(engine: EngineRegistration): () => void {
  registry.unshift(engine)
  return () => {
    const i = registry.indexOf(engine)
    if (i >= 0) registry.splice(i, 1)
  }
}

/** The registered custom engines, most recent first. */
export const registeredEngines = (): readonly EngineRegistration[] => [...registry]

/** Built-in engine names. */
export type BuiltInEngine = 'enumeration' | 'variable-elimination' | 'belief-propagation' | 'gibbs'

/** An engine chosen for a model: its name, the algorithm and the options to run it with. */
export interface Inference {
  engine: string
  algorithm: Algorithm<unknown, unknown>
  options: unknown
}

/**
 * Pick an inference engine for a model and data. Discrete engines tabulate the model with
 * `toDiscreteFactorGraph` (log Z of the result plus its `logConstant` is log p(data)); Gibbs needs a stream when run.
 */
export function infer(model: Model, bindings: Bindings = {}, engine?: BuiltInEngine | string): Inference {
  const custom = engine === undefined ? registry.find((e) => e.matches(model)) : registry.find((e) => e.name === engine)
  if (custom) return { engine: custom.name, ...custom.create(model, bindings) }
  const discrete = () => toDiscreteFactorGraph(model, bindings).graph
  const as = <O, S>(a: Algorithm<O, S>) => a as unknown as Algorithm<unknown, unknown>
  switch (engine) {
    case 'enumeration':
      return { engine, algorithm: as(enumerationSteps), options: { graph: discrete() } }
    case 'variable-elimination':
      return { engine, algorithm: as(variableEliminationSteps), options: { graph: discrete() } }
    case 'belief-propagation':
      return { engine, algorithm: as(beliefPropagationSteps), options: { graph: discrete() } }
    case 'gibbs':
      return { engine, algorithm: as(gibbsSteps), options: { model, bindings } }
    case undefined:
      try {
        return {
          engine: 'variable-elimination',
          algorithm: as(variableEliminationSteps),
          options: { graph: discrete() },
        }
      } catch {
        return { engine: 'gibbs', algorithm: as(gibbsSteps), options: { model, bindings } }
      }
    default:
      throw new Error(`infer: no engine named ${engine}`)
  }
}

// ── LDA ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

/** The plan's LDA description: topics φ_k ~ Dir(β), mixtures θ_d ~ Dir(α), z_dn ~ Cat(θ_d), w_dn ~ Cat(φ_{z_dn}). */
export function ldaModel(): Model {
  return ldaDescription
}

const ldaDescription: Model = describe('Latent Dirichlet allocation', (m) => {
  const K = m.size('K')
  const V = m.size('V')
  const topics = m.plate('topics', K, { label: 'K' })
  const docs = m.plate('documents', 'D', { label: 'D' })
  const words = docs.plate('words', 'N', { label: 'N_d' })
  const alpha = m.constant('α', undefined, { label: '\\alpha' })
  const beta = m.constant('β', undefined, { label: '\\beta' })
  const phi = topics.variable('φ', dist.Dirichlet(beta, V), { label: '\\boldsymbol{\\phi}_k' })
  const theta = docs.variable('θ', dist.Dirichlet(alpha, K), { label: '\\boldsymbol{\\theta}_d' })
  const z = words.variable('z', dist.Categorical(theta), { label: 'z_{dn}' })
  words.observed('w', dist.Categorical(phi.at(z)), { label: 'w_{dn}' })
})

/** The names of an LDA-shaped model's nodes, or null when the model is not LDA-shaped. */
export function matchLda(
  m: Model,
): { phi: string; theta: string; z: string; w: string; alpha: Arg; beta: Arg; K: Arg; V: Arg } | null {
  const w = m.nodes.find(
    (n) =>
      n.role === 'observed' &&
      n.dist?.family === 'Categorical' &&
      typeof n.dist.args[0] === 'object' &&
      'select' in (n.dist.args[0] as object),
  )
  if (!w) return null
  const ref = w.dist!.args[0] as { node: string; select: string }
  const phi = m.nodes.find((n) => n.name === ref.node)
  const z = m.nodes.find((n) => n.name === ref.select)
  if (phi?.dist?.family !== 'Dirichlet' || z?.dist?.family !== 'Categorical' || z.role !== 'latent') return null
  const thetaName = argRefs(z.dist.args)[0]
  const theta = m.nodes.find((n) => n.name === thetaName)
  if (theta?.dist?.family !== 'Dirichlet' || z.plate !== w.plate) return null
  // θ is per document, z and w per word inside the document plate.
  if (!plateChain(m, z.plate).includes(theta.plate ?? '')) return null
  return {
    phi: phi.name,
    theta: theta.name,
    z: z.name,
    w: w.name,
    alpha: theta.dist.args[0],
    beta: phi.dist.args[0],
    K: theta.dist.args[1] ?? 0,
    V: phi.dist.args[1] ?? 0,
  }
}

/** Options of {@link ldaCollapsedGibbsSteps}. */
export interface LdaOptions {
  /** Documents as arrays of word ids in 0 … V − 1. */
  documents: readonly (readonly number[])[]
  topics: number
  vocabulary: number
  /** Symmetric Dirichlet concentrations on θ (α) and φ (β). */
  alpha: number
  beta: number
}

/** The state of collapsed Gibbs sampling for LDA: topic assignments and the count tables they imply. */
export interface LdaState {
  documents: readonly (readonly number[])[]
  topics: number
  vocabulary: number
  alpha: number
  beta: number
  stream: Stream
  /** z_dn, per document (int32 vectors). */
  assignments: Tensor[]
  /** n_dk (D × K), n_kw (K × V) and n_k (K). */
  docTopic: Matrix
  topicWord: Matrix
  topicTotals: Tensor
  sweep: number
  /** log p(w | z) with φ integrated out. */
  logLikelihood: number
}

function ldaLogLikelihood(nkw: Float64Array, nk: Float64Array, K: number, V: number, beta: number): number {
  let total = 0
  for (let k = 0; k < K; k++) {
    total += (logGamma(V * beta) as number) - V * (logGamma(beta) as number) - (logGamma(nk[k] + V * beta) as number)
    for (let w = 0; w < V; w++) total += logGamma(nkw[k * V + w] + beta) as number
  }
  return total
}

/**
 * Collapsed Gibbs sampling for LDA as a traceable algorithm (Griffiths & Steyvers 2004): θ and φ are integrated out,
 * and each step resamples every token's topic from p(z = k | rest) ∝ (n_dk + α)(n_kw + β)/(n_k + Vβ), with the
 * token's own counts removed. Token (d, n) in sweep t draws from `stream.child(t, d, n)`.
 */
export const ldaCollapsedGibbsSteps: Algorithm<LdaOptions, LdaState> = {
  name: 'pgm.lda.collapsed-gibbs',
  init: (o, s) => {
    if (!s) throw new Error('ldaCollapsedGibbsSteps: needs a stream')
    const { topics: K, vocabulary: V } = o
    const D = o.documents.length
    const ndk = new Float64Array(D * K)
    const nkw = new Float64Array(K * V)
    const nk = new Float64Array(K)
    const assignments = o.documents.map((doc, d) => {
      const z = Int32Array.from(doc, (w, n) => {
        const k = s.child('init', d, n).int(K)
        ndk[d * K + k]++
        nkw[k * V + w]++
        nk[k]++
        return k
      })
      return fromData(z, [z.length])
    })
    return {
      ...o,
      topics: K,
      vocabulary: V,
      stream: s,
      assignments,
      docTopic: fromData(ndk, [D, K]),
      topicWord: fromData(nkw, [K, V]),
      topicTotals: fromData(nk, [K]),
      sweep: 0,
      logLikelihood: ldaLogLikelihood(nkw, nk, K, V, o.beta),
    }
  },
  step: (s) => {
    const { topics: K, vocabulary: V, alpha, beta } = s
    const ndk = Float64Array.from(s.docTopic.data)
    const nkw = Float64Array.from(s.topicWord.data)
    const nk = Float64Array.from(s.topicTotals.data)
    const p = new Float64Array(K)
    const assignments = s.documents.map((doc, d) => {
      const z = Int32Array.from(s.assignments[d].data)
      doc.forEach((w, n) => {
        const old = z[n]
        ndk[d * K + old]--
        nkw[old * V + w]--
        nk[old]--
        for (let k = 0; k < K; k++) p[k] = ((ndk[d * K + k] + alpha) * (nkw[k * V + w] + beta)) / (nk[k] + V * beta)
        const k = categorical(s.stream.child(s.sweep, d, n), p)
        z[n] = k
        ndk[d * K + k]++
        nkw[k * V + w]++
        nk[k]++
      })
      return fromData(z, [z.length])
    })
    return {
      ...s,
      assignments,
      docTopic: fromData(ndk, s.docTopic.shape),
      topicWord: fromData(nkw, s.topicWord.shape),
      topicTotals: fromData(nk, s.topicTotals.shape),
      sweep: s.sweep + 1,
      logLikelihood: ldaLogLikelihood(nkw, nk, K, V, beta),
    }
  },
}

/** Point estimates φ_kw = (n_kw + β)/(n_k + Vβ) (K × V) and θ_dk = (n_dk + α)/(N_d + Kα) (D × K). */
export function ldaEstimates(s: LdaState): { topicWord: Matrix; docTopic: Matrix } {
  const { topics: K, vocabulary: V, alpha, beta } = s
  const phi = s.topicWord.data.map((n, i) => (n + beta) / (s.topicTotals.data[Math.floor(i / V)] + V * beta))
  const theta = s.docTopic.data.map((n, i) => {
    const d = Math.floor(i / K)
    return (n + alpha) / (s.documents[d].length + K * alpha)
  })
  return { topicWord: fromData(phi, [K, V]), docTopic: fromData(theta, s.docTopic.shape) }
}

const scalar = (a: Arg, b: Bindings): number => {
  if (typeof a === 'number') return a
  if (typeof a === 'object' && a !== null && 'kind' in a) {
    if (a.kind === 'size') return b.sizes?.[a.name] as number
    const c = b.constants?.[a.node]
    if (typeof c === 'number') return c
  }
  throw new Error('lda: α, β, K and V must be numbers')
}

/** The LDA engine: matches LDA-shaped models and runs collapsed Gibbs on the observed words. */
export const ldaEngine: EngineRegistration = {
  name: 'lda-collapsed-gibbs',
  matches: (m) => matchLda(m) !== null,
  create: (m, b) => {
    const shape = matchLda(m)!
    const docs = b.data?.[shape.w] as Nested
    const documents = (Array.isArray(docs) ? docs : []).map((d) =>
      isTensor(d) ? Array.from(toFlat(d as Tensor)) : Array.from(d as number[]),
    )
    const options: LdaOptions = {
      documents,
      topics: scalar(shape.K, b),
      vocabulary: scalar(shape.V, b),
      alpha: scalar(shape.alpha, b),
      beta: scalar(shape.beta, b),
    }
    return { algorithm: ldaCollapsedGibbsSteps as unknown as Algorithm<unknown, unknown>, options }
  },
}

registerEngine(ldaEngine)
