/**
 * The compute worker of `useComputed(…, { mode: 'worker' })`: it evaluates one task at a time (`state/task.ts`),
 * resolving each address to an aifn export by importing that module on first use, and answers with the result made
 * cloneable. aifn is DOM-free, so its modules run here unchanged. A stale job is cancelled by the page, which
 * terminates this worker and starts a fresh one.
 */
import { fromMessage, isTask, toMessage, type WorkerRequest, type WorkerResponse } from './task'

// Every aifn module (a directory with an index.ts; `_` folders are private), imported lazily by path.
const MODULES = {
  ...import.meta.glob(['../../../aifn-js/core/src/**/index.ts', '!**/_*/**']),
  ...import.meta.glob(['../../../aifn-js/applications/src/**/index.ts', '!**/_*/**']),
} as Record<string, () => Promise<Record<string, unknown>>>

const CORE = '../../../aifn-js/core/src/'
const APPLIED = '../../../aifn-js/applications/src/'

/** The aifn export at `address` (`<module>/<export>`), from the core or the applications. */
async function resolve(address: string): Promise<unknown> {
  const cut = address.lastIndexOf('/')
  if (cut <= 0) throw new Error(`worker: '${address}' is not an address (<module>/<export>)`)
  const module = address.slice(0, cut)
  const key = address.slice(cut + 1)
  const local = module.replace(/^applied\//, '')
  const load = MODULES[`${CORE}${module}/index.ts`] ?? MODULES[`${APPLIED}${local}/index.ts`]
  if (!load) throw new Error(`worker: no aifn module '${module}' (from '${address}')`)
  const ns = await load()
  if (key in ns) return ns[key]
  // An entry kept only in a table of entries (a registry) is found by its key.
  for (const v of Object.values(ns))
    if (v !== null && typeof v === 'object' && key in v) {
      const entry = (v as Record<string, unknown>)[key] as { info?: { key?: unknown } } | undefined
      if (entry?.info?.key === key) return entry
    }
  throw new Error(`worker: module '${module}' has no export '${key}'`)
}

/** Evaluate a task tree: calls (innermost first), arrays and plain objects; leaves re-branded. */
async function evaluate(x: unknown): Promise<unknown> {
  if (isTask(x)) {
    const f = await resolve(x.$call)
    const args = await Promise.all(x.args.map(evaluate))
    if (typeof f === 'function') return (f as (...a: unknown[]) => unknown)(...args)
    if (args.length > 0) throw new Error(`worker: '${x.$call}' is not a function`)
    return f
  }
  if (Array.isArray(x)) return Promise.all(x.map(evaluate))
  if (typeof x === 'object' && x !== null && Object.getPrototypeOf(x) === Object.prototype) {
    if ('shape' in x && 'data' in x) return fromMessage(x)
    const out: Record<string, unknown> = {}
    for (const [k, v] of Object.entries(x)) out[k] = await evaluate(v)
    return out
  }
  return x
}

// The worker's global scope; the lab's TypeScript program has the DOM library, not the worker one.
const scope = self as unknown as {
  postMessage(r: WorkerResponse): void
  onmessage: ((e: MessageEvent<WorkerRequest>) => void) | null
}
const post = (r: WorkerResponse) => scope.postMessage(r)

scope.onmessage = async (e) => {
  const { id, task } = e.data
  const t0 = performance.now()
  try {
    const value = toMessage(await evaluate(task))
    post({ id, ok: true, value, ms: performance.now() - t0 })
  } catch (err) {
    post({ id, ok: false, error: err instanceof Error ? err.message : String(err), ms: performance.now() - t0 })
  }
}
