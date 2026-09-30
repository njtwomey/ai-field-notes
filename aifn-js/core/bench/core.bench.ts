/**
 * Core micro-benchmarks (design K §0 and §10.5): the baseline table, re-measured against today's rules. Run with
 * `make bench` (not part of `make check`; reported, not gated). Each group pairs aifn operations with their plain-JS
 * reference where the table has one, and prints the mean time per call in µs.
 */
import { test, type TestContext } from 'vitest'
import { grad } from 'aifn/foundation/autodiff'
import { cholesky } from 'aifn/numerics/linalg'
import { normal, stream } from 'aifn/foundation/random'
import { add, exp, eye, fromData, greater, matmul, mul, sin, square, sum, where, type Value } from 'aifn/foundation/tensor'

type Fn = () => void

/** Benchmark the entries together and print each one's mean time per call. */
async function group(bench: TestContext['bench'], title: string, entries: readonly [string, Fn][]): Promise<void> {
  const result = await bench.compare(...entries.map(([name, fn]) => bench(name, fn)), { time: 300 })
  const lines = entries.map(([name]) => `  ${name.padEnd(40)} ${(result.get(name).latency.mean * 1000).toFixed(3)} µs`)
  process.stderr.write(`${title}\n${lines.join('\n')}\n`)
}

const s = stream('bench')
const vector = (n: number, seed = 1) => fromData(Float64Array.from({ length: n }, (_, k) => Math.sin(k * seed + 0.5)))
const v = vector(1000)
const w = vector(1000, 2)
const va = v.data as Float64Array
const wa = w.data as Float64Array
const tall = fromData(
  Float64Array.from({ length: 1000 }, (_, k) => k / 1000),
  [1000, 1],
)
const wide = fromData(
  Float64Array.from({ length: 1000 }, (_, k) => k / 1000),
  [10, 100],
)
const rows = fromData(
  Float64Array.from({ length: 1000 }, (_, k) => Math.cos(k)),
  [100, 10],
)
const row = vector(10, 3)
const m64 = fromData(
  Float64Array.from({ length: 64 * 64 }, (_, k) => Math.cos(k)),
  [64, 64],
)
const b50 = fromData(
  Float64Array.from({ length: 2500 }, (_, k) => Math.sin(k)),
  [50, 50],
)
const spd50 = add(matmul(b50, b50), mul(50, eye(50)))
const f = (x: Value) => sum(mul(sin(x), square(x)))
const g = (x: Value) => mul(sin(x), square(x))
const df = grad(f)
const dg = grad(g)
const ddg = grad(dg as (x: Value) => Value)

test('scalar dispatch', ({ bench }) =>
  group(bench, 'exp(0.3)', [
    ['exp primitive', () => void exp(0.3)],
    ['Math.exp', () => void Math.exp(0.3)],
  ]))

test('elementwise', ({ bench }) =>
  group(bench, 'add, 1000-vectors', [
    ['add(v, w)', () => void add(v, w)],
    ['add(m, row), 100×10 + 10', () => void add(rows, row)],
    ['mul(v, 2)', () => void mul(v, 2)],
    ['where(v > 0, v, w)', () => void where(greater(v, 0), v, w)],
    [
      'raw loop',
      () => {
        const out = new Float64Array(1000)
        for (let k = 0; k < 1000; k++) out[k] = va[k] + wa[k]
      },
    ],
  ]))

test('reductions', ({ bench }) =>
  group(bench, 'sum along axis 1', [
    ['sum(m, 1), 10×100', () => void sum(wide, 1)],
    ['sum(m, 1), 1000×1', () => void sum(tall, 1)],
  ]))

test('gradients', ({ bench }) =>
  group(bench, 'gradients', [
    ['grad(Σ sin x · x²), 1000-vector', () => void df(v)],
    ['grad(f), scalar', () => void dg(0.7)],
    ['grad(grad(f)), scalar', () => void ddg(0.7)],
  ]))

test('linear algebra', ({ bench }) =>
  group(bench, 'linear algebra', [
    ['matmul 64×64', () => void matmul(m64, m64)],
    ['cholesky 50×50', () => void cholesky(spd50)],
  ]))

test('normal draws', ({ bench }) =>
  group(bench, 'normal draws', [
    ['normal(s, 0, 1), scalar', () => void normal(s, 0, 1)],
    ['normal(s, 0, 1, { shape: [1000] })', () => void normal(s, 0, 1, { shape: [1000] })],
    [
      'Math.random × 1000',
      () => {
        const out = new Float64Array(1000)
        for (let k = 0; k < 1000; k++) out[k] = Math.random()
      },
    ],
  ]))
