/**
 * Type tests: representative implementations against their contracts in `aifn/contracts`. A contract that an
 * implementation does not satisfy yet is marked `// @ts-expect-error phase 1`, so the gap is visible here and `tsc`
 * fails once phase 1 closes it (the directive is then unused). The tests run no numerics; `tsc -b` checks them.
 */

import { describe, expectTypeOf, it } from 'vitest'
import type * as C from 'aifn/contracts'
import { transferFunction } from 'aifn/systems'
import { moons } from 'aifn-applied/data/synthetic'
import { expBijector } from 'aifn/probability/bijectors'
import { Normal, type Distribution, type Target } from 'aifn/probability/distributions'
import { welch } from 'aifn/signal/spectral'
import { hasDecide, type Decides, type Distribution as Predictive, type Predicts } from 'aifn/learning/estimators'
import { fromEdges, type Graph } from 'aifn/graph'
import { rbf } from 'aifn/learning/kernels'
import { huber } from 'aifn/learning/losses'
import { auroc } from 'aifn/learning/metrics'
import { gradientDescent } from 'aifn/optim/first-order'
import { normal, stream, type Stream } from 'aifn/foundation/random'
import { softplus } from 'aifn/numerics/special'
import { add, exp, sum, type Tensor, type Traced, type Value, type VectorLike } from 'aifn/foundation/tensor'
import type { Algorithm, Trace } from 'aifn/foundation/trace'

describe('numbers', () => {
  it('aifn/tensor re-exports the contract types themselves', () => {
    expectTypeOf<Tensor>().toEqualTypeOf<C.Tensor>()
    expectTypeOf<Traced>().toEqualTypeOf<C.Traced>()
    expectTypeOf<Value>().toEqualTypeOf<C.Value>()
    expectTypeOf<VectorLike>().toEqualTypeOf<C.VectorLike>()
  })

  it('scalars and integer metadata are numbers', () => {
    expectTypeOf<C.Scalar>().toEqualTypeOf<number>()
    expectTypeOf<C.Shape>().toEqualTypeOf<readonly number[]>()
  })

  it('an object shaped like a tensor without the brand is not a tensor', () => {
    type Unbranded = Omit<C.Tensor, keyof C.Tensor & symbol>
    expectTypeOf<Unbranded>().not.toExtend<C.Tensor>()
  })
})

describe('function families', () => {
  it('elementwise primitives are Unary or Binary', () => {
    expectTypeOf(exp).toExtend<C.Unary>()
    expectTypeOf(softplus).toExtend<C.Unary>()
    expectTypeOf(add).toExtend<C.Binary>()
  })

  it('reductions are Reduction', () => {
    expectTypeOf(sum).toExtend<C.Reduction>()
  })

  it('samplers are Sampler', () => {
    expectTypeOf(normal).toExtend<C.Sampler<[C.Raw, C.Raw]>>()
  })

  it('a kernel evaluates as a KernelFn', () => {
    expectTypeOf(rbf().evaluate).toExtend<C.KernelFn>()
  })

  it('metrics and losses are MetricFn and LossFn', () => {
    expectTypeOf(auroc).toExtend<C.MetricFn>()
    expectTypeOf(huber).toExtend<C.LossFn>()
  })
})

describe('protocols: today (Legacy…)', () => {
  it('distributions, bijectors and log-densities', () => {
    expectTypeOf(Normal(0, 1)).toExtend<C.LegacyDistribution>()
    expectTypeOf<Distribution>().toEqualTypeOf<C.LegacyDistribution>()
    expectTypeOf(expBijector).toExtend<C.Bijector>()
    expectTypeOf<Target>().toEqualTypeOf<C.LegacyLogDensity>()
  })

  it('streams, algorithms and traces', () => {
    expectTypeOf(stream(1)).toExtend<C.LegacyStream>()
    expectTypeOf<Stream>().toEqualTypeOf<C.LegacyStream>()
    expectTypeOf(gradientDescent).returns.toExtend<C.LegacyAlgorithm<never, unknown>>()
    expectTypeOf<Trace<number>>().toEqualTypeOf<C.LegacyTrace<number>>()
    expectTypeOf<Algorithm<number, number>>().toEqualTypeOf<C.LegacyAlgorithm<number, number>>()
  })

  it('kernels, graphs, metric and loss metadata', () => {
    expectTypeOf(rbf()).toExtend<C.LegacyKernel>()
    expectTypeOf(fromEdges(2, [[0, 1]])).toExtend<C.LegacyGraph>()
    expectTypeOf<Graph>().toEqualTypeOf<C.LegacyGraph>()
    expectTypeOf(auroc.info).toExtend<C.LegacyMetricInfo>()
    expectTypeOf(huber.info).toExtend<C.LegacyLossInfo>()
  })

  it('capabilities', () => {
    expectTypeOf<Decides<Tensor>>().toEqualTypeOf<C.Decides<Tensor>>()
    expectTypeOf<Predicts<Tensor>>().toEqualTypeOf<C.Predicts<Tensor, Predictive>>()
    expectTypeOf(hasDecide<{ decide(x: Tensor): Tensor }>).guards.toExtend<C.Decides<Tensor>>()
  })
})

describe('protocols: target (phase 1 closes these gaps)', () => {
  it('a distribution lacks the kind brand, rsample and a plain-data stream', () => {
    // @ts-expect-error phase 1: `kind: 'distribution'`, `sample(s: Stream)`
    expectTypeOf(Normal(0, 1)).toExtend<C.Distribution>()
  })

  it('an algorithm takes its stream in init, not a StepContext in step', () => {
    // @ts-expect-error phase 1: `step(state, ctx)`, states carry `Status`
    expectTypeOf(gradientDescent).returns.toExtend<C.Algorithm<never, C.Status>>()
  })

  it("estimators' minimal predictive is not a supertype of the distribution protocol", () => {
    // @ts-expect-error phase 1: `sample(s, { shape })` returns `number | Tensor` on a univariate law; one protocol
    expectTypeOf<C.LegacyDistribution>().toExtend<Predictive>()
  })

  it('a stream is an object with methods, not plain data', () => {
    // @ts-expect-error phase 1: `{ key: Key, position }`
    expectTypeOf(stream(1)).toExtend<C.Stream>()
  })

  it('traces, kernels, graphs and datasets lack the kind brand', () => {
    // @ts-expect-error phase 1: `kind: 'trace'`
    expectTypeOf<Trace<number>>().toExtend<C.Trace<number>>()
    // @ts-expect-error phase 1: `kind: 'kernel'`
    expectTypeOf(rbf()).toExtend<C.Kernel>()
    // @ts-expect-error phase 1: `kind: 'graph'`
    expectTypeOf(fromEdges(2, [[0, 1]])).toExtend<C.Graph>()
    // @ts-expect-error phase 1: `kind: 'dataset'`; meta holds today's truth and recipe steps
    expectTypeOf(moons(stream(1))).toExtend<C.Dataset>()
  })

  it('metric and loss metadata lack the registry fields', () => {
    // @ts-expect-error phase 1: `kind`, `module`, `stability`, `notes`
    expectTypeOf(auroc.info).toExtend<C.MetricInfo>()
    // @ts-expect-error phase 1: `kind`, `module`, `stability`, `notes`
    expectTypeOf(huber.info).toExtend<C.LossInfo>()
  })

  it('signal processing returns bespoke shapes', () => {
    // @ts-expect-error phase 1: `welch` returns `{ f, psd }`, not a `Spectrum`
    expectTypeOf(welch).returns.toExtend<C.Spectrum>()
    // @ts-expect-error phase 1: `TransferFunction` becomes an `LtiSystem` with `repr: { form: 'tf' }`
    expectTypeOf(transferFunction).returns.toExtend<C.LtiSystem>()
  })
})
