/**
 * Type tests: application implementations against their contracts in `aifn/contracts`. Gaps phase 1 closes are
 * marked `// @ts-expect-error phase 1`; `tsc -b` checks this file.
 */

import { describe, expectTypeOf, it } from 'vitest'
import { kmeans, kmeansSteps, type KMeansState } from 'aifn-applied/unsupervised/clustering'
import type * as C from 'aifn/contracts'
import type { Tensor } from 'aifn/foundation/tensor'

type KMeansModel = ReturnType<ReturnType<typeof kmeans>['fit']>

describe('k-means', () => {
  it("Lloyd's algorithm is today's algorithm", () => {
    expectTypeOf(kmeansSteps).returns.toExtend<C.LegacyAlgorithm<never, KMeansState>>()
  })

  it("Lloyd's algorithm is not yet the target algorithm", () => {
    // @ts-expect-error phase 1: factory `init(start, stream)`, `step(state, ctx: StepContext)`, `t` in every state
    expectTypeOf(kmeansSteps).returns.toExtend<C.Algorithm<never, KMeansState & C.Status>>()
  })

  it('the fitted model has its capabilities', () => {
    expectTypeOf<KMeansModel>().toExtend<C.Decides<Tensor> & C.Trained<KMeansState>>()
  })

  it('the fitted model lacks the model brand', () => {
    // @ts-expect-error phase 1: `kind: 'model'`
    expectTypeOf<KMeansModel>().toExtend<C.Model>()
  })
})
