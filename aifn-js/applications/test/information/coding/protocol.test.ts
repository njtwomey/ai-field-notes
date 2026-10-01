import { describe, expect, it } from 'vitest'
import { huffmanSteps } from 'aifn-applied/information/coding'
import { trace } from 'aifn/foundation/trace'
import { expectProtocol } from '../../protocol'

describe('huffmanSteps', () => {
  it('merges the two least probable nodes n − 1 times and follows the trace protocol', () => {
    const p = [0.4, 0.2, 0.2, 0.1, 0.1]
    const tr = trace(huffmanSteps(p), undefined, 20)
    expect(tr.meta.steps).toBe(4)
    expect(tr.meta.stopped).toBe('done')
    expectProtocol(huffmanSteps(p), undefined, { n: 4 })
  })
})
