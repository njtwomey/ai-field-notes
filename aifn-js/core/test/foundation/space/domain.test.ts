import { describe, expect, it } from 'vitest'
import { child, stream } from 'aifn/foundation/random'
import { discreteDomain, domainContains, domainSize, sampleDomain } from 'aifn/foundation/space'

describe('domains', () => {
  it('a discrete domain holds the integers 0 … n − 1, with optional names', () => {
    const d = discreteDomain(4, ['up', 'right', 'down', 'left'])
    expect(d).toEqual({ kind: 'discrete', n: 4, names: ['up', 'right', 'down', 'left'] })
    expect(domainSize(d)).toBe(4)
    expect([0, 3].map((x) => domainContains(d, x))).toEqual([true, true])
    expect([-1, 4, 1.5, '1', null].map((x) => domainContains(d, x))).toEqual([false, false, false, false, false])
    expect(() => discreteDomain(0)).toThrow()
    expect(() => discreteDomain(2, ['a'])).toThrow()
  })

  it('sampleDomain draws uniformly and reproducibly', () => {
    const d = discreteDomain(3)
    const draws = Array.from({ length: 3000 }, (_, i) => sampleDomain(child(stream(1), i), d))
    expect(draws.every((x) => domainContains(d, x))).toBe(true)
    for (let v = 0; v < 3; v++) expect(Math.abs(draws.filter((x) => x === v).length / 3000 - 1 / 3)).toBeLessThan(0.04)
    expect(sampleDomain(stream(5), d)).toBe(sampleDomain(stream(5), d))
  })
})
