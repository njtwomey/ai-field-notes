import { useMemo } from 'react'
import { Interactive, Readout, XYChart, type XYSeries } from 'aifn-render'
import { rng } from '@/lib/math'
import { corrupt, randomPattern } from '../_shared/spins'
import { denseRecall, softmaxUpdate } from './dense'

const N = 100
const TESTS = 20
const NOISE = 0.1
const SIZES = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 40, 60, 80, 100, 150, 200, 300, 400]

const MEMORIES: {
  name: string
  slot: number
  recall: (p: Int8Array[], cue: Int8Array, seed: number) => ArrayLike<number>
}[] = [
  {
    name: 'classical, F(x) = x²',
    slot: 0,
    recall: (p, cue, seed) => denseRecall(p, cue, { kind: 'power', n: 2 }, rng(seed).uniform),
  },
  {
    name: 'dense, F(x) = x³',
    slot: 1,
    recall: (p, cue, seed) => denseRecall(p, cue, { kind: 'power', n: 3 }, rng(seed).uniform),
  },
  { name: 'softmax update, β = 0.1', slot: 2, recall: (p, cue) => softmaxUpdate(p, cue, 0.1) },
]

const exact = (s: ArrayLike<number>, xi: ArrayLike<number>) => {
  for (let i = 0; i < xi.length; i++) if ((s[i] >= 0 ? 1 : -1) !== xi[i]) return false
  return true
}

/**
 * The fraction of random patterns recalled exactly from a 10%-corrupted cue, against the number of stored patterns,
 * for N = 100 neurons. Computed once in the browser; the classical curve stops once it has reached zero.
 */
export function CapacityCurves() {
  const curves = useMemo(
    () =>
      MEMORIES.map((memory) => {
        const x: number[] = []
        const y: number[] = []
        for (const K of SIZES) {
          const { uniform } = rng(K)
          const patterns = Array.from({ length: K }, () => randomPattern(N, uniform))
          const tests = Math.min(K, TESTS)
          let ok = 0
          for (let mu = 0; mu < tests; mu++) {
            const cue = corrupt(patterns[mu], NOISE, rng(1000 * K + mu).uniform)
            if (exact(memory.recall(patterns, cue, mu + 1), patterns[mu])) ok++
          }
          x.push(Math.log10(K))
          y.push(ok / tests)
          // Past total failure the classical network only gets slower; its curve stays at zero.
          if (ok === 0 && y.length > 3 && y[y.length - 2] === 0) break
        }
        return { x, y }
      }),
    [],
  )
  const series = useMemo((): XYSeries[] => {
    const out: XYSeries[] = MEMORIES.map((m, k) => ({ name: m.name, type: 'line', ...curves[k], slot: m.slot }))
    const limit = Math.log10(0.138 * N)
    out.push({ name: 'K = 0.138 N', type: 'line', x: [limit, limit], y: [0, 1], dashed: true, muted: true })
    return out
  }, [curves])

  return (
    <Interactive
      title="Capacity: classical against dense memories"
      caption={
        <>
          N = 100 neurons store K random ±1 patterns. Each point is the fraction of {TESTS} patterns recalled exactly
          from a cue with 10% of its entries flipped. The x-axis is log₁₀ K. The classical network fails near K = 0.138N
          ≈ 14. The cubic dense memory holds past K = 200, and one softmax update holds for every K shown, beyond the
          number of neurons.
        </>
      }
      readout={<Readout label="neurons N" value={N} />}
    >
      <XYChart
        height={280}
        xLabel="log₁₀ K (stored patterns)"
        yLabel="fraction recalled"
        xRange={[0, Math.log10(400)]}
        yRange={[0, 1.05]}
        series={series}
        ariaLabel="Fraction of patterns recalled against the number stored, for three memories"
      />
    </Interactive>
  )
}
