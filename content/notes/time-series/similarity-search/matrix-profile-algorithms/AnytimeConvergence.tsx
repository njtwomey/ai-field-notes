import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { rng } from '@/lib/math'
import { anytimeProfile, distanceMatrix, selfJoin } from '../_shared/matrix-profile'
import { beats } from '../_shared/synthetic'

const N = 400
const M = 32

/**
 * STAMP as an anytime algorithm. Rows of the distance matrix are processed in a random order; after each row every
 * entry of the profile is an upper bound on the exact value. The figure replays the run on a precomputed matrix.
 */
export function AnytimeConvergence() {
  const percent = useParam(5, { min: 0, max: 100, step: 1 })
  const seed = useParam(1, { min: 1, max: 20, step: 1 })

  const run = useMemo(() => {
    const x = beats(N, seed.value, 0.05, [6]).x
    const exact = selfJoin(x, M).profile
    const D = distanceMatrix(x, M)
    const n = D.length
    // Fisher–Yates shuffle with a seeded generator: the random row order STAMP relies on.
    const g = rng(seed.value + 7)
    const order = Array.from({ length: n }, (_, i) => i)
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(g.uniform() * (i + 1))
      ;[order[i], order[j]] = [order[j], order[i]]
    }
    // After each percent of rows: the share of entries already exact, and the mean excess over the exact profile as a
    // fraction of the largest possible distance 2√m. Computed in one pass.
    const p = new Float64Array(n).fill(Infinity)
    const exactShare: number[] = [0]
    const excess: number[] = [NaN]
    let done = 0
    for (let pct = 1; pct <= 100; pct++) {
      const target = Math.round((pct / 100) * n)
      for (; done < target; done++) {
        const i = order[done]
        const row = D[i]
        for (let j = 0; j < n; j++) {
          if (row[j] < p[i]) p[i] = row[j]
          if (row[j] < p[j]) p[j] = row[j]
        }
      }
      let same = 0
      let s = 0
      for (let i = 0; i < n; i++) {
        if (p[i] - exact[i] < 1e-9) same++
        s += p[i] - exact[i]
      }
      exactShare.push(same / n)
      excess.push(s / n / (2 * Math.sqrt(M)))
    }
    return { exact, D, order, exactShare, excess, n }
  }, [seed.value])

  const rows = Math.round((percent.value / 100) * run.n)
  const approx = useMemo(() => anytimeProfile(run.D, run.order, rows), [run, rows])
  const idx = Array.from(run.exact, (_, i) => i)
  const profiles: XYSeries[] = [
    { name: 'exact profile', type: 'line', x: idx, y: Array.from(run.exact), muted: true },
    ...(rows > 0
      ? [{ name: `after ${percent.value}% of rows`, type: 'line' as const, x: idx, y: Array.from(approx), slot: 0 }]
      : []),
  ]
  const pcts = Array.from({ length: 100 }, (_, k) => k + 1)
  const curve: XYSeries[] = [
    { name: 'share of entries exact', type: 'line', x: pcts, y: pcts.map((k) => run.exactShare[k]), slot: 1 },
    {
      name: 'predicted, 1 − (1 − r)²',
      type: 'line',
      x: pcts,
      y: pcts.map((k) => 1 - (1 - k / 100) ** 2),
      dashed: true,
      muted: true,
    },
    { name: 'mean excess / 2√m', type: 'line', x: pcts, y: pcts.map((k) => run.excess[k]), slot: 2 },
    ...(rows > 0
      ? [
          {
            name: 'now',
            type: 'scatter' as const,
            x: [percent.value],
            y: [run.exactShare[percent.value]],
            emphasis: true,
          },
        ]
      : []),
  ]

  return (
    <Interactive
      title="STAMP converges fast in random row order"
      caption={`A heartbeat-like series of ${N} samples with one abnormal beat, m = ${M}. Top: the exact matrix profile (grey) and the anytime approximation after the given share r of distance-matrix rows, taken in random order. Each processed row i sets P[i] exactly and lowers every other P[j] to at most the distance to i, so the approximation is an upper bound that only falls. Bottom: the share of entries that are already exact follows 1 − (1 − r)², because entry i is exact once row i or the row of its nearest neighbour has been processed. The mean excess falls faster than that share rises, because a near neighbour almost as close as the nearest one also brings P[i] close.`}
      controls={
        <>
          <ParamSlider label="rows computed (%)" param={percent} format={(v) => `${v}%`} withArrows />
          <ParamSlider label="seed" param={seed} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="rows computed" value={`${rows} of ${run.n}`} />
          <Readout label="entries exact" value={formatNumber(run.exactShare[percent.value])} />
          <Readout label="predicted 1 − (1 − r)²" value={formatNumber(1 - (1 - percent.value / 100) ** 2)} />
          <Readout label="mean excess / 2√m" value={rows > 0 ? formatNumber(run.excess[percent.value]) : '∞'} />
        </>
      }
    >
      <div className="space-y-4">
        <XYChart series={profiles} xLabel="subsequence start i" yLabel="P[i]" height={220} yRange={[0, undefined]} />
        <XYChart
          series={curve}
          xLabel="rows computed (%)"
          yLabel="share"
          height={180}
          xRange={[0, 100]}
          yRange={[0, 1]}
        />
      </div>
    </Interactive>
  )
}
