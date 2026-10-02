/**
 * Random projections and the Johnson–Lindenstrauss lemma (`aifn/numerics/factorisation`): Gaussian points in d
 * dimensions projected to k by Gaussian and sparse random matrices; the ratios of squared pairwise distances after and
 * before, against the distortion ε the lemma guarantees at that k; and the lemma's dimension against n.
 */
import { normals, stream } from 'aifn/foundation/random'
import {
  distanceDistortion,
  johnsonLindenstraussDimension,
  johnsonLindenstraussEpsilon,
  randomProjection,
} from 'aifn/numerics/factorisation'
import { Figure } from '@lab/layout'
import { choice, int, row, slider, useComputed, useFigureState } from '@lab/state'
import { Area, Curve, formatNumber, Handle, Histogram, Plot, Plots, Points, Readout, useAxis } from '@lab/viz'

const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : '—')

const KINDS = [
  { value: 'gaussian', label: 'Gaussian N(0, 1/k)' },
  { value: 'sparse', label: 'sparse ±1/√(sk), s = 1/√d' },
  { value: 'achlioptas', label: 'Achlioptas ±√(3/k) or 0' },
] as const

const quantile = (sorted: Float64Array, p: number) => {
  const h = (sorted.length - 1) * p
  const lo = Math.floor(h)
  return sorted[lo] + (h - lo) * ((sorted[Math.min(lo + 1, sorted.length - 1)] ?? sorted[lo]) - sorted[lo])
}

export function JohnsonLindenstraussSpecimen() {
  const state = useFigureState({
    data: row('1 · points', {
      n: int(80, { ge: 5, le: 400, suggestions: [20, 80, 200], label: 'points n' }),
      d: int(1000, { ge: 10, le: 5000, suggestions: [100, 1000, 3000], label: 'input dimension d' }),
      seed: slider(0, 20, 0, { step: 1, label: 'seed' }),
    }),
    map: row('2 · projection', {
      kind: choice(KINDS as unknown as { value: string; label: string }[], 'gaussian', { label: 'matrix' }),
      k: slider(4, 600, 100, { step: 1, label: 'target dimension k' }),
    }),
  })
  const { n, d, seed } = state.data
  const { kind, k } = state.map
  const ks = [4, 8, 16, 32, 64, 100, 150, 200, 300, 400, 600]
  const options = (kind: string) =>
    kind === 'sparse'
      ? { kind: 'sparse' as const }
      : kind === 'achlioptas'
        ? { kind: 'sparse' as const, density: 1 / 3 }
        : {}
  const sweep = useComputed(
    () => {
      const X = normals(stream(`jl-points-${seed}`), [n, d])
      const rows = ks.map((kk) => {
        const { projected } = randomProjection(X, kk, stream(`jl-${kind}-${seed}-${kk}`), options(kind))
        const r = distanceDistortion(X, projected)
        const sorted = Float64Array.from(r.ratios).sort()
        return {
          k: kk,
          lo: quantile(sorted, 0.025),
          mid: quantile(sorted, 0.5),
          hi: quantile(sorted, 0.975),
          min: sorted[0],
          max: sorted[sorted.length - 1],
          worst: r.maxDistortion,
        }
      })
      return { X, rows }
    },
    [n, d, seed, kind],
    { mode: 'release' },
  )
  const at = useComputed(() => {
    const { projected } = randomProjection(sweep.value.X, k, stream(`jl-${kind}-${seed}-${k}`), options(kind))
    return distanceDistortion(sweep.value.X, projected)
  }, [sweep.value, k, kind, seed])
  const boundK = Array.from({ length: 120 }, (_, i) => 4 * (600 / 4) ** (i / 119))
  const boundEps = boundK.map((kk) => johnsonLindenstraussEpsilon(n, kk))
  const epsNow = johnsonLindenstraussEpsilon(n, k)
  const rows = sweep.value.rows
  const kAxis = useAxis({ label: 'target dimension k', log: true, range: [1, 1000] })
  const ratioAxis = useAxis({ label: '‖Rxᵢ − Rxⱼ‖² / ‖xᵢ − xⱼ‖²', range: [0, 3] })
  const histAxis = useAxis({ label: `ratio at k = ${k}`, range: [0, 3] })
  const countAxis = useAxis({ label: 'density of pairs', hold: 'union', key: `${n}-${d}-${kind}-${seed}` })
  const worstAxis = useAxis({ label: '1 + largest |ratio − 1|', range: [1, 3] })
  const nAxis = useAxis({ label: 'points n', log: true, range: [10, 1e6] })
  const dimAxis = useAxis({ label: 'JL dimension k', log: true, range: [10, 1e6] })
  const ns = Array.from({ length: 50 }, (_, i) => 10 * 10 ** ((5 * i) / 49))
  const epsilons = [0.1, 0.25, 0.5]
  const ok = (v: number) => (Number.isFinite(v) ? v : null)
  return (
    <Figure
      title="Random projections keep every distance within 1 ± ε once k grows like log n"
      purpose="A random k × d matrix scaled so that E‖Rx‖² = ‖x‖² preserves each squared distance in expectation; the Johnson–Lindenstrauss lemma bounds the worst pair's distortion by an ε that depends on k and the number of points n, not on d."
      state={state}
      defaultSize="L"
      readouts={{
        [`at k = ${k}`]: (
          <>
            <Readout label="largest |ratio − 1| over the pairs" value={fmt(at.value.maxDistortion)} />
            <Readout label="JL guarantee ε" value={Number.isFinite(epsNow) ? fmt(epsNow) : 'none (k < 24 ln n)'} />
            <Readout label="pairs" value={String(at.value.ratios.length)} />
          </>
        ),
        lemma: (
          <>
            {epsilons.map((e) => (
              <Readout key={e} label={`k for ε = ${e}, n = ${n}`} value={String(johnsonLindenstraussDimension(n, e))} />
            ))}
          </>
        ),
      }}
      caption={`Data: ${n} points with independent standard normal coordinates in d = ${d} dimensions (seeded), projected by a fresh random matrix at each k. Top: the 2.5–97.5% band (shaded), median and extremes (points) of the ratio of squared distances after and before over all ${(n * (n - 1)) / 2} pairs, with the band 1 ± ε the lemma guarantees (dashed; it exists only once k ≥ 24 ln n). The extremes stay inside the guarantee, which is loose by a constant factor. Drag the vertical handle (or the k slider) to see the ratios at that k, bottom left; bottom right, the largest deviation against the guarantee. Top right: the lemma's dimension 4 ln n / (ε²/2 − ε³/3) against n; it grows like log n and does not depend on d. Recomputed on release.`}
    >
      <Plots rows={2} cols={2} heights={[55, 45]} widths={[60, 40]}>
        <Plot x={kAxis} y={ratioAxis}>
          <Area
            name="2.5–97.5% of pairs"
            x={rows.map((r) => r.k)}
            y={rows.map((r) => r.hi)}
            base={rows.map((r) => r.lo)}
            slot={0}
            opacity={0.2}
            line={false}
            stale={sweep.stale}
          />
          <Curve name="median ratio" x={rows.map((r) => r.k)} y={rows.map((r) => r.mid)} slot={0} stale={sweep.stale} />
          <Points
            name="smallest and largest ratio"
            x={[...rows.map((r) => r.k), ...rows.map((r) => r.k)]}
            y={[...rows.map((r) => r.min), ...rows.map((r) => r.max)]}
            slot={0}
            size={6}
            stale={sweep.stale}
          />
          <Curve name="1 ± ε (JL)" x={boundK} y={boundEps.map((e) => ok(1 + e) ?? NaN)} emphasis dashed width={1.5} />
          <Curve
            name="1 ± ε (JL)"
            x={boundK}
            y={boundEps.map((e) => ok(1 - e) ?? NaN)}
            emphasis
            dashed
            width={1.5}
            silent
          />
          <Handle {...state.handle('map.k', { label: 'k', axis: 'x' })} />
        </Plot>
        <Plot x={nAxis} y={dimAxis}>
          {epsilons.map((e, i) => (
            <Curve
              key={e}
              name={`ε = ${e}`}
              x={ns}
              y={ns.map((m) => johnsonLindenstraussDimension(Math.round(m), e))}
              slot={i + 2}
            />
          ))}
          <Points
            name="this n"
            x={epsilons.map(() => n)}
            y={epsilons.map((e) => johnsonLindenstraussDimension(n, e))}
            emphasis
          />
        </Plot>
        <Plot x={histAxis} y={countAxis}>
          <Histogram values={Array.from(at.value.ratios)} name="pairs" slot={0} stale={at.stale} />
        </Plot>
        <Plot x={kAxis} y={worstAxis}>
          <Curve
            name="largest |ratio − 1|"
            x={rows.map((r) => r.k)}
            y={rows.map((r) => 1 + r.worst)}
            slot={1}
            showPoints
            stale={sweep.stale}
          />
          <Curve name="1 + ε (JL)" x={boundK} y={boundEps.map((e) => ok(1 + e) ?? NaN)} emphasis dashed width={1.5} />
        </Plot>
      </Plots>
    </Figure>
  )
}
