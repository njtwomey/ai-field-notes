import { useMemo, useState } from 'react'
import {
  Figure,
  ControlGroup,
  Select,
  NumberSelector,
  Plots,
  Plot,
  Area,
  Curve,
  Points,
  Histogram,
  Handle,
  Readout,
  useAxis,
} from 'aifn-render'
import { normals, stream } from 'aifn/foundation/random'
import {
  distanceDistortion,
  johnsonLindenstraussDimension,
  johnsonLindenstraussEpsilon,
  randomProjection,
} from 'aifn/numerics/factorisation'

const KINDS = [
  { value: 'gaussian', label: 'Gaussian N(0, 1/k)' },
  { value: 'sparse', label: 'sparse ±1/√(sk), s = 1/√d' },
  { value: 'achlioptas', label: 'Achlioptas ±√(3/k) or 0' },
] as const

type MatrixKind = (typeof KINDS)[number]['value']

const quantile = (sorted: Float64Array, p: number) => {
  const h = (sorted.length - 1) * p
  const lo = Math.floor(h)
  return sorted[lo] + (h - lo) * ((sorted[Math.min(lo + 1, sorted.length - 1)] ?? sorted[lo]) - sorted[lo])
}

const KS = [4, 8, 16, 32, 64, 100, 150, 200, 300, 400, 600]

export function RandomProjectionExplorer() {
  const [kind, setKind] = useState<MatrixKind>('gaussian')
  const [k, setK] = useState(100)
  const [n, setN] = useState(80)
  const [d, setD] = useState(1000)
  const seed = 0

  const options = (knd: string) =>
    knd === 'sparse'
      ? { kind: 'sparse' as const }
      : knd === 'achlioptas'
        ? { kind: 'sparse' as const, density: 1 / 3 }
        : {}

  const sweep = useMemo(() => {
    const X = normals(stream(`jl-points-${seed}`), [n, d])
    const rows = KS.map((kk) => {
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
  }, [n, d, seed, kind])

  const at = useMemo(() => {
    const { projected } = randomProjection(sweep.X, k, stream(`jl-${kind}-${seed}-${k}`), options(kind))
    return distanceDistortion(sweep.X, projected)
  }, [sweep.X, k, kind, seed])

  const boundK = Array.from({ length: 120 }, (_, i) => 4 * (600 / 4) ** (i / 119))
  const boundEps = boundK.map((kk) => johnsonLindenstraussEpsilon(n, kk))
  const epsNow = johnsonLindenstraussEpsilon(n, k)
  const rows = sweep.rows

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
      title="Random projections and the Johnson–Lindenstrauss lemma"
      purpose="A random k × d projection matrix preserves all pairwise squared Euclidean distances within 1 ± ε with high probability once k = O(log n / ε²), completely independent of the ambient dimension d."
      defaultSize="L"
      controls={
        <ControlGroup title="Projection configuration">
          <Select
            label="random matrix distribution"
            value={kind}
            onChange={(v) => setKind(v as MatrixKind)}
            options={KINDS.map((x) => ({ value: x.value, label: x.label }))}
          />
          <NumberSelector
            label="target dimension k"
            value={k}
            onChange={(val) => setK(Math.max(4, Math.min(600, Math.round(val))))}
            min={4}
            max={600}
            step={10}
            suggestions={[20, 50, 100, 200, 400]}
          />
          <NumberSelector
            label="points n"
            value={n}
            onChange={setN}
            min={20}
            max={150}
            step={10}
            suggestions={[40, 80, 120]}
          />
          <NumberSelector
            label="input dimension d"
            value={d}
            onChange={setD}
            min={100}
            max={3000}
            step={100}
            suggestions={[500, 1000, 2000]}
          />
        </ControlGroup>
      }
      readouts={
        <>
          <Readout label="largest |ratio − 1| at k" value={at.maxDistortion.toFixed(3)} />
          <Readout
            label="JL theoretical bound ε"
            value={Number.isFinite(epsNow) ? epsNow.toFixed(3) : 'none (k < 24 ln n)'}
          />
          <Readout label="evaluated pairs" value={String(at.ratios.length)} />
          <Readout
            label={`k for ε = 0.25 (n = ${n})`}
            value={String(johnsonLindenstraussDimension(n, 0.25))}
          />
        </>
      }
      caption="Top-left: 2.5%–97.5% quantile band (shaded), median, and extremes of squared distance ratios after projection vs before over all pairs, alongside the theoretical 1 ± ε(k, n) envelope (dashed). Drag the vertical handle k to inspect the ratio histogram (bottom-left) and worst-case distortion (bottom-right). Top-right: Theoretical target dimension k as a function of n for ε ∈ {0.1, 0.25, 0.5}, illustrating logarithmic scaling in n."
    >
      <Plots rows={2} cols={2} heights={[55, 45]} widths={[60, 40]}>
        <Plot x={kAxis} y={ratioAxis} title="Distance ratio vs target dimension k">
          <Area
            name="2.5%–97.5% quantile band"
            x={rows.map((r) => r.k)}
            y={rows.map((r) => r.hi)}
            base={rows.map((r) => r.lo)}
            slot={0}
            opacity={0.2}
            line={false}
          />
          <Curve name="median ratio" x={rows.map((r) => r.k)} y={rows.map((r) => r.mid)} slot={0} />
          <Points
            name="min and max ratios"
            x={[...rows.map((r) => r.k), ...rows.map((r) => r.k)]}
            y={[...rows.map((r) => r.min), ...rows.map((r) => r.max)]}
            slot={0}
            size={6}
          />
          <Curve name="1 + ε (JL bound)" x={boundK} y={boundEps.map((e) => ok(1 + e) ?? NaN)} emphasis dashed width={1.5} />
          <Curve name="1 - ε (JL bound)" x={boundK} y={boundEps.map((e) => ok(1 - e) ?? NaN)} emphasis dashed width={1.5} />
          <Handle
            kind="x"
            at={k}
            label={`k = ${k}`}
            onDrag={(xVal) => setK(Math.max(4, Math.min(600, Math.round(xVal))))}
          />
        </Plot>
        <Plot x={nAxis} y={dimAxis} title="Theoretical JL dimension k vs n">
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
            name="current n"
            x={epsilons.map(() => n)}
            y={epsilons.map((e) => johnsonLindenstraussDimension(n, e))}
            emphasis
          />
        </Plot>
        <Plot x={histAxis} y={countAxis} title={`Pairwise ratios at k = ${k}`}>
          <Histogram values={Array.from(at.ratios)} name="pairs" slot={0} />
        </Plot>
        <Plot x={kAxis} y={worstAxis} title="Worst-case distortion vs JL bound">
          <Curve
            name="largest |ratio − 1|"
            x={rows.map((r) => r.k)}
            y={rows.map((r) => 1 + r.worst)}
            slot={1}
            showPoints
          />
          <Curve name="1 + ε (JL)" x={boundK} y={boundEps.map((e) => ok(1 + e) ?? NaN)} emphasis dashed width={1.5} />
        </Plot>
      </Plots>
    </Figure>
  )
}

export { RandomProjectionExplorer as DistortionHistogram }
