import { useMemo, useState } from 'react'
import { rateDistortionCurve } from 'aifn-applied/information/channels'
import { binaryEntropy } from 'aifn/numerics/special'
import { toFlat } from 'aifn/foundation/tensor'
import {
  ControlRow,
  Curve,
  Figure,
  formatNumber,
  Plot,
  Points,
  Readout,
  Slider,
  useAxis,
} from 'aifn-render'

const h2 = (p: number) => binaryEntropy(p, 2)

export function RateDistortionExplorer() {
  const [p, setP] = useState(0.3)

  const curve = useMemo(() => {
    const betas = Array.from({ length: 60 }, (_, i) => 0.05 * 1.12 ** i)
    return rateDistortionCurve(
      [1 - p, p],
      [
        [0, 1],
        [1, 0],
      ],
      betas,
      { base: 2 },
    )
  }, [p])

  const closed = useMemo(() => {
    const maxD = Math.min(p, 1 - p)
    const ds = Array.from({ length: 200 }, (_, i) => (maxD * i) / 199)
    return {
      x: ds,
      y: ds.map((d) => Math.max(0, h2(p) - h2(d))),
    }
  }, [p])

  const points = useMemo(
    () => ({
      x: toFlat(curve.distortion),
      y: toFlat(curve.rate),
    }),
    [curve],
  )

  const da = useAxis({ label: 'distortion budget D', range: [0, Math.min(p, 1 - p) * 1.05] })
  const ra = useAxis({ label: 'minimum rate R(D) (bits)', range: [0, h2(p) * 1.05] })

  const entropyH = h2(p)

  return (
    <Figure
      title="Rate–distortion curve of a binary source under Hamming distortion"
      purpose="Explore the fundamental trade-off between lossy compression rate R and fidelity distortion D using the Blahut–Arimoto algorithm."
      caption="Each scatter point is computed by the Blahut–Arimoto rate-distortion algorithm at Lagrange multiplier slope -β. The curve matches Shannon's analytical solution R(D) = H_2(p) - H_2(D) for binary memoryless sources up to D_max = min(p, 1 - p). At zero distortion D = 0, lossy compression recovers the lossless entropy rate R = H(p)."
    >
      <ControlRow>
        <Slider
          label="Bernoulli source parameter p = P(X = 1)"
          value={p}
          min={0.02}
          max={0.5}
          step={0.01}
          onChange={setP}
        />
      </ControlRow>

      <div className="flex flex-wrap gap-4 text-xs font-mono text-muted-foreground my-2">
        <Readout label="lossless rate R(0) = H(p)" value={`${formatNumber(entropyH)} bits`} />
        <Readout label="max distortion D_max" value={formatNumber(Math.min(p, 1 - p))} />
        <Readout label="evaluated points" value={points.x.length} />
        <Readout label="converged" value={String(curve.converged)} />
      </div>

      <Plot x={da} y={ra} title="Rate-distortion trade-off R(D)">
        <Curve name="Analytical R(D) = H(p) − H(D)" x={closed.x} y={closed.y} slot={1} />
        <Points name="Blahut–Arimoto numerical points" x={points.x} y={points.y} slot={0} size={5} />
      </Plot>
    </Figure>
  )
}
