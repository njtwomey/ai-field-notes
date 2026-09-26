import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'

const log2 = (x: number) => Math.log(x) / Math.LN2
const entropy = (ps: number[]) => -ps.reduce((s, p) => (p > 0 ? s + p * log2(p) : s), 0)

/** Closed outline of the bar [x0, x1] at height row, drawn as a rectangle 0.6 tall. */
function bar(name: string, x0: number, x1: number, row: number, slot: number): XYSeries {
  const [y0, y1] = [row - 0.3, row + 0.3]
  return { name, type: 'line', x: [x0, x1, x1, x0, x0], y: [y0, y0, y1, y1, y0], slot }
}

/**
 * The information diagram for two binary variables. X ~ Bern(a); Y given X is Bern(b) or Bern(c). Every quantity is
 * a length along the same axis in bits, so the identities between them read off as sums of lengths.
 */
export function InformationDiagram() {
  const a = useParam(0.25, { min: 0.01, max: 0.99, step: 0.01 })
  const b = useParam(1 / 3, { min: 0, max: 1, step: 0.01 })
  const c = useParam(1, { min: 0, max: 1, step: 0.01 })

  const r = useMemo(() => {
    // Joint table p(x, y) for x, y ∈ {0, 1}.
    const joint = [
      [(1 - a.value) * (1 - b.value), (1 - a.value) * b.value],
      [a.value * (1 - c.value), a.value * c.value],
    ]
    const px = [joint[0][0] + joint[0][1], joint[1][0] + joint[1][1]]
    const py = [joint[0][0] + joint[1][0], joint[0][1] + joint[1][1]]
    const hx = entropy(px)
    const hy = entropy(py)
    const hxy = entropy(joint.flat())
    return { joint, hx, hy, hxy, hxGivenY: hxy - hy, hyGivenX: hxy - hx, mi: hx + hy - hxy }
  }, [a.value, b.value, c.value])

  const series: XYSeries[] = [
    bar('H(X, Y)', 0, r.hxy, 4, 3),
    bar('H(X)', 0, r.hx, 3, 4),
    bar('H(Y)', r.hxGivenY, r.hxy, 2, 5),
    bar('H(X | Y)', 0, r.hxGivenY, 1, 0),
    bar('I(X; Y)', r.hxGivenY, r.hx, 1, 1),
    bar('H(Y | X)', r.hx, r.hxy, 1, 2),
  ]

  return (
    <Interactive
      title="The information diagram"
      caption="X is a coin with P(X = 1) = a. Given X = 0, Y is a coin with P(Y = 1) = b; given X = 1, with P(Y = 1) = c. Each bar is a quantity in bits, laid on one axis. The bottom row splits the joint entropy H(X, Y) into what only X carries, what the two share, and what only Y carries. H(X) spans the first two pieces and H(Y) the last two, so I(X; Y) = H(X) + H(Y) − H(X, Y). Set b = c to make X and Y independent: the shared piece vanishes."
      controls={
        <>
          <ParamSlider label="a = P(X = 1)" param={a} />
          <ParamSlider label="b = P(Y = 1 | X = 0)" param={b} />
          <ParamSlider label="c = P(Y = 1 | X = 1)" param={c} />
        </>
      }
      readout={
        <>
          <Readout
            label="p(x, y)"
            value={`[${r.joint.map((row) => row.map((v) => formatNumber(v)).join(', ')).join(' | ')}]`}
          />
          <Readout label="H(X)" value={formatNumber(r.hx)} />
          <Readout label="H(Y)" value={formatNumber(r.hy)} />
          <Readout label="H(X, Y)" value={formatNumber(r.hxy)} />
          <Readout label="H(X | Y)" value={formatNumber(r.hxGivenY)} />
          <Readout label="H(Y | X)" value={formatNumber(r.hyGivenX)} />
          <Readout label="I(X; Y)" value={formatNumber(r.mi)} />
        </>
      }
    >
      <XYChart series={series} xLabel="bits" xRange={[0, 2]} yRange={[0.4, 4.6]} height={300} />
    </Interactive>
  )
}
