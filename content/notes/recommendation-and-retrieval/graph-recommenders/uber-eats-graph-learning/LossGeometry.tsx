import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'

const hinge = (x: number) => Math.max(0, x)

/**
 * The max-margin loss with low-rank positives for one user with a heavily ordered dish v, a lightly ordered dish l and
 * a never-ordered dish n. Scores s = z_u · z_item. Edge (u, v): hinge(s_n − s_v + γ1) + λ·hinge(s_l − s_v + γ2).
 * Edge (u, l): hinge(s_n − s_l + γ1); it has no lighter positive.
 */
export function LossGeometry() {
  const sv = useParam(1.2, { min: -2, max: 2, step: 0.01 })
  const sl = useParam(0.9, { min: -2, max: 2, step: 0.01 })
  const sn = useParam(-0.2, { min: -2, max: 2, step: 0.01 })
  const g1 = useParam(1, { min: 0.1, max: 2, step: 0.05 })
  const g2 = useParam(0.4, { min: 0, max: 2, step: 0.05 })
  const lambda = useParam(0.5, { min: 0, max: 2, step: 0.05 })

  const negV = hinge(sn.value - sv.value + g1.value)
  const lowV = hinge(sl.value - sv.value + g2.value)
  const negL = hinge(sn.value - sl.value + g1.value)
  const total = negV + lambda.value * lowV + negL

  // Margin zones drawn as horizontal bars: an item inside a zone is too close to the positive above it.
  const zone = (from: number, to: number, y: number) => ({
    from: [from, y] as [number, number],
    to: [to, y] as [number, number],
  })
  const series: XYSeries[] = [
    { name: 'heavily ordered dish v', type: 'scatter', x: [sv.value], y: [0.8], slot: 0 },
    { name: 'lightly ordered dish l', type: 'scatter', x: [sl.value], y: [0.8], slot: 1 },
    { name: 'never-ordered dish n', type: 'scatter', x: [sn.value], y: [0.8], slot: 2 },
    {
      name: 'margin γ₂ below v (for l)',
      type: 'line',
      x: [sv.value - g2.value, sv.value],
      y: [0.55, 0.55],
      slot: 1,
    },
    {
      name: 'margin γ₁ below v (for n)',
      type: 'line',
      x: [sv.value - g1.value, sv.value],
      y: [0.4, 0.4],
      slot: 2,
    },
    {
      name: 'margin γ₁ below l (for n)',
      type: 'line',
      x: [sl.value - g1.value, sl.value],
      y: [0.25, 0.25],
      slot: 2,
      dashed: true,
    },
  ]

  return (
    <Interactive
      title="Margins for negatives and low-rank positives"
      caption="Scores of three dishes for one user on a line: a dish ordered often (v), a dish ordered rarely (l) and a dish never ordered (n). Bars show the margin each score must clear: l must sit at least γ₂ below v, and n at least γ₁ below both v and l. Any item inside a bar adds a hinge term. Drag the three scores; with γ₂ < γ₁ the loss is zero only for the order v > l > n with gaps, so heavily ordered dishes rank above lightly ordered ones, which rank above the rest. After Liu et al. (2019)."
      controls={
        <>
          <ParamSlider label="negative margin γ₁" param={g1} />
          <ParamSlider label="low-rank margin γ₂" param={g2} />
          <ParamSlider label="low-rank weight λ" param={lambda} />
        </>
      }
      readout={
        <>
          <Readout label="(u,v) vs n" value={formatNumber(negV)} />
          <Readout label="λ × (u,v) vs l" value={formatNumber(lambda.value * lowV)} />
          <Readout label="(u,l) vs n" value={formatNumber(negL)} />
          <Readout label="total loss" value={formatNumber(total)} />
        </>
      }
    >
      <XYChart
        series={series}
        segments={[zone(-2.5, 2.5, 0.8)]}
        xLabel="score s = z_u · z_dish"
        yRange={[0, 1]}
        xRange={[-2.5, 2.5]}
        height={260}
        handles={[
          { kind: 'x', at: sv.value, label: 'v', onDrag: (x) => sv.set(x) },
          { kind: 'x', at: sl.value, label: 'l', onDrag: (x) => sl.set(x) },
          { kind: 'x', at: sn.value, label: 'n', onDrag: (x) => sn.set(x) },
        ]}
      />
    </Interactive>
  )
}
