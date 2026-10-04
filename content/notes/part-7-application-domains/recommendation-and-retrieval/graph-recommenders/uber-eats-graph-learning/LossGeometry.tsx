import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  Segments,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'

const hinge = (x: number) => Math.max(0, x)

/**
 * The max-margin loss with low-rank positives for one user with a heavily ordered dish v, a lightly ordered dish l and
 * a never-ordered dish n. Scores s = z_u · z_item. Edge (u, v): hinge(s_n − s_v + γ1) + λ·hinge(s_l − s_v + γ2).
 * Edge (u, l): hinge(s_n − s_l + γ1); it has no lighter positive.
 */
export function LossGeometry() {
  const state = useFigureState({
    g1: float(1, { min: 0.1, max: 2, step: 0.05, label: 'negative margin γ₁' }),
    g2: float(0.4, { min: 0, max: 2, step: 0.05, label: 'low-rank margin γ₂' }),
    lambda: float(0.5, { min: 0, max: 2, step: 0.05, label: 'low-rank weight λ' }),
    sv: slider(-2, 2, 1.2, { step: 0.01, onChart: true }),
    sl: slider(-2, 2, 0.9, { step: 0.01, onChart: true }),
    sn: slider(-2, 2, -0.2, { step: 0.01, onChart: true }),
  })

  const negV = hinge(state.sn - state.sv + state.g1)
  const lowV = hinge(state.sl - state.sv + state.g2)
  const negL = hinge(state.sn - state.sl + state.g1)
  const total = negV + state.lambda * lowV + negL

  // Margin zones drawn as horizontal bars: an item inside a zone is too close to the positive above it.
  const zone = (from: number, to: number, y: number) => ({
    from: [from, y] as [number, number],
    to: [to, y] as [number, number],
  })
  const series = [
    { name: 'heavily ordered dish v', x: [state.sv], y: [0.8], slot: 0 },
    { name: 'lightly ordered dish l', x: [state.sl], y: [0.8], slot: 1 },
    { name: 'never-ordered dish n', x: [state.sn], y: [0.8], slot: 2 },
    {
      name: 'margin γ₂ below v (for l)',
      x: [state.sv - state.g2, state.sv],
      y: [0.55, 0.55],
      slot: 1,
    },
    {
      name: 'margin γ₁ below v (for n)',
      x: [state.sv - state.g1, state.sv],
      y: [0.4, 0.4],
      slot: 2,
    },
    {
      name: 'margin γ₁ below l (for n)',
      x: [state.sl - state.g1, state.sl],
      y: [0.25, 0.25],
      slot: 2,
      dashed: true,
    },
  ] as const

  const xAxis = useAxis({ label: 'score s = z_u · z_dish', range: [-2.5, 2.5] })
  const yAxis = useAxis({ range: [0, 1] })
  return (
    <Figure
      title="Margins for negatives and low-rank positives"
      state={state}
      caption="Scores of three dishes for one user on a line: a dish ordered often (v), a dish ordered rarely (l) and a dish never ordered (n). Bars show the margin each score must clear: l must sit at least γ₂ below v, and n at least γ₁ below both v and l. Any item inside a bar adds a hinge term. Drag the three scores; with γ₂ < γ₁ the loss is zero only for the order v > l > n with gaps, so heavily ordered dishes rank above lightly ordered ones, which rank above the rest. After Liu et al. (2019)."

      readouts={
        <>
          <Readout label="(u,v) vs n" value={formatNumber(negV)} />
          <Readout label="λ × (u,v) vs l" value={formatNumber(state.lambda * lowV)} />
          <Readout label="(u,l) vs n" value={formatNumber(negL)} />
          <Readout label="total loss" value={formatNumber(total)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={260}>
        <Points {...series[0]} />
        <Points {...series[1]} />
        <Points {...series[2]} />
        <Curve {...series[3]} />
        <Curve {...series[4]} />
        <Curve {...series[5]} />
        <Segments segments={[zone(-2.5, 2.5, 0.8)]} />
        <Handle {...state.handle('sv', { label: 'v' })} />
        <Handle {...state.handle('sl', { label: 'l' })} />
        <Handle {...state.handle('sn', { label: 'n' })} />
      </Plot>
    </Figure>
  )
}
