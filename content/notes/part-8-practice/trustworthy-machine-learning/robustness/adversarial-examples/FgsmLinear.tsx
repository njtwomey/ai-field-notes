import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
  Vectors,
} from 'aifn-render'

const R = 2.5

/** Corners of the ℓ∞ ball of radius eps around (x, y), closed so it draws as one line. */
function box(x: number, y: number, eps: number) {
  return {
    x: [x - eps, x + eps, x + eps, x - eps, x - eps],
    y: [y - eps, y - eps, y + eps, y + eps, y - eps],
  }
}

/**
 * A linear classifier s(x) = w·x in two dimensions. The FGSM step x − ε·y·sgn(w) goes to the corner of the ℓ∞ box that
 * lowers the margin most; it reaches the boundary exactly when ε ≥ |s(x)| / ‖w‖₁.
 */
export function FgsmLinear() {
  const state = useFigureState({
    eps: float(0.3, { min: 0, max: 1.5, step: 0.01, label: 'ε' }),
    angle: slider(0, 90, 30, { step: 1, label: 'angle of w (degrees)' }),
    px: slider(-R, R, 0.9, { step: 0.01, onChart: true }),
    py: slider(-R, R, 0.4, { step: 0.01, onChart: true }),
  })

  const t = (state.angle * Math.PI) / 180
  const w: [number, number] = [Math.cos(t), Math.sin(t)]
  const s = w[0] * state.px + w[1] * state.py
  const label = s >= 0 ? 1 : -1
  const l1 = Math.abs(w[0]) + Math.abs(w[1])
  const adv: [number, number] = [
    state.px - state.eps * label * Math.sign(w[0]),
    state.py - state.eps * label * Math.sign(w[1]),
  ]
  const sAdv = w[0] * adv[0] + w[1] * adv[1]

  // The boundary w·x = 0 runs along (−w₂, w₁); draw it across the whole plot.
  const boundary = { x: [-w[1] * 2 * R, w[1] * 2 * R], y: [w[0] * 2 * R, -w[0] * 2 * R] }
  const b = box(state.px, state.py, state.eps)
  const series = [
    { name: 'decision boundary', x: boundary.x, y: boundary.y, emphasis: true },
    { name: 'ℓ∞ ball', x: b.x, y: b.y, slot: 0 },
    { name: 'input x', x: [state.px], y: [state.py], emphasis: true },
    { name: 'FGSM point', x: [adv[0]], y: [adv[1]], slot: 1 },
  ] as const

  const xAxis = useAxis({ label: 'x₁', range: [-R, R] })
  const yAxis = useAxis({ label: 'x₂', range: [-R, R], equal: xAxis })
  return (
    <Figure
      title="FGSM on a linear classifier"
      purpose="Drag the input and change ε and the weight angle to see where the fast gradient sign step moves it and whether the label flips."
      state={state}
      caption="The line is the decision boundary of the score s(x) = w·x, with w at the chosen angle. Drag the input; the square is the ℓ∞ ball of radius ε around it. FGSM moves to the corner in the direction −y·sgn(w), which lowers the margin by ε‖w‖₁, the most any point in the square can. The prediction flips once ε exceeds the ℓ∞ distance |s|/‖w‖₁, which is smaller than the Euclidean distance |s|/‖w‖₂ unless w lies along an axis."

      readouts={
        <>
          <Readout label="score s(x)" value={formatNumber(s)} />
          <Readout label="score after FGSM" value={formatNumber(sAdv)} />
          <Readout label="ℓ∞ distance |s|/‖w‖₁" value={formatNumber(Math.abs(s) / l1)} />
          <Readout label="ℓ₂ distance |s|/‖w‖₂" value={formatNumber(Math.abs(s))} />
          <Readout label="prediction flipped" value={Math.sign(sAdv) !== label && sAdv !== 0 ? 'yes' : 'no'} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-md">
        <Plot
          x={xAxis}
          y={yAxis}
          ariaLabel={'A linear decision boundary, an input point with its ℓ∞ ball and the FGSM perturbation'}
        >
          <Curve {...series[0]} />
          <Curve {...series[1]} />
          <Points {...series[2]} />
          <Points {...series[3]} />
          <Vectors vectors={[{ from: [state.px, state.py], to: adv }]} />
          <Handle
            kind="point"
            at={[state.px, state.py]}
            label="input"
            onDrag={([x, y]) => {
              state.set('px', x)
              state.set('py', y)
            }}
          />
        </Plot>
      </div>
    </Figure>
  )
}
