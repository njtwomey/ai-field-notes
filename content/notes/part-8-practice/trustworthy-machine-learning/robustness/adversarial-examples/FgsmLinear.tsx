import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
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
  const angle = useParam(30, { min: 0, max: 90, step: 1 })
  const eps = useParam(0.3, { min: 0, max: 1.5, step: 0.01 })
  const px = useParam(0.9, { min: -R, max: R, step: 0.01 })
  const py = useParam(0.4, { min: -R, max: R, step: 0.01 })

  const t = (angle.value * Math.PI) / 180
  const w: [number, number] = [Math.cos(t), Math.sin(t)]
  const s = w[0] * px.value + w[1] * py.value
  const label = s >= 0 ? 1 : -1
  const l1 = Math.abs(w[0]) + Math.abs(w[1])
  const adv: [number, number] = [
    px.value - eps.value * label * Math.sign(w[0]),
    py.value - eps.value * label * Math.sign(w[1]),
  ]
  const sAdv = w[0] * adv[0] + w[1] * adv[1]

  // The boundary w·x = 0 runs along (−w₂, w₁); draw it across the whole plot.
  const boundary = { x: [-w[1] * 2 * R, w[1] * 2 * R], y: [w[0] * 2 * R, -w[0] * 2 * R] }
  const b = box(px.value, py.value, eps.value)
  const series: XYSeries[] = [
    { name: 'decision boundary', type: 'line', x: boundary.x, y: boundary.y, emphasis: true },
    { name: 'ℓ∞ ball', type: 'line', x: b.x, y: b.y, slot: 0 },
    { name: 'input x', type: 'scatter', x: [px.value], y: [py.value], emphasis: true },
    { name: 'FGSM point', type: 'scatter', x: [adv[0]], y: [adv[1]], slot: 1 },
  ]
  const handles: Handle[] = [
    {
      kind: 'point',
      at: [px.value, py.value],
      label: 'input',
      onDrag: ([x, y]) => {
        px.set(x)
        py.set(y)
      },
    },
  ]

  return (
    <Interactive
      title="FGSM on a linear classifier"
      caption="The line is the decision boundary of the score s(x) = w·x, with w at the chosen angle. Drag the input; the square is the ℓ∞ ball of radius ε around it. FGSM moves to the corner in the direction −y·sgn(w), which lowers the margin by ε‖w‖₁, the most any point in the square can. The prediction flips once ε exceeds the ℓ∞ distance |s|/‖w‖₁, which is smaller than the Euclidean distance |s|/‖w‖₂ unless w lies along an axis."
      controls={
        <>
          <ParamSlider label="ε" param={eps} />
          <ParamSlider label="angle of w (degrees)" param={angle} />
        </>
      }
      readout={
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
        <XYChart
          series={series}
          vectors={[{ from: [px.value, py.value], to: adv }]}
          handles={handles}
          xRange={[-R, R]}
          yRange={[-R, R]}
          equalAspect
          xLabel="x₁"
          yLabel="x₂"
          ariaLabel="A linear decision boundary, an input point with its ℓ∞ ball and the FGSM perturbation"
        />
      </div>
    </Interactive>
  )
}
