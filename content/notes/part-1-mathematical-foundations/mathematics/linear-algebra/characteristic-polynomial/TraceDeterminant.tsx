import { useContext, useMemo, type ReactNode } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  FrameContext,
  Handle,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'

const SPAN = 4
const PANEL = 300
const LAMBDA = Array.from({ length: 161 }, (_, i) => -SPAN + (2 * SPAN * i) / 160)
const TRACE = Array.from({ length: 161 }, (_, i) => -SPAN + (2 * SPAN * i) / 160)

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

/**
 * A 2 × 2 matrix's characteristic polynomial is λ² − tλ + d, fixed by the trace t and determinant d alone. Dragging
 * (t, d) across the parabola d = t²/4 turns two real eigenvalues into a complex-conjugate pair.
 */
export function TraceDeterminant() {
  const state = useFigureState({
    t: slider(-SPAN, SPAN, 1, { step: 0.05, label: 'trace t', onChart: true }),
    d: slider(-SPAN, SPAN, -2, { step: 0.05, label: 'determinant d', onChart: true }),
  })
  const { t, d } = state

  const disc = t * t - 4 * d
  const roots = disc >= 0 ? [(t - Math.sqrt(disc)) / 2, (t + Math.sqrt(disc)) / 2] : []
  const eig =
    disc >= 0
      ? roots.map((r) => formatNumber(r)).join(', ')
      : `${formatNumber(t / 2)} ± ${formatNumber(Math.sqrt(-disc) / 2)}i`

  const parabola = useMemo(() => TRACE.map((x) => (x * x) / 4), [])
  const poly = useMemo(() => LAMBDA.map((l) => l * l - t * l + d), [t, d])
  const zero = useMemo(() => LAMBDA.map(() => 0), [])

  const tx = useAxis({ label: 'trace t', range: [-SPAN, SPAN] })
  const ty = useAxis({ label: 'determinant d', range: [-SPAN, SPAN] })
  const lx = useAxis({ label: 'λ', range: [-SPAN, SPAN] })
  const ly = useAxis({ label: 'p(λ) = λ² − tλ + d', range: [-6, 10] })

  return (
    <Figure
      title="Eigenvalues from the trace and determinant"
      state={state}
      caption="Every 2 × 2 matrix with trace t and determinant d has the characteristic polynomial p(λ) = λ² − tλ + d, so its eigenvalues depend on (t, d) alone. Left: drag the point in the trace–determinant plane. Below the parabola d = t²/4 the discriminant t² − 4d is positive and there are two real eigenvalues; on it they coincide; above it they form a complex-conjugate pair with real part t/2. Right: the polynomial, with its real roots marked. Moving d shifts the parabola p up and down; once it clears the axis, the real roots are gone. The companion matrix in the readouts is one matrix with this polynomial."
      readouts={
        <>
          <Readout label="eigenvalues" value={eig} />
          <Readout label="discriminant t² − 4d" value={formatNumber(disc)} />
          <Readout label="companion matrix" value={`[[0, ${formatNumber(-d)}], [1, ${formatNumber(t)}]]`} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <FixedHeight height={PANEL}>
          <Plot x={tx} y={ty} title="trace–determinant plane">
            <Curve name="d = t²/4 (repeated eigenvalue)" x={TRACE} y={parabola} emphasis />
            <Handle {...state.handle(['t', 'd'], { label: `(t, d) = (${formatNumber(t)}, ${formatNumber(d)})` })} />
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={lx} y={ly} title="characteristic polynomial">
            <Curve name="zero" x={LAMBDA} y={zero} muted dashed silent />
            <Curve name="p(λ)" x={LAMBDA} y={poly} emphasis />
            {roots.length > 0 && <Points name="eigenvalues" x={roots} y={[0, 0]} emphasis size={9} live />}
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
