import { useMemo } from 'react'
import { choice, Curve, Figure, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { FILTERS, VANISHING_MOMENTS, cascade, type Family } from '../_shared/wavelets'

/**
 * Scaling function φ and wavelet ψ of the Haar and Daubechies families, computed by the cascade algorithm: iterating
 * the two-scale equation from a unit impulse.
 */
export function CascadeShapes() {
  const state = useFigureState({
    family: choice<Family>(
      [
        { value: 'haar', label: 'Haar (db1)' },
        { value: 'db2', label: 'db2' },
        { value: 'db3', label: 'db3' },
        { value: 'db4', label: 'db4' },
      ],
      'db2',
      { label: 'family' },
    ),
    iterations: int(7, { min: 1, max: 9, step: 1, label: 'cascade iterations', format: (v) => String(v) }),
  })

  const r = useMemo(() => cascade(FILTERS[state.family], state.iterations), [state.family, state.iterations])
  const L = FILTERS[state.family].length

  const xAxis = useAxis({ label: 't', hold: 'union' })
  const yAxis = useAxis({ label: 'φ(t)', hold: 'union' })
  const xAxis2 = useAxis({ label: 't', hold: 'union' })
  const yAxis2 = useAxis({ label: 'ψ(t)', hold: 'union' })
  return (
    <Figure
      title="Daubechies scaling functions and wavelets"
      state={state}
      caption="The scaling function φ and wavelet ψ, drawn by the cascade algorithm: start from a single impulse and repeatedly upsample and filter with √2·h, which iterates the two-scale equation. After a few iterations the samples settle on the limit functions, supported on [0, L − 1] for a filter of length L. Haar gives boxes. Daubechies wavelets with more vanishing moments are longer and smoother, but none of them is symmetric. The db2 functions are continuous but have no continuous derivative: their Hölder exponent is about 0.55."

      readouts={
        <>
          <Readout label="filter length L" value={L} />
          <Readout label="support" value={`[0, ${L - 1}]`} />
          <Readout label="vanishing moments" value={VANISHING_MOMENTS[state.family]} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={260}>
          <Curve name="φ (scaling function)" x={r.t} y={r.phi} slot={0} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={260}>
          <Curve name="ψ (wavelet)" x={r.t} y={r.psi} slot={1} />
        </Plot>
      </div>
    </Figure>
  )
}
