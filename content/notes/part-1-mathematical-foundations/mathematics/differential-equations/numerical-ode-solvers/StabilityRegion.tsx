import { useMemo } from 'react'
import { amplification, stabilityRegion, type StabilityMethod } from 'aifn/dynamics/ode'
import { toFlat, toRows } from 'aifn/foundation/tensor'
import {
  choice,
  Contours,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Raster,
  Readout,
  row,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'

const fmt = (v: number) => formatNumber(v)

const STABILITY_LEVELS = [0]
const STABILITY_METHODS = [
  { value: 'euler', label: 'explicit Euler' },
  { value: 'heun', label: 'Heun (order 2)' },
  { value: 'rk4', label: 'RK4 (order 4)' },
  { value: 'dormand-prince', label: 'Dormand–Prince (order 5)' },
  { value: 'implicit-euler', label: 'implicit Euler (A-stable)' },
  { value: 'implicit-trapezoid', label: 'implicit trapezoid (A-stable)' },
  { value: 'bdf2', label: 'BDF2 (A-stable)' },
  { value: 'bdf3', label: 'BDF3 (stiff-stable)' },
] as const

export function StabilityRegion() {
  const state = useFigureState({
    method: row('Method', { method: choice(STABILITY_METHODS, 'rk4', { label: 'solver' }) }),
    zr: slider(-6, 3, -2, { onChart: true, label: 'Re z' }),
    zi: slider(-4.5, 4.5, 1.5, { onChart: true, label: 'Im z' }),
  })

  const method = state.method.method
  const z: [number, number] = [state.zr, state.zi]

  const region = useMemo(
    () => stabilityRegion(method as StabilityMethod, { real: [-6, 3], imag: [-4.5, 4.5], nx: 91, ny: 91 }),
    [method],
  )

  const grid = useMemo(() => {
    const rows = toRows(region.amplification).map((r) => r.map((v) => Math.max(-1.5, Math.min(1.5, Math.log10(v)))))
    return { x: toFlat(region.real), y: toFlat(region.imag), z: rows }
  }, [region])

  const a = amplification(method as StabilityMethod, z[0], z[1])
  const re = useAxis({ label: 'Re z = Re(hλ)' })
  const im = useAxis({ label: 'Im z = Im(hλ)', equal: re })

  return (
    <Figure
      title="Linear stability regions in the complex plane"
      purpose="A solver applied to ẋ = λx is stable when z = hλ lies where amplification |R(z)| ≤ 1 (blue). Explicit methods have bounded regions; A-stable implicit methods encompass the entire left half-plane Re(z) ≤ 0."
      state={state}
      readouts={{
        atZ: (
          <>
            <Readout label="z = hλ" value={`${fmt(z[0])} ${z[1] < 0 ? '−' : '+'} ${fmt(Math.abs(z[1]))}i`} />
            <Readout label="amplification |R(z)|" value={fmt(a)} />
            <Readout label="linearly stable" value={a <= 1 + 1e-12 ? 'yes' : 'no'} />
          </>
        ),
      }}
      caption="Colour: log₁₀ of the amplification factor |R(z)|, clipped to ±1.5. Blue region (|R| ≤ 1) indicates stability; red (|R| > 1) indicates exponential growth/divergence. The contour is the stability boundary |R| = 1. Drag point z to probe the response. Notice that implicit Euler and trapezoid cover the entire left half-plane (A-stability), allowing arbitrarily large steps on stiff decay modes."
    >
      <Plot x={re} y={im}>
        <Raster
          x={grid.x}
          y={grid.y}
          z={grid.z}
          scale="diverging"
          range={[-1.5, 1.5]}
          valueLabel="log₁₀ amplification"
        />
        <Contours x={grid.x} y={grid.y} z={grid.z} levels={STABILITY_LEVELS} />
        <Handle {...state.handle(['zr', 'zi'], { label: 'z' })} />
      </Plot>
    </Figure>
  )
}
