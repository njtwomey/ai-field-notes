import { useMemo } from 'react'
import { henonMap, lyapunovSpectrum, orbit, standardMap, type MapN } from 'aifn-methods/dynamics/maps'
import { histogram } from 'aifn/probability/stats'
import { toFlat, toRows } from 'aifn/foundation/tensor'
import {
  ControlGroup,
  ControlRow,
  Figure,
  formatNumber,
  Player,
  Plot,
  Points,
  Raster,
  Readout,
  slider,
  useAxis,
  useFigureState,
  usePlayhead,
  variants,
} from 'aifn-render'

const fmt = (v: number) => formatNumber(v)

function visits(
  xs: readonly number[],
  ys: readonly number[],
  xr: [number, number],
  yr: [number, number],
  nx: number,
  ny: number,
) {
  const dy = (yr[1] - yr[0]) / ny
  const rows: number[][] = []
  for (let i = 0; i < ny; i++) {
    const lo = yr[0] + i * dy
    const inRow = xs.filter((_, k) => ys[k] >= lo && ys[k] < lo + dy)
    const h = histogram(inRow.length ? inRow : [NaN], { bins: nx, range: xr })
    rows.push(toFlat(h.counts).map((c) => Math.log1p(c)))
  }
  const cx = Array.from({ length: nx }, (_, j) => xr[0] + ((j + 0.5) * (xr[1] - xr[0])) / nx)
  const cy = Array.from({ length: ny }, (_, i) => yr[0] + (i + 0.5) * dy)
  return { x: cx, y: cy, z: rows }
}

type PlaneMap = {
  label: string
  param: { label: string; min: number; max: number; initial: number }
  make: (p: number) => MapN
  xr: [number, number]
  yr: [number, number]
  starts: number[][]
}

const TAU = 2 * Math.PI
const REVEAL = 1000

const PLANE_MAPS = {
  henon: {
    label: 'Hénon map',
    param: { label: 'a (with b = 0.3)', min: 1, max: 1.42, initial: 1.4 },
    make: (a) => henonMap(a, 0.3),
    xr: [-1.5, 1.5],
    yr: [-0.45, 0.45],
    starts: [[0.1, 0.1]],
  },
  standard: {
    label: 'Chirikov standard map',
    param: { label: 'stochasticity parameter K', min: 0, max: 2.5, initial: 0.97 },
    make: standardMap,
    xr: [0, TAU],
    yr: [0, TAU],
    starts: Array.from({ length: 40 }, (_, k) => [Math.PI + 0.01 * k, (TAU * (k + 0.5)) / 40]),
  },
} satisfies Record<string, PlaneMap>
type PlaneName = keyof typeof PLANE_MAPS

const PLANE = variants(
  Object.fromEntries(
    (Object.keys(PLANE_MAPS) as PlaneName[]).map((k) => {
      const m: PlaneMap = PLANE_MAPS[k]
      return [
        k,
        {
          label: m.label,
          params: { p: slider(m.param.min, m.param.max, m.param.initial, { label: m.param.label, step: 0.01 }) },
        },
      ]
    }),
  ) as Record<PlaneName, { label: string; params: { p: ReturnType<typeof slider> } }>,
  { label: 'Map of the plane', choiceLabel: 'map' },
)

export function PlaneMaps() {
  const state = useFigureState({ map: PLANE })
  const which = state.map.key as PlaneName
  const pm: PlaneMap = PLANE_MAPS[which]
  const p = state.map.values.p as number
  const map = useMemo(() => pm.make(p), [pm, p])

  const picture = useMemo(() => {
    const per = Math.floor(30_000 / pm.starts.length)
    const xs: number[] = []
    const ys: number[] = []
    for (const s of pm.starts) {
      const pts = toRows(orbit(map, s, per, { discard: which === 'henon' ? 100 : 0 }))
      for (const [a, b] of pts) {
        xs.push(a)
        ys.push(b)
      }
    }
    return visits(xs, ys, pm.xr, pm.yr, 220, which === 'henon' ? 120 : 220)
  }, [map, pm, which])

  const spectrum = useMemo(() => toFlat(lyapunovSpectrum(map, pm.starts[0], { keep: 3000 }).exponents), [map, pm])

  const first = useMemo(() => {
    const rows = toRows(orbit(map, pm.starts[0], REVEAL))
    return { x: rows.map((q) => q[0]), y: rows.map((q) => q[1]) }
  }, [map, pm])

  const [k, setK] = usePlayhead(REVEAL + 1)
  const so = { x: first.x.slice(0, k + 1), y: first.y.slice(0, k + 1) }

  const xa = useAxis({ label: which === 'henon' ? 'x' : 'position θ', range: pm.xr, nice: false })
  const ya = useAxis({ label: which === 'henon' ? 'y' : 'momentum p', range: pm.yr, nice: false })

  return (
    <Figure
      title="Chaotic maps of the plane: Hénon and standard map"
      purpose="Dissipative maps like the Hénon attractor contract phase space volume onto a fractal strange attractor. Area-preserving maps like the Chirikov standard map break invariant KAM tori into a chaotic sea."
      state={state}
      defaultSize="L"
      controls={
        <ControlGroup title="Playback" collapsible={false}>
          <ControlRow label="Orbit animation">
            <Player
              className="col-span-full"
              value={k}
              onChange={setK}
              count={REVEAL + 1}
              format={(i) => `iteration n = ${i}`}
              label="iteration"
            />
          </ControlRow>
        </ControlGroup>
      }
      readouts={{
        spectrum: (
          <>
            <Readout label="Lyapunov spectrum (λ₁, λ₂)" value={spectrum.map(fmt).join(', ')} />
            <Readout label="trace log|det J| (sum)" value={fmt(spectrum[0] + spectrum[1])} />
            <Readout
              label="dissipative vs Hamiltonian"
              value={which === 'henon' ? 'dissipative (det J = −b)' : 'symplectic (det J = 1)'}
            />
          </>
        ),
      }}
      caption="30,000 iterates in phase space (one orbit for Hénon; 40 orbits started along θ = π for the standard map). Play the iteration slider to watch the trajectory advance step by step. The Hénon orbit collapses onto the Cantor-like strange attractor; the standard map transitions from smooth invariant KAM curves to chaotic diffusion as K increases."
    >
      <Plot x={xa} y={ya}>
        <Raster
          x={picture.x}
          y={picture.y}
          z={picture.z}
          valueLabel="log(1 + visits)"
          colorBar={false}
          fillOpacity={0.6}
        />
        <Points name="first orbit so far" x={so.x} y={so.y} slot={7} thin live />
        <Points name="current iterate" x={[first.x[k]]} y={[first.y[k]]} emphasis live />
      </Plot>
    </Figure>
  )
}
