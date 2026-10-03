import { useMemo } from 'react'
import {
  bifurcationDiagram,
  cobweb,
  logisticMap,
  lyapunovCurve,
  lyapunovExponent,
  orbit,
  sineMap,
  tentMap,
  type Map1,
} from 'aifn-applied/dynamics/maps'
import { histogram } from 'aifn/probability/stats'
import { toFlat } from 'aifn/foundation/tensor'
import {
  ControlGroup,
  ControlRow,
  Curve,
  Dashboard,
  DashboardCell,
  DashboardRow,
  Figure,
  formatNumber,
  Handle,
  Player,
  Plot,
  Plots,
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

type Family = { label: string; make: (r: number) => Map1; range: [number, number]; initial: number }

const FAMILIES = {
  logistic: { label: 'logistic r · x(1 − x)', make: logisticMap, range: [2.5, 4], initial: 3.56 },
  sine: { label: 'sine r · sin(πx)', make: sineMap, range: [0.6, 1], initial: 0.87 },
  tent: { label: 'tent μ · min(x, 1 − x)', make: tentMap, range: [1, 2], initial: 1.5 },
} satisfies Record<string, Family>
type FamilyName = keyof typeof FAMILIES

const FAMILY = variants(
  Object.fromEntries(
    (Object.keys(FAMILIES) as FamilyName[]).map((k) => {
      const f: Family = FAMILIES[k]
      return [
        k,
        {
          label: f.label,
          params: { r: slider(f.range[0], f.range[1], f.initial, { label: 'parameter r', step: 0.001 }) },
        },
      ]
    }),
  ) as Record<FamilyName, { label: string; params: { r: ReturnType<typeof slider> } }>,
  { label: 'Map family', choiceLabel: 'map' },
)

const NR = 300
const NX = 200
const COBWEB_N = 200
const SWEEP = 301

export function BifurcationDiagram() {
  const state = useFigureState({
    family: FAMILY,
    x0: slider(0, 1, 0.2, { label: 'initial state x₀', step: 0.001, onChart: true }),
  })

  const which = state.family.key as FamilyName
  const family: Family = FAMILIES[which]
  const r = state.family.values.r as number
  const setR = (v: number) => state.set('family.r', v)
  const x0 = state.x0
  const [n, setN] = usePlayhead(COBWEB_N + 1)

  const rs = useMemo(
    () => Array.from({ length: NR }, (_, i) => family.range[0] + ((family.range[1] - family.range[0]) * (i + 0.5)) / NR),
    [family],
  )

  const diagram = useMemo(() => {
    const d = bifurcationDiagram(family.make, rs, { x0: 0.2345, transient: 400, keep: 300 })
    const rr = toFlat(d.r)
    const xx = toFlat(d.x)
    const columns = rs
      .map((rv) =>
        histogram(
          xx.filter((_, k) => rr[k] === rv),
          { bins: NX, range: [0, 1] },
        ).counts,
      )
      .map((c) => toFlat(c))
    const z = Array.from({ length: NX }, (_, i) => columns.map((c) => Math.log1p(c[i])))
    const y = Array.from({ length: NX }, (_, i) => (i + 0.5) / NX)
    return { x: rs, y, z }
  }, [family, rs])

  const lyap = useMemo(() => toFlat(lyapunovCurve(family.make, rs, { x0: 0.2345, keep: 600 })), [family, rs])
  const map = useMemo(() => family.make(r), [family, r])

  const web = useMemo(() => {
    const w = cobweb(map, x0, COBWEB_N)
    return { x: toFlat(w.x), y: toFlat(w.y) }
  }, [map, x0])

  const lambda = useMemo(() => lyapunovExponent(map, 0.2345, { keep: 5000 }).exponent, [map])
  const late = useMemo(() => toFlat(orbit(map, x0, 8, { discard: 2000 })), [map, x0])

  const graph = useMemo(() => {
    const xs = Array.from({ length: 201 }, (_, i) => i / 200)
    return { x: xs, y: xs.map(map.f) }
  }, [map])

  const zero = useMemo(() => ({ x: [rs[0], rs[rs.length - 1]], y: [0, 0] }), [rs])
  const shown = Math.min(web.x.length, 2 * n + 1)
  const xn = web.x[shown - 1]
  const webNow = { x: web.x.slice(0, shown), y: web.y.slice(0, shown) }

  const rAxis = useAxis({ label: 'parameter r', range: family.range, nice: false })
  const xAxis = useAxis({ label: 'long-term state x', range: [0, 1], nice: false })
  const lAxis = useAxis({ label: 'Lyapunov exponent λ', range: [-2, 1] })
  const cx = useAxis({ label: 'x_n', range: [0, 1] })
  const cy = useAxis({ label: 'x_{n+1} = f(x_n)', range: [0, 1], equal: cx })

  const sweepAt = Math.round(((r - family.range[0]) / (family.range[1] - family.range[0])) * (SWEEP - 1))
  const sweepR = (k: number) => family.range[0] + ((family.range[1] - family.range[0]) * k) / (SWEEP - 1)

  return (
    <Figure
      title="Bifurcation diagram and cobweb analysis"
      purpose="Long-run orbit of a 1D map as its parameter r varies: period-doubling route to chaos, periodic windows, and the Lyapunov exponent λ(r). Positive λ indicates deterministic chaos."
      state={state}
      defaultSize="XL"
      controls={
        <ControlGroup title="Playback" collapsible={false}>
          <ControlRow label="Parameter sweep">
            <Player
              className="col-span-full"
              value={sweepAt}
              onChange={(k) => setR(sweepR(k))}
              count={SWEEP}
              format={(k) => `r = ${sweepR(k).toFixed(3)}`}
              label="r"
              startReason="Sweep r across the bifurcation parameter range."
            />
          </ControlRow>
          <ControlRow label="Cobweb steps">
            <Player
              className="col-span-full"
              value={n}
              onChange={setN}
              count={COBWEB_N + 1}
              format={(k) => `n = ${k}`}
              label="iteration"
            />
          </ControlRow>
        </ControlGroup>
      }
      readouts={{
        metrics: (
          <>
            <Readout label="Lyapunov exponent λ(r)" value={fmt(lambda)} />
            <Readout label="current state x_n" value={`n = ${Math.floor((shown - 1) / 2)}: ${fmt(xn)}`} />
            <Readout label="late orbit cycle" value={late.map((v) => v.toFixed(3)).join(', ')} />
          </>
        ),
      }}
      caption="Left: Attractor for each parameter value r (log visit counts of 300 iterates after 400 discarded transients), with the Lyapunov exponent λ(r) below. Drag the vertical handle to select r. Right: The 1D map y = f(x), line y = x, and cobweb trajectory from initial point x₀ (drag to change). The vertical jump is x_{n+1} = f(x_n), and the horizontal line to y = x transfers the output as the next input."
    >
      <Dashboard>
        <DashboardRow minHeight={420}>
          <DashboardCell ratio={1.6}>
            <Plots rows={2} heights={[3, 1.2]}>
              <Plot x={rAxis} y={xAxis}>
                <Raster x={diagram.x} y={diagram.y} z={diagram.z} valueLabel="log(1 + visits)" colorBar={false} />
                <Handle kind="x" at={r} onDrag={setR} label="r" />
              </Plot>
              <Plot x={rAxis} y={lAxis} legend={false}>
                <Curve name="λ(r)" x={rs} y={lyap} slot={1} />
                <Curve name="zero" x={zero.x} y={zero.y} muted />
                <Handle kind="x" at={r} onDrag={setR} label="r" />
              </Plot>
            </Plots>
          </DashboardCell>
          <DashboardCell aspect="square">
            <Plot x={cx} y={cy}>
              <Curve name="f" x={graph.x} y={graph.y} slot={0} />
              <Curve name="y = x" x={[0, 1]} y={[0, 1]} muted />
              <Curve name="cobweb" x={webNow.x} y={webNow.y} slot={2} live />
              <Points name="x_n" x={[xn]} y={[xn]} emphasis live />
              <Handle {...state.handle('x0', { label: 'x₀', axis: 'x' })} />
            </Plot>
          </DashboardCell>
        </DashboardRow>
      </Dashboard>
    </Figure>
  )
}
