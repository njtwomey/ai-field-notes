import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from '@/components/viz'
import { formatComplex, polyAdd, roots, type Complex, type Vec } from '../../_shared/control'

type Plant = {
  label: string
  num: Vec
  den: Vec
  poles: Complex[]
  zeros: Complex[]
  x: [number, number]
  y: [number, number]
  initial: number
}

const re = (...xs: number[]): Complex[] => xs.map((v) => ({ re: v, im: 0 }))

const PLANTS = {
  three: {
    label: 'K / (s(s+1)(0.2s+1))',
    num: [1],
    den: [0.2, 1.2, 1, 0],
    poles: re(0, -1, -5),
    zeros: [],
    x: [-8, 3],
    y: [-5, 5],
    initial: 0,
  },
  unstable: {
    label: 'K (s+2) / (s(s−1))',
    num: [1, 2],
    den: [1, -1, 0],
    poles: re(0, 1),
    zeros: re(-2),
    x: [-9, 3],
    y: [-4, 4],
    initial: 0.3,
  },
  lead: {
    label: 'K (s+1) / (s²(s+9))',
    num: [1, 1],
    den: [1, 9, 0, 0],
    poles: re(0, 0, -9),
    zeros: re(-1),
    x: [-11, 2],
    y: [-7, 7],
    initial: 1,
  },
} satisfies Record<string, Plant>
type PlantKey = keyof typeof PLANTS

const closedLoop = (p: Plant, k: number) =>
  polyAdd(
    p.den,
    p.num.map((c) => k * c),
  )
/** Range of log10 K swept to draw the locus and offered by the slider. */
const LOG_K: [number, number] = [-3, 3]
const maxRe = (zs: Complex[]) => Math.max(...zs.map((z) => z.re))

export function RootLocus() {
  const [key, setKey] = useState<PlantKey>('three')
  const plant: Plant = PLANTS[key]
  const logK = useParam(plant.initial, { min: LOG_K[0], max: LOG_K[1], step: 0.01 })
  const choose = (k: PlantKey) => {
    setKey(k)
    logK.set(PLANTS[k].initial)
  }

  // The locus: roots for a dense sweep of K, each warm-started from the previous so branches stay continuous.
  const locus = useMemo(() => {
    const n = 500
    const ks: number[] = []
    const rows: Complex[][] = []
    let prev: Complex[] | undefined
    for (let i = 0; i <= n; i++) {
      const k = 10 ** (LOG_K[0] + ((LOG_K[1] - LOG_K[0]) * i) / n)
      prev = roots(closedLoop(plant, k), prev)
      ks.push(k)
      rows.push(prev)
    }
    // Gain at which the loop changes stability, refined by bisection between grid points.
    const crossings: number[] = []
    for (let i = 1; i <= n; i++) {
      const a = maxRe(rows[i - 1])
      const b = maxRe(rows[i])
      if (Math.sign(a) !== Math.sign(b)) {
        let lo = ks[i - 1]
        let hi = ks[i]
        for (let it = 0; it < 50; it++) {
          const mid = Math.sqrt(lo * hi)
          if (Math.sign(maxRe(roots(closedLoop(plant, mid)))) === Math.sign(a)) lo = mid
          else hi = mid
        }
        crossings.push(Math.sqrt(lo * hi))
      }
    }
    return { ks, rows, crossings }
  }, [plant])

  const k = 10 ** logK.value
  const now = useMemo(() => roots(closedLoop(plant, k)), [plant, k])
  const stable = maxRe(now) < 0
  // The handle is the closed-loop pole with the largest real part (upper one of a pair).
  const handlePole = [...now].sort((a, b) => b.re - a.re || b.im - a.im)[0]

  const series = useMemo<XYSeries[]>(() => {
    const branches = locus.rows[0].map((_, j) => ({
      name: 'root locus',
      type: 'line' as const,
      x: locus.rows.map((r) => r[j].re),
      y: locus.rows.map((r) => r[j].im),
      slot: 0,
    }))
    return [
      ...branches,
      {
        name: 'open-loop poles',
        type: 'scatter',
        x: plant.poles.map((z) => z.re),
        y: plant.poles.map((z) => z.im),
        slot: 1,
      },
      ...(plant.zeros.length
        ? [
            {
              name: 'open-loop zeros',
              type: 'scatter' as const,
              x: plant.zeros.map((z) => z.re),
              y: plant.zeros.map((z) => z.im),
              slot: 2,
            },
          ]
        : []),
      {
        name: 'closed-loop poles at K',
        type: 'scatter',
        x: now.map((z) => z.re),
        y: now.map((z) => z.im),
        emphasis: true,
      },
    ]
  }, [locus, plant, now])

  // Dragging a closed-loop pole: find the nearest point on the locus and adopt its gain.
  const dragTo = ([x, y]: [number, number]) => {
    let best = Infinity
    let bestK = k
    locus.rows.forEach((r, i) =>
      r.forEach((z) => {
        const d = (z.re - x) ** 2 + (z.im - y) ** 2
        if (d < best) {
          best = d
          bestK = locus.ks[i]
        }
      }),
    )
    logK.set(Math.log10(bestK))
  }

  const uniquePoles = now.filter((z) => z.im >= -1e-9)
  return (
    <Interactive
      title="Closed-loop poles as the gain grows"
      caption="The closed-loop poles of 1 + K·G(s) = 0 for every K > 0 (lines). They start at the open-loop poles when K is small and end at the open-loop zeros, or run off to infinity along asymptotes. The large dots are the poles at the current gain; drag one along the locus, or use the slider, to change K. The loop is stable while every pole is left of the imaginary axis. The second plant is unstable on its own and needs K > 1; the third shows a zero pulling two poles off the imaginary axis."
      controls={
        <>
          <ParamChoice
            label="open loop"
            value={key}
            onChange={choose}
            options={(Object.keys(PLANTS) as PlantKey[]).map((p) => ({ value: p, label: PLANTS[p].label }))}
          />
          <ParamSlider label="gain K" param={logK} format={(v) => formatNumber(10 ** v)} />
        </>
      }
      readout={
        <>
          <Readout label="K" value={formatNumber(k)} />
          <Readout
            label="closed-loop poles"
            value={uniquePoles
              .map((z) => (Math.abs(z.im) > 1e-6 ? `${formatComplex(z)} (pair)` : formatComplex(z)))
              .join(', ')}
          />
          <Readout label="closed loop" value={stable ? 'stable' : 'unstable'} />
          <Readout
            label="stability changes at K"
            value={locus.crossings.length ? locus.crossings.map((c) => formatNumber(c)).join(', ') : 'never'}
          />
        </>
      }
    >
      <div className="mx-auto w-full max-w-xl">
        <XYChart
          series={series}
          xLabel="real part"
          yLabel="imaginary part"
          xRange={plant.x}
          yRange={plant.y}
          equalAspect
          handles={[{ kind: 'point', at: [handlePole.re, handlePole.im], onDrag: dragTo, label: 'closed-loop pole' }]}
        />
      </div>
    </Interactive>
  )
}
