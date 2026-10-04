import { useMemo } from 'react'
import { choice, Figure, float, formatNumber, int, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
import { instanceHardness, respond, simulate, type Point } from '../_shared/instanceIrt'

type Colour = 'ih' | 'kdn' | 'label'

const K = 5

/** k-disagreeing neighbours: the share of an instance's k nearest neighbours in the dataset with another label. */
function kdn(x: Point[], y: number[], k: number): number[] {
  return x.map((p, i) => {
    const d = x
      .map((q, j) => [(q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2, j])
      .filter(([, j]) => j !== i)
      .sort((a, b) => a[0] - b[0])
    return d.slice(0, k).filter(([, j]) => y[j] !== y[i]).length / k
  })
}

const NAMES: Record<Colour, string[]> = {
  ih: ['IH < 1/3', '1/3 ≤ IH < 2/3', 'IH ≥ 2/3'],
  kdn: ['kDN = 0', 'kDN = 0.2 or 0.4', 'kDN ≥ 0.6'],
  label: ['given class 0', 'given class 1', 'flipped label'],
}

export function HardnessMap() {
  const state = useFigureState({
    colour: choice<Colour>(
      [
        { value: 'ih', label: 'instance hardness' },
        { value: 'kdn', label: 'kDN' },
        { value: 'label', label: 'given label' },
      ],
      'ih',
      { label: 'colour by' },
    ),
    noise: float(0.1, { min: 0, max: 0.3, step: 0.05, label: 'label noise rate' }),
    seed: int(1, { min: 1, max: 20, step: 1, label: 'seed' }),
  })

  const run = useMemo(() => {
    const data = simulate(state.seed, state.noise)
    const resp = respond(data, state.seed)
    return { data, ih: instanceHardness(resp.probs), kdn: kdn(data.test.x, data.test.y, K) }
  }, [state.seed, state.noise])
  const { data, ih } = run

  const series = useMemo(() => {
    const group =
      state.colour === 'label'
        ? data.test.y.map((y, j) => (data.test.flipped[j] ? 2 : y))
        : state.colour === 'ih'
          ? ih.map((v) => (v < 1 / 3 ? 0 : v < 2 / 3 ? 1 : 2))
          : run.kdn.map((v) => (v === 0 ? 0 : v < 0.5 ? 1 : 2))
    return [
      {
        name: 'instances',
        x: data.test.x.map((p) => p[0]),
        y: data.test.x.map((p) => p[1]),
        group,
        groupNames: NAMES[state.colour],
      },
    ] as const
  }, [data, ih, run.kdn, state.colour])

  const scatter = useMemo(() => {
    // Spread the five possible kDN values sideways so that tied points stay visible.
    const jitter = data.test.x.map((_, j) => ((j * 0.618) % 1) * 0.1 - 0.05)
    const x = run.kdn.map((v, j) => v + jitter[j])
    const pick = (flip: boolean) => data.test.flipped.flatMap((f, j) => (f === flip ? [j] : []))
    const clean = pick(false)
    const flip = pick(true)
    return [
      { name: 'clean label', x: clean.map((j) => x[j]), y: clean.map((j) => ih[j]), slot: 0 },
      { name: 'flipped label', x: flip.map((j) => x[j]), y: flip.map((j) => ih[j]), slot: 2 },
    ] as const
  }, [data, ih, run.kdn])

  const flipped = data.test.flipped
  const meanOf = (v: number[], pick: (j: number) => boolean) => {
    const sel = v.filter((_, j) => pick(j))
    return sel.length ? sel.reduce((a, b) => a + b, 0) / sel.length : NaN
  }

  const xAxis = useAxis({ label: 'x₁', range: [-4.5, 4.5] })
  const yAxis = useAxis({ label: 'x₂', range: [-2.5, 2.5], equal: xAxis })
  const xAxis2 = useAxis({ label: 'kDN (k = 5)', range: [-0.1, 1.1] })
  const yAxis2 = useAxis({ label: 'instance hardness', range: [0, 1] })
  return (
    <Figure
      title="Instance hardness from a population of classifiers, and one of its explanations"
      state={state}
      caption="Seventy test instances from two overlapping Gaussian classes, with a fraction of the labels flipped. Instance hardness (IH) is one minus the mean probability that seventeen classifiers of different skill give the instance's label. kDN is the share of its five nearest neighbours with another label; it needs no classifiers. Colour the points by either, or by the label. The lower chart plots one against the other: flipped labels sit at the top right, boundary points in the middle."

      readouts={
        <>
          <Readout label="mean IH, clean labels" value={formatNumber(meanOf(ih, (j) => !flipped[j]))} />
          <Readout label="mean IH, flipped labels" value={formatNumber(meanOf(ih, (j) => flipped[j]))} />
          <Readout label="mean kDN, flipped labels" value={formatNumber(meanOf(run.kdn, (j) => flipped[j]))} />
        </>
      }
    >
      <div className="space-y-2">
        <Plot x={xAxis} y={yAxis} ariaLabel={'Test instances in feature space coloured by hardness'}>
          <Points {...series[0]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={300} ariaLabel={'Instance hardness against k-disagreeing neighbours'}>
          <Points {...scatter[0]} />
          <Points {...scatter[1]} />
        </Plot>
      </div>
    </Figure>
  )
}
