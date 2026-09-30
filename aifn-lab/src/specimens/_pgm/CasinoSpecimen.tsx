import { dishonestCasino, forwardBackward, sampleHmm, viterbi } from 'aifn/pgm'
import { stream } from 'aifn/random'
import { toRows } from 'aifn/tensor'
import { useMemo, useState } from 'react'
import { Slider, Switch, useParam } from '@lab/controls'
import { Figure } from '@lab/layout'
import { Panel, Readout, Subplots, XYChart, formatNumber, type XYSeries } from '@lab/viz'

const CASINO = dishonestCasino()

/** Forward–backward and Viterbi on rolls of the occasionally dishonest casino. */
export function CasinoSpecimen() {
  const n = useParam(200, { min: 20, max: 400, step: 10 })
  const seed = useParam(1, { min: 1, max: 50, step: 1 })
  const [showFiltered, setShowFiltered] = useState(false)
  const data = useMemo(() => sampleHmm(stream('casino').child(seed.value), CASINO, n.value), [n.value, seed.value])
  const obs = useMemo(() => Array.from(data.observations.data), [data])
  const fb = useMemo(() => forwardBackward(CASINO, obs), [obs])
  const vit = useMemo(() => viterbi(CASINO, obs), [obs])
  const t = obs.map((_, i) => i)
  const truth = Array.from(data.states.data)
  const smoothed = toRows(fb.marginals).map((r) => r[1])
  const filtered = toRows(fb.filtered).map((r) => r[1])
  const path = Array.from(vit.path.data)
  const accuracy = (xs: number[]) => xs.filter((x, i) => x === truth[i]).length / xs.length
  const rolls: XYSeries[] = [
    {
      name: 'roll',
      type: 'scatter',
      x: t,
      y: obs.map((o) => o + 1),
      group: truth,
      groupNames: ['fair die', 'loaded die'],
    },
  ]
  const posterior: XYSeries[] = [
    { name: 'true state (loaded = 1)', type: 'area', x: t, y: truth, muted: true },
    { name: 'p(loaded | all rolls)', type: 'line', x: t, y: smoothed, slot: 0 },
    ...(showFiltered
      ? [{ name: 'p(loaded | rolls so far)', type: 'line' as const, x: t, y: filtered, slot: 2, dashed: true }]
      : []),
    { name: 'Viterbi path', type: 'line', x: t, y: path.map((k) => k * 1.04), slot: 1 },
  ]
  return (
    <Figure
      title="Forward–backward and Viterbi on the dishonest casino"
      description="The posterior probability that each roll came from the loaded die, and the single most probable sequence of dice."
      defaultSize="L"
      controls={
        <>
          <Slider label="rolls" param={n} />
          <Slider label="seed" param={seed} />
          <Switch label="show the filtered probability" checked={showFiltered} onChange={setShowFiltered} />
        </>
      }
      readouts={
        <>
          <Readout label="log p(rolls)" value={formatNumber(fb.logLikelihood)} />
          <Readout
            label="posterior decoding accuracy"
            value={formatNumber(accuracy(smoothed.map((p) => (p > 0.5 ? 1 : 0))))}
          />
          <Readout label="Viterbi accuracy" value={formatNumber(accuracy(path))} />
        </>
      }
      caption="The casino switches to a loaded die (a six half the time) with probability 0.05 per roll and back with 0.1 (Durbin et al. 1998). Top: the rolls, coloured by the die that produced them. Bottom: the grey band is the true die; the smoothed posterior uses every roll, forward and backward; the Viterbi path is the single best sequence, lifted slightly to stay visible. The filtered probability (a toggle) uses only the rolls so far, so it lags behind each switch."
    >
      <Subplots rows={2} sharex heightRatios={[1, 1.4]} hoverGroup>
        <Panel>
          <XYChart series={rolls} yLabel="face" yRange={[0.5, 6.5]} integerX />
        </Panel>
        <Panel>
          <XYChart series={posterior} xLabel="roll" yLabel="p(loaded)" yRange={[0, 1.1]} />
        </Panel>
      </Subplots>
    </Figure>
  )
}
