import { useMemo, useState } from 'react'
import { Diagram } from 'aifn-render'
import { factor, link, variable } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from 'aifn-render'

type Weapon = 'unknown' | 'revolver' | 'dagger'
type Hair = 'unknown' | 'found' | 'absent'

/** The book's conditional probability tables: P(evidence | murderer), for Grey and for Auburn. */
const P_REVOLVER = { grey: 0.9, auburn: 0.2 }
const P_HAIR = { grey: 0.5, auburn: 0.05 }

const weaponLik = (w: Weapon, who: 'grey' | 'auburn') =>
  w === 'unknown' ? 1 : w === 'revolver' ? P_REVOLVER[who] : 1 - P_REVOLVER[who]
const hairLik = (h: Hair, who: 'grey' | 'auburn') =>
  h === 'unknown' ? 1 : h === 'found' ? P_HAIR[who] : 1 - P_HAIR[who]

/** Posterior P(Grey) from the prior and the incoming factor messages. */
function posterior(prior: number, messages: [number, number][]) {
  const g = messages.reduce((p, m) => p * m[0], prior)
  const a = messages.reduce((p, m) => p * m[1], 1 - prior)
  return g / (g + a)
}

const STEPS = [0, 1, 2]

/** Toggle the evidence and watch the belief about the murderer change, one factor message at a time. */
export function MurderMystery() {
  const prior = useParam(0.3, { min: 0.05, max: 0.95, step: 0.05 })
  const [weapon, setWeapon] = useState<Weapon>('revolver')
  const [hair, setHair] = useState<Hair>('found')

  const { steps, points } = useMemo(() => {
    const wMsg: [number, number] = [weaponLik(weapon, 'grey'), weaponLik(weapon, 'auburn')]
    const hMsg: [number, number] = [hairLik(hair, 'grey'), hairLik(hair, 'auburn')]
    const steps = [posterior(prior.value, []), posterior(prior.value, [wMsg]), posterior(prior.value, [wMsg, hMsg])]
    const points: XYSeries[] = [
      { name: 'P(Grey)', type: 'line', x: STEPS, y: steps, slot: 0 },
      { name: 'after each clue', type: 'scatter', x: STEPS, y: steps, emphasis: true },
    ]
    return { steps, points }
  }, [prior.value, weapon, hair])
  const final = steps[2]
  const weaponSeen = weapon !== 'unknown'
  const hairSeen = hair !== 'unknown'
  const spec = useMemo((): DiagramSpec => graph(final, weaponSeen, hairSeen), [final, weaponSeen, hairSeen])

  return (
    <Interactive
      title="Updating the belief about the murderer"
      caption="Choose the evidence. The murderer node is shaded by the posterior probability that Grey did it, and observed clues are filled. The chart follows P(Grey) from the prior, through the weapon factor's message, to the hair factor's message. The probabilities of each clue under each suspect are those of Model-Based Machine Learning, chapter 1."
      controls={
        <>
          <ParamSlider label="prior P(Grey)" param={prior} format={(v) => v.toFixed(2)} />
          <ParamChoice
            label="weapon found"
            value={weapon}
            onChange={setWeapon}
            options={[
              { value: 'unknown', label: 'not yet' },
              { value: 'revolver', label: 'revolver' },
              { value: 'dagger', label: 'dagger' },
            ]}
          />
          <ParamChoice
            label="grey hair at the scene"
            value={hair}
            onChange={setHair}
            options={[
              { value: 'unknown', label: 'not checked' },
              { value: 'found', label: 'found' },
              { value: 'absent', label: 'not found' },
            ]}
          />
        </>
      }
      readout={
        <>
          <Readout label="prior" value={formatNumber(steps[0])} />
          <Readout label="after weapon" value={formatNumber(steps[1])} />
          <Readout label="after hair" value={formatNumber(steps[2])} />
          <Readout label="P(Auburn)" value={formatNumber(1 - final)} />
        </>
      }
    >
      <div className="grid grid-cols-1 items-center gap-4 md:grid-cols-2">
        <Diagram
          spec={spec}
          ariaLabel="Factor graph: a prior factor on the murderer, and two factors linking the murderer to the weapon and to the hair"
        />
        <XYChart
          series={points}
          xLabel="clues included (0 = prior, 1 = weapon, 2 = weapon and hair)"
          yLabel="P(Grey is the murderer)"
          xRange={[0, 2]}
          yRange={[0, 1]}
          height={260}
        />
      </div>
    </Interactive>
  )
}

function graph(pGrey: number, weaponSeen: boolean, hairSeen: boolean): DiagramSpec {
  return {
    unit: 44,
    nodes: [
      factor('prior', 2, 0, '$\\pr(m)$', 'n'),
      variable('m', 2, 1.4, '$m$', { shade: pGrey }),
      factor('fw', 0.8, 2.8, '$\\pr(w \\mid m)$', 'w'),
      factor('fh', 3.2, 2.8, '$\\pr(h \\mid m)$', 'e'),
      variable('w', 0.8, 4.2, '$w$', { filled: weaponSeen }),
      variable('h', 3.2, 4.2, '$h$', { filled: hairSeen }),
    ],
    edges: [
      link('prior', 'm', false),
      link('m', 'fw', false),
      link('m', 'fh', false),
      link('fw', 'w', false),
      link('fh', 'h', false),
    ],
  }
}
