import { useMemo } from 'react'
import {
  choice,
  Curve,
  Diagram,
  factor,
  Figure,
  float,
  formatNumber,
  link,
  Plot,
  Points,
  Readout,
  useAxis,
  useFigureState,
  variable,
} from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'

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
  const state = useFigureState({
    prior: float(0.3, { min: 0.05, max: 0.95, step: 0.05, label: 'prior P(Grey)', format: (v) => v.toFixed(2) }),
    weapon: choice<Weapon>(
      [
        { value: 'unknown', label: 'not yet' },
        { value: 'revolver', label: 'revolver' },
        { value: 'dagger', label: 'dagger' },
      ],
      'revolver',
      { label: 'weapon found' },
    ),
    hair: choice<Hair>(
      [
        { value: 'unknown', label: 'not checked' },
        { value: 'found', label: 'found' },
        { value: 'absent', label: 'not found' },
      ],
      'found',
      { label: 'grey hair at the scene' },
    ),
  })

  const { steps, points } = useMemo(() => {
    const wMsg: [number, number] = [weaponLik(state.weapon, 'grey'), weaponLik(state.weapon, 'auburn')]
    const hMsg: [number, number] = [hairLik(state.hair, 'grey'), hairLik(state.hair, 'auburn')]
    const steps = [posterior(state.prior, []), posterior(state.prior, [wMsg]), posterior(state.prior, [wMsg, hMsg])]
    const points = [
      { name: 'P(Grey)', x: STEPS, y: steps, slot: 0 },
      { name: 'after each clue', x: STEPS, y: steps, emphasis: true },
    ] as const
    return { steps, points }
  }, [state.prior, state.weapon, state.hair])
  const final = steps[2]
  const weaponSeen = state.weapon !== 'unknown'
  const hairSeen = state.hair !== 'unknown'
  const spec = useMemo((): DiagramSpec => graph(final, weaponSeen, hairSeen), [final, weaponSeen, hairSeen])

  const xAxis = useAxis({ label: 'clues included (0 = prior, 1 = weapon, 2 = weapon and hair)', range: [0, 2] })
  const yAxis = useAxis({ label: 'P(Grey is the murderer)', range: [0, 1] })
  return (
    <Figure
      title="Updating the belief about the murderer"
      state={state}
      caption="Choose the evidence. The murderer node is shaded by the posterior probability that Grey did it, and observed clues are filled. The chart follows P(Grey) from the prior, through the weapon factor's message, to the hair factor's message. The probabilities of each clue under each suspect are those of Model-Based Machine Learning, chapter 1."

      readouts={
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
        <Plot x={xAxis} y={yAxis} height={260}>
          <Curve {...points[0]} />
          <Points {...points[1]} />
        </Plot>
      </div>
    </Figure>
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
