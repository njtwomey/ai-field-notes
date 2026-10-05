import { FastForward } from 'lucide-react'
import { useMemo, useState } from 'react'
import {
  Bars,
  Button,
  choice,
  Curve,
  Figure,
  formatNumber,
  Handle,
  int,
  Pixels,
  Player,
  Plot,
  type PlotPointer,
  Points,
  Readout,
  setting,
  slider,
  useAxis,
  useFigureState,
  variants,
} from 'aifn-render'
import { stream, uniform } from 'aifn-compute/foundation/random'
import { GLYPH_SIZE, glyph } from '../_shared/glyphs'
import { corrupt, hebbian, mosaic, overlap, pixelBox, runAsync, stateAt } from '../_shared/spins'

const N = GLYPH_SIZE * GLYPH_SIZE
/** Stored in this order: the first P letters of the word. */
const WORD = 'HOPFIELD'
const NO_EDITS: number[] = []
const PIXEL_RANGE: [number, number] = [-1, 1]
const MOSAIC_WIDTH = 4 * GLYPH_SIZE + 3

/** Seeded uniform draws. */
function draws(seed: number) {
  const g = stream(seed)
  return () => uniform(g)
}

type Scenario = 'clean' | 'noisy' | 'overload' | 'reversed' | 'mixture'
const SCENARIOS: { value: Scenario; label: string; P: number; noise: number; reversed: boolean; cue: string }[] = [
  { value: 'clean', label: 'clean recall', P: 3, noise: 0.1, reversed: false, cue: 'H' },
  { value: 'noisy', label: 'noisy recall', P: 3, noise: 0.35, reversed: false, cue: 'O' },
  { value: 'overload', label: 'too many memories', P: 8, noise: 0.1, reversed: false, cue: 'H' },
  { value: 'reversed', label: 'reversed pattern', P: 3, noise: 0.1, reversed: true, cue: 'P' },
  { value: 'mixture', label: 'mixture state', P: 3, noise: 0.1, reversed: false, cue: 'mix' },
]

/** The start state: a stored letter or the mixture sgn(H + O + P), optionally reversed, corrupted, then edited. */
function startState(cue: string, reversed: boolean, noise: number, seed: number, editKey: string): Int8Array {
  const base =
    cue === 'mix'
      ? Int8Array.from({ length: N }, (_, i) => Math.sign(glyph('H')[i] + glyph('O')[i] + glyph('P')[i]))
      : glyph(cue)
  const signed = reversed ? base.map((v) => -v) : base
  const s = corrupt(signed, noise, draws(seed))
  for (const i of editKey ? editKey.split(',').map(Number) : []) s[i] = -s[i]
  return s
}

const CUES = [...WORD.split('').map((l) => ({ value: l, label: l })), { value: 'mix', label: 'H+O+P (needs P ≥ 3)' }]

/** Each scenario opens with its own settings, which can then be changed. */
const SCENARIO = variants(
  Object.fromEntries(
    SCENARIOS.map((sc) => [
      sc.value,
      {
        label: sc.label,
        params: {
          P: int(sc.P, {
            min: 1,
            max: WORD.length,
            step: 1,
            label: 'stored letters P',
            format: (v) => `${v} (${WORD.slice(0, v)})`,
          }),
          noise: slider(0, 0.5, sc.noise, { step: 0.01, label: 'fraction of pixels flipped' }),
          cue: choice<string>(CUES, sc.cue, { label: 'cue' }),
          reversed: setting(sc.reversed, 'reverse the cue'),
        },
      },
    ]),
  ) as Record<
    Scenario,
    {
      label: string
      params: {
        P: ReturnType<typeof int>
        noise: ReturnType<typeof slider>
        cue: ReturnType<typeof choice<string>>
        reversed: ReturnType<typeof setting>
      }
    }
  >,
  { label: 'Scenario', choiceLabel: 'scenario', initial: 'clean' },
)

/**
 * Letters stored in a 100-neuron Hopfield network with Hebbian weights. A corrupted letter is the start state; each
 * step updates one neuron to the sign of its local field, and the energy never rises.
 */
export function HopfieldMemory() {
  const state = useFigureState({ scenario: SCENARIO, seed: int(1, { ge: 0, label: 'noise seed' }) })
  const scenario = state.scenario.key as Scenario
  const { P, noise, cue, reversed } = state.scenario.values as {
    P: number
    noise: number
    cue: string
    reversed: boolean
  }
  const seed = state.seed

  const letters = useMemo(() => WORD.slice(0, P).split(''), [P])
  const patterns = useMemo(() => letters.map(glyph), [letters])
  const W = useMemo(() => hebbian(patterns, N), [patterns])

  const cueLetter = (cue === 'mix' && P >= 3) || WORD.slice(0, P).includes(cue) ? cue : WORD[0]
  // Clicked pixels belong to their cue; another cue, scenario or noise sample starts unedited.
  const editFor = `${scenario}:${cueLetter}:${seed}`
  const [edited, setEdited] = useState({ editFor, edits: NO_EDITS })
  const edits = edited.editFor === editFor ? edited.edits : NO_EDITS
  const editKey = edits.join(',')
  const start = useMemo(
    () => startState(cueLetter, reversed, noise, seed, editKey),
    [cueLetter, reversed, noise, seed, editKey],
  )

  const traj = useMemo(() => runAsync(W, start, draws(seed + 1000)), [W, start, seed])
  const total = traj.sites.length
  // The walk through the updates restarts at 0 for a new run.
  const [position, setPosition] = useState({ traj, t: 0 })
  const shown = position.traj === traj ? Math.min(position.t, total) : 0
  const go = (v: number) => setPosition({ traj, t: Math.max(0, Math.min(total, Math.round(v))) })
  const spins = useMemo(() => stateAt(traj, shown), [traj, shown])
  const overlaps = useMemo(() => patterns.map((p) => overlap(p, spins)), [patterns, spins])

  const lastSite = shown > 0 ? traj.sites[shown - 1] : null
  const box = useMemo(
    () => (lastSite === null ? null : pixelBox(lastSite % GLYPH_SIZE, Math.floor(lastSite / GLYPH_SIZE))),
    [lastSite],
  )
  const stored = useMemo(() => mosaic(patterns, GLYPH_SIZE, GLYPH_SIZE, 4, 1, 0), [patterns])

  const onPointer = (event: PlotPointer) => {
    if (event.type !== 'click') return
    const c = Math.round(event.point[0])
    const r = Math.round(event.point[1])
    if (c < 0 || r < 0 || c >= GLYPH_SIZE || r >= GLYPH_SIZE) return
    const i = r * GLYPH_SIZE + c
    setEdited({
      editFor,
      edits: edits.includes(i) ? edits.filter((k) => k !== i) : [...edits, i].sort((a, b) => a - b),
    })
  }

  const updates = useMemo(() => Array.from({ length: total + 1 }, (_, i) => i), [total])
  const positions = useMemo(() => letters.map((_, k) => k), [letters])

  const best = overlaps.reduce((b, m, k) => (Math.abs(m) > Math.abs(overlaps[b]) ? k : b), 0)
  const verdict =
    Math.abs(overlaps[best]) === 1
      ? `${overlaps[best] > 0 ? '' : 'reversed '}${letters[best]} exactly`
      : `nearest ${letters[best]} (${formatNumber(overlaps[best])}), not a stored letter`

  const xAxis = useAxis({ label: 'update', hold: 'union' })
  const yAxis = useAxis({ label: 'energy E', hold: 'union' })
  // Pixel images: row 0 at the top, square pixels.
  const netX = useAxis({ range: [-0.5, GLYPH_SIZE - 0.5], nice: false })
  const netY = useAxis({ range: [-0.5, GLYPH_SIZE - 0.5], nice: false, inverse: true, equal: netX })
  const storedX = useAxis({ range: [-0.5, MOSAIC_WIDTH - 0.5], nice: false })
  const storedY = useAxis({ range: [-0.5, stored.height - 0.5], nice: false, inverse: true, equal: storedX })
  const overlapX = useAxis({ label: 'stored letter', categories: letters })
  const overlapY = useAxis({ label: 'm', range: [-1, 1] })
  return (
    <Figure
      title="Hopfield memory: recall one neuron at a time"
      state={state}
      caption={
        <>
          The first P letters of HOPFIELD are stored in a network of 100 neurons (10 × 10, red +1, blue −1) with Hebbian
          weights. The start state is a letter with a fraction of its pixels flipped. Each update sets one neuron,
          outlined, to the sign of its local field; neurons are visited in a random order each sweep, and the run stops
          after a sweep with no change. The energy (right) never rises. Click pixels to flip them in the start state;
          drag the update line on the energy trace. The bars are the overlaps m with each stored letter, in the order of
          the stored letters, read row by row.
        </>
      }
      controls={
        <>
          <Player value={shown} onChange={go} count={total + 1} label="update" format={(v) => `update ${v}`} />
          <Button
            variant="outline"
            size="sm"
            disabled={shown >= total}
            onClick={() => go(Math.min(total, (Math.floor(shown / N) + 1) * N))}
          >
            <FastForward /> Sweep
          </Button>
        </>
      }
      readouts={
        <>
          <Readout label="update" value={`${shown} of ${total} (sweep ${Math.ceil(shown / N)})`} />
          <Readout label="energy" value={formatNumber(traj.energy[shown])} />
          <Readout label="state" value={shown >= total && traj.converged ? `stable: ${verdict}` : verdict} />
          <Readout
            label="overlaps"
            value={letters
              .map((l, k) => `${l} ${overlaps[k] >= 0 ? '+' : '−'}${Math.abs(overlaps[k]).toFixed(2)}`)
              .join('  ')}
          />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-[1fr_1.4fr_1.6fr]">
        <div className="flex flex-col gap-1">
          <span className="text-xs text-muted-foreground">network state</span>
          <Plot x={netX} y={netY} bare height={220} onPointer={onPointer} ariaLabel="Current state of the 100 neurons">
            <Pixels width={GLYPH_SIZE} height={GLYPH_SIZE} values={spins} scale="diverging" range={PIXEL_RANGE} />
            {box && <Curve name="updated neuron" x={box.x} y={box.y} slot={3} live />}
          </Plot>
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-xs text-muted-foreground">stored letters and overlaps m</span>
          <Plot x={storedX} y={storedY} bare height={120} ariaLabel="The stored letters">
            <Pixels
              width={stored.width}
              height={stored.height}
              values={stored.values}
              scale="diverging"
              range={PIXEL_RANGE}
            />
          </Plot>
          <Plot x={overlapX} y={overlapY} height={170} ariaLabel="Overlap of the state with each stored letter">
            <Bars name="overlap" x={positions} y={overlaps} slot={0} />
          </Plot>
        </div>
        <Plot x={xAxis} y={yAxis} height={260} ariaLabel={'Energy after each update'}>
          <Curve name="energy" x={updates} y={traj.energy} slot={3} />
          <Points name="now" x={[shown]} y={[traj.energy[shown]]} emphasis />
          <Handle kind="x" at={shown} label="update" onDrag={go} />
        </Plot>
      </div>
    </Figure>
  )
}
