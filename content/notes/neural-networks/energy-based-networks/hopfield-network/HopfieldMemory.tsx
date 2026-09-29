import { FastForward, RotateCcw } from 'lucide-react'
import { useMemo, useState } from 'react'
import {
  ImagePlot,
  Interactive,
  ParamButton,
  ParamChoice,
  ParamSlider,
  ParamSwitch,
  Readout,
  XYChart,
  formatNumber,
  type Handle,
  type ImagePlotLine,
  type PlotPointer,
  type XYSeries,
} from '@/components/viz'
import { rng } from '@/lib/math'
import { GLYPH_SIZE, glyph } from '../_shared/glyphs'
import { PlayButton } from '../_shared/Playback'
import { usePlayLoop } from '../_shared/usePlayLoop'
import { corrupt, hebbian, mosaic, overlap, pixelBox, runAsync, stateAt } from '../_shared/spins'

const N = GLYPH_SIZE * GLYPH_SIZE
/** Stored in this order: the first P letters of the word. */
const WORD = 'HOPFIELD'
const NO_EDITS: number[] = []

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
  const s = corrupt(signed, noise, rng(seed).uniform)
  for (const i of editKey ? editKey.split(',').map(Number) : []) s[i] = -s[i]
  return s
}

function overlapBars(y: number[]): XYSeries[] {
  return [{ name: 'overlap', type: 'bar', x: y.map((_, k) => k + 1), y, slot: 0 }]
}

/**
 * Letters stored in a 100-neuron Hopfield network with Hebbian weights. A corrupted letter is the start state; each
 * step updates one neuron to the sign of its local field, and the energy never rises.
 */
export function HopfieldMemory() {
  const [scenario, setScenario] = useState<Scenario>('clean')
  const [P, setP] = useState(3)
  const [noise, setNoise] = useState(0.1)
  const [reversed, setReversed] = useState(false)
  const [cue, setCue] = useState('H')
  const [seed, setSeed] = useState(1)
  const [edits, setEdits] = useState<number[]>(NO_EDITS)
  const [t, setT] = useState(0)
  const [playing, setPlaying] = useState(false)

  const restart = () => {
    setT(0)
    setPlaying(false)
  }
  const choose = (value: Scenario) => {
    const spec = SCENARIOS.find((s) => s.value === value)!
    setScenario(value)
    setP(spec.P)
    setNoise(spec.noise)
    setReversed(spec.reversed)
    setCue(spec.cue)
    setEdits(NO_EDITS)
    restart()
  }

  const letters = WORD.slice(0, P).split('')
  const patterns = useMemo(() => WORD.slice(0, P).split('').map(glyph), [P])
  const W = useMemo(() => hebbian(patterns, N), [patterns])

  const cueLetter = cue === 'mix' || WORD.slice(0, P).includes(cue) ? cue : WORD[0]
  const editKey = edits.join(',')
  const start = useMemo(
    () => startState(cueLetter, reversed, noise, seed, editKey),
    [cueLetter, reversed, noise, seed, editKey],
  )

  const traj = useMemo(() => runAsync(W, start, rng(seed + 1000).uniform), [W, start, seed])
  const total = traj.sites.length
  const shown = Math.min(t, total)
  const state = useMemo(() => stateAt(traj, shown), [traj, shown])
  const overlaps = useMemo(() => patterns.map((p) => overlap(p, state)), [patterns, state])

  usePlayLoop(playing, 40, (n) => {
    const next = Math.min(total, shown + n)
    setT(next)
    if (next >= total) setPlaying(false)
    return next < total
  })

  const lastSite = shown > 0 ? traj.sites[shown - 1] : null
  const lines = useMemo(
    (): ImagePlotLine[] =>
      lastSite === null
        ? []
        : [{ name: 'updated neuron', ...pixelBox(lastSite % GLYPH_SIZE, Math.floor(lastSite / GLYPH_SIZE)), slot: 3 }],
    [lastSite],
  )
  const stored = useMemo(() => mosaic(patterns, GLYPH_SIZE, GLYPH_SIZE, 4, 1, 0), [patterns])

  const onPointer = (event: PlotPointer) => {
    if (event.type !== 'click') return
    const c = Math.round(event.point[0])
    const r = Math.round(event.point[1])
    if (c < 0 || r < 0 || c >= GLYPH_SIZE || r >= GLYPH_SIZE) return
    const i = r * GLYPH_SIZE + c
    setEdits((e) => (e.includes(i) ? e.filter((k) => k !== i) : [...e, i].sort((a, b) => a - b)))
    restart()
  }

  const updates = useMemo(() => Array.from({ length: total + 1 }, (_, i) => i), [total])
  const energySeries = useMemo(
    (): XYSeries[] => [
      { name: 'energy', type: 'line', x: updates, y: traj.energy, slot: 3 },
      { name: 'now', type: 'scatter', x: [shown], y: [traj.energy[shown]], emphasis: true },
    ],
    [updates, traj, shown],
  )
  const energyHandles = useMemo(
    (): Handle[] => [
      {
        kind: 'x',
        at: shown,
        label: 'update',
        onDrag: (x) => {
          setPlaying(false)
          setT(Math.max(0, Math.min(total, Math.round(x))))
        },
      },
    ],
    [shown, total],
  )
  const overlapSeries = useMemo(() => overlapBars(overlaps), [overlaps])

  const best = overlaps.reduce((b, m, k) => (Math.abs(m) > Math.abs(overlaps[b]) ? k : b), 0)
  const verdict =
    Math.abs(overlaps[best]) === 1
      ? `${overlaps[best] > 0 ? '' : 'reversed '}${letters[best]} exactly`
      : `nearest ${letters[best]} (${formatNumber(overlaps[best])}), not a stored letter`

  return (
    <Interactive
      title="Hopfield memory: recall one neuron at a time"
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
          <ParamChoice label="scenario" value={scenario} onChange={choose} options={SCENARIOS} />
          <ParamSlider
            label="stored letters P"
            value={P}
            onChange={(v) => {
              setP(v)
              restart()
            }}
            min={1}
            max={WORD.length}
            step={1}
            format={(v) => `${v} (${WORD.slice(0, v)})`}
          />
          <ParamSlider
            label="fraction of pixels flipped"
            value={noise}
            onChange={(v) => {
              setNoise(v)
              restart()
            }}
            min={0}
            max={0.5}
            step={0.01}
          />
          <ParamChoice
            label="cue"
            value={cueLetter}
            onChange={(v) => {
              setCue(v)
              setEdits(NO_EDITS)
              restart()
            }}
            options={[
              ...letters.map((l) => ({ value: l, label: l })),
              ...(P >= 3 ? [{ value: 'mix', label: 'H+O+P' }] : []),
            ]}
          />
          <ParamSwitch
            label="reverse the cue"
            checked={reversed}
            onChange={(v) => {
              setReversed(v)
              restart()
            }}
          />
          <ParamSlider
            label="update"
            value={shown}
            onChange={(v) => {
              setPlaying(false)
              setT(v)
            }}
            min={0}
            max={total}
            step={1}
            withArrows
            debounceMs={0}
          />
          <div className="flex flex-wrap gap-2 self-end">
            <PlayButton
              playing={playing}
              disabled={shown >= total && !playing}
              onToggle={() => setPlaying((p) => !p)}
            />
            <ParamButton
              disabled={shown >= total}
              onClick={() => {
                setPlaying(false)
                setT(Math.min(total, (Math.floor(shown / N) + 1) * N))
              }}
            >
              <FastForward /> Sweep
            </ParamButton>
            <ParamButton
              onClick={() => {
                setSeed((s) => s + 1)
                setEdits(NO_EDITS)
                restart()
              }}
            >
              <RotateCcw /> New noise
            </ParamButton>
          </div>
        </>
      }
      readout={
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
          <ImagePlot
            width={GLYPH_SIZE}
            height={GLYPH_SIZE}
            values={state}
            scale="diverging"
            range={[-1, 1]}
            lines={lines}
            onPointer={onPointer}
            ariaLabel="Current state of the 100 neurons"
          />
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-xs text-muted-foreground">stored letters and overlaps m</span>
          <ImagePlot
            width={stored.width}
            height={stored.height}
            xExtent={4 * GLYPH_SIZE + 3}
            values={stored.values}
            scale="diverging"
            range={[-1, 1]}
            ariaLabel="The stored letters"
          />
          <XYChart
            height={170}
            xLabel="stored letter"
            yLabel="m"
            xRange={[0.5, Math.max(P, 3) + 0.5]}
            yRange={[-1, 1]}
            integerX
            series={overlapSeries}
            ariaLabel="Overlap of the state with each stored letter"
          />
        </div>
        <XYChart
          height={260}
          xLabel="update"
          yLabel="energy E"
          series={energySeries}
          handles={energyHandles}
          ariaLabel="Energy after each update"
        />
      </div>
    </Interactive>
  )
}
