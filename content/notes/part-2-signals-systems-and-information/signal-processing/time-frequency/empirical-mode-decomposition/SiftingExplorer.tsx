import { useMemo } from 'react'
import {
  choice,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  Segments,
  seriesLayers,
  type SeriesSpec,
  setting,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { emd, findExtrema, RILLING, rillingStats, sdCriterion, sift, type StopRule } from '../_shared/emd'

const N = 400
const SIFTS = 10
const TIME = Array.from({ length: N }, (_, n) => n)
const BURST: [number, number] = [130, 270]

type RuleName = 'sd' | 'snumber' | 'rilling' | 'fixed'
const RULES: Record<RuleName, StopRule> = {
  sd: { kind: 'sd', threshold: 0.2 },
  snumber: { kind: 'snumber', s: 4 },
  rilling: RILLING,
  fixed: { kind: 'fixed', sifts: 10 },
}

/** Two tones (frequencies in cycles per sample) and a slow linear trend; the faster tone optionally a burst. */
function makeSignal(f1: number, a1: number, f2: number, a2: number, burst: boolean) {
  const fast = Float64Array.from(TIME, (n) =>
    !burst || (n >= BURST[0] && n < BURST[1]) ? a1 * Math.cos(2 * Math.PI * f1 * n) : 0,
  )
  const slow = Float64Array.from(TIME, (n) => a2 * Math.cos(2 * Math.PI * f2 * n + 0.7))
  const trend = Float64Array.from(TIME, (n) => 0.8 * (n / N - 0.5))
  const x = Float64Array.from(TIME, (n) => fast[n] + slow[n] + trend[n])
  return { x, parts: [fast, slow, trend] }
}

const line = (name: string, y: ArrayLike<number>, extra: Partial<SeriesSpec> = {}): SeriesSpec => ({
  name,
  type: 'line',
  x: TIME,
  y: Array.from(y),
  ...extra,
})

/** One component of the decomposition on its own small chart. */
function PartPlot({ name, y, emphasis }: { name: string; y: ArrayLike<number>; emphasis: boolean }) {
  const xAxis = useAxis({ range: [0, N - 1] })
  const yAxis = useAxis({ hold: 'union' })
  const series = useMemo(() => [line(name, y, emphasis ? { emphasis: true } : { slot: 0 })], [name, y, emphasis])
  return (
    <Plot x={xAxis} y={yAxis} height={150}>
      {seriesLayers(series)}
    </Plot>
  )
}

/**
 * Step through the sifting of one IMF: extrema, cubic-spline envelopes through them, their mean, and the candidate
 * IMF left after subtracting it. The tones' frequencies and amplitudes are handles on a line spectrum.
 */
export function SiftingExplorer() {
  const state = useFigureState({
    step: int(1, { min: 1, max: SIFTS, step: 1, label: 'sift', format: (v) => `${v}` }),
    which: choice<'0' | '1' | '2'>(
      [
        { value: '0', label: 'IMF 1' },
        { value: '1', label: 'IMF 2' },
        { value: '2', label: 'IMF 3' },
      ],
      '0',
      { label: 'sifting for' },
    ),
    ruleName: choice<RuleName>(
      [
        { value: 'sd', label: 'SD' },
        { value: 'snumber', label: 'S-number' },
        { value: 'rilling', label: 'Rilling' },
        { value: 'fixed', label: 'fixed' },
      ],
      'sd',
      { label: 'stopping rule' },
    ),
    mirror: setting(true, 'mirror extrema at the ends'),
    burst: setting(false, 'faster tone as a burst'),
    f1: slider(0.005, 0.15, 0.06, { step: 0.001, onChart: true }),
    a1: slider(0.1, 1.5, 1, { step: 0.01, onChart: true }),
    f2: slider(0.005, 0.15, 0.015, { step: 0.001, onChart: true }),
    a2: slider(0.1, 1.5, 0.8, { step: 0.01, onChart: true }),
  })

  const signal = useMemo(
    () => makeSignal(state.f1, state.a1, state.f2, state.a2, state.burst),
    [state.f1, state.a1, state.f2, state.a2, state.burst],
  )

  const result = useMemo(() => {
    const rule = RULES[state.ruleName]
    const d = emd(signal.x, { rule, mirror: state.mirror, maxImfs: 5 })
    // The signal the chosen IMF is sifted from: x minus the IMFs before it.
    const j = Math.min(Number(state.which), d.imfs.length)
    const start = Float64Array.from(signal.x, (v, i) => v - d.imfs.slice(0, j).reduce((s, c) => s + c[i], 0))
    const shown = sift(start, { kind: 'fixed', sifts: SIFTS }, state.mirror)
    const stopsAt = sift(start, rule, state.mirror).steps.length
    return { d, j, steps: shown.steps, stopsAt }
  }, [signal, state.ruleName, state.mirror, state.which])

  const k = Math.min(state.step, result.steps.length)
  const s = k > 0 ? result.steps[k - 1] : null

  const readout = useMemo(() => {
    if (!s) return null
    const e = findExtrema(s.next)
    const next = result.steps[k]
    const r = next ? rillingStats(next, 0.05) : null
    return {
      sd: sdCriterion(s.h, s.next),
      extrema: e.maxima.length + e.minima.length,
      zeros: e.zeroCrossings,
      frac: r?.fractionAbove,
      maxSigma: r?.maxSigma,
    }
  }, [s, result.steps, k])

  const siftSeries: SeriesSpec[] = s
    ? [
        line('h (being sifted)', s.h, { slot: 0 }),
        line('upper envelope', s.upper, { slot: 1 }),
        line('lower envelope', s.lower, { slot: 1, dashed: true }),
        line('mean of envelopes', s.mean, { emphasis: true }),
        { name: 'maxima', type: 'scatter', x: s.maxima, y: s.maxima.map((i) => s.h[i]), slot: 1 },
        { name: 'minima', type: 'scatter', x: s.minima, y: s.minima.map((i) => s.h[i]), slot: 2 },
      ]
    : []

  const truth = result.j < 2 ? signal.parts[result.j] : null
  const candidateSeries: SeriesSpec[] = s
    ? [
        ...(truth ? [line(result.j === 0 ? 'faster tone' : 'slower tone', truth, { muted: true, dashed: true })] : []),
        line(`candidate IMF after sift ${k}`, s.next, { slot: 0 }),
      ]
    : []

  const spectrum = [
    { name: 'faster tone', x: [state.f1], y: [state.a1], slot: 0 },
    { name: 'slower tone', x: [state.f2], y: [state.a2], slot: 0 },
  ] as const

  const parts = [
    ...result.d.imfs.map((c, i) => ({ name: `IMF ${i + 1}`, y: c })),
    { name: 'residue', y: result.d.residue },
  ]

  const xAxis = useAxis({ label: 'frequency (cycles/sample)', range: [0, 0.15] })
  const yAxis = useAxis({ label: 'amplitude', range: [0, 1.5] })
  const xAxis2 = useAxis({ label: 'sample', range: [0, N - 1] })
  const yAxis2 = useAxis({ hold: 'union' })
  const xAxis3 = useAxis({ range: [0, N - 1] })
  const yAxis3 = useAxis({ hold: 'union' })
  return (
    <Figure
      title="Sifting, one step at a time"
      state={state}
      caption="Drag the two tones on the line spectrum to set their frequencies (cycles per sample) and amplitudes; a slow linear trend is always added. Step through the sifts: the top panel shows the signal being sifted, its maxima and minima, the not-a-knot cubic-spline envelopes through them, and their mean (black). The middle panel shows what is left after the mean is subtracted, against the true component. The bottom panels show the full decomposition under the chosen stopping rule. Stopping rules: SD below 0.2, S-number 4, Rilling's thresholds, or a fixed 10 sifts. Switch off mirroring to see the envelopes swing at the ends, turn on the burst to see mode mixing, and bring the tones within a frequency ratio of about 0.7 to see them extracted as one IMF."

      readouts={
        readout && (
          <>
            <Readout label="SD this sift" value={formatNumber(readout.sd)} />
            <Readout label="extrema / zero crossings" value={`${readout.extrema} / ${readout.zeros}`} />
            {readout.frac !== undefined && (
              <Readout
                label="Rilling: share with σ > 0.05, max σ"
                value={`${formatNumber(readout.frac)}, ${formatNumber(readout.maxSigma ?? 0)}`}
              />
            )}
            <Readout label="chosen rule stops after" value={`${result.stopsAt} sifts`} />
          </>
        )
      }
    >
      <div className="grid gap-4 md:grid-cols-[1fr_2fr]">
        <div className="min-w-0 space-y-1">
          <div className="text-center text-xs text-muted-foreground">line spectrum (drag the tones)</div>
          <Plot x={xAxis} y={yAxis} height={260}>
            <Points {...spectrum[0]} />
            <Points {...spectrum[1]} />
            <Segments
              segments={[
                { from: [state.f1, 0], to: [state.f1, state.a1] },
                { from: [state.f2, 0], to: [state.f2, state.a2] },
              ]}
            />
            <Handle
              kind="point"
              at={[state.f1, state.a1]}
              onDrag={([f, a]) => (state.set('f1', f), state.set('a1', a))}
              label="tone 1"
            />
            <Handle
              kind="point"
              at={[state.f2, state.a2]}
              onDrag={([f, a]) => (state.set('f2', f), state.set('a2', a))}
              label="tone 2"
            />
          </Plot>
        </div>
        <div className="min-w-0 space-y-1">
          <div className="text-center text-xs text-muted-foreground">
            {s ? `sift ${k} of IMF ${result.j + 1}: extrema, envelopes and their mean` : 'no extrema left to sift'}
          </div>
          <Plot x={xAxis2} y={yAxis2} height={260}>
            {seriesLayers(siftSeries)}
          </Plot>
        </div>
      </div>
      <div className="min-w-0 space-y-1">
        <div className="text-center text-xs text-muted-foreground">candidate IMF: h minus the mean</div>
        <Plot x={xAxis3} y={yAxis3} height={170}>
          {seriesLayers(candidateSeries)}
        </Plot>
      </div>
      <div className="grid gap-2 sm:grid-cols-2">
        {parts.map((p, i) => (
          <div key={p.name} className="min-w-0">
            <div className="text-center text-xs text-muted-foreground">{p.name}</div>
            <PartPlot name={p.name} y={p.y} emphasis={i === parts.length - 1} />
          </div>
        ))}
      </div>
    </Figure>
  )
}
