import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  ParamSwitch,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
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

const line = (name: string, y: ArrayLike<number>, extra: Partial<XYSeries> = {}): XYSeries => ({
  name,
  type: 'line',
  x: TIME,
  y: Array.from(y),
  ...extra,
})

/**
 * Step through the sifting of one IMF: extrema, cubic-spline envelopes through them, their mean, and the candidate
 * IMF left after subtracting it. The tones' frequencies and amplitudes are handles on a line spectrum.
 */
export function SiftingExplorer() {
  const f1 = useParam(0.06, { min: 0.005, max: 0.15, step: 0.001 })
  const a1 = useParam(1, { min: 0.1, max: 1.5, step: 0.01 })
  const f2 = useParam(0.015, { min: 0.005, max: 0.15, step: 0.001 })
  const a2 = useParam(0.8, { min: 0.1, max: 1.5, step: 0.01 })
  const step = useParam(1, { min: 1, max: SIFTS, step: 1 })
  const [which, setWhich] = useState<'0' | '1' | '2'>('0')
  const [ruleName, setRuleName] = useState<RuleName>('sd')
  const [mirror, setMirror] = useState(true)
  const [burst, setBurst] = useState(false)

  const signal = useMemo(
    () => makeSignal(f1.value, a1.value, f2.value, a2.value, burst),
    [f1.value, a1.value, f2.value, a2.value, burst],
  )

  const result = useMemo(() => {
    const rule = RULES[ruleName]
    const d = emd(signal.x, { rule, mirror, maxImfs: 5 })
    // The signal the chosen IMF is sifted from: x minus the IMFs before it.
    const j = Math.min(Number(which), d.imfs.length)
    const start = Float64Array.from(signal.x, (v, i) => v - d.imfs.slice(0, j).reduce((s, c) => s + c[i], 0))
    const shown = sift(start, { kind: 'fixed', sifts: SIFTS }, mirror)
    const stopsAt = sift(start, rule, mirror).steps.length
    return { d, j, steps: shown.steps, stopsAt }
  }, [signal, ruleName, mirror, which])

  const k = Math.min(step.value, result.steps.length)
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

  const siftSeries: XYSeries[] = s
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
  const candidateSeries: XYSeries[] = s
    ? [
        ...(truth ? [line(result.j === 0 ? 'faster tone' : 'slower tone', truth, { muted: true, dashed: true })] : []),
        line(`candidate IMF after sift ${k}`, s.next, { slot: 0 }),
      ]
    : []

  const spectrum: XYSeries[] = [
    { name: 'faster tone', type: 'scatter', x: [f1.value], y: [a1.value], slot: 0 },
    { name: 'slower tone', type: 'scatter', x: [f2.value], y: [a2.value], slot: 0 },
  ]
  const handles: Handle[] = [
    { kind: 'point', at: [f1.value, a1.value], onDrag: ([f, a]) => (f1.set(f), a1.set(a)), label: 'tone 1' },
    { kind: 'point', at: [f2.value, a2.value], onDrag: ([f, a]) => (f2.set(f), a2.set(a)), label: 'tone 2' },
  ]

  const parts = [
    ...result.d.imfs.map((c, i) => ({ name: `IMF ${i + 1}`, y: c })),
    { name: 'residue', y: result.d.residue },
  ]

  return (
    <Interactive
      title="Sifting, one step at a time"
      caption="Drag the two tones on the line spectrum to set their frequencies (cycles per sample) and amplitudes; a slow linear trend is always added. Step through the sifts: the top panel shows the signal being sifted, its maxima and minima, the not-a-knot cubic-spline envelopes through them, and their mean (black). The middle panel shows what is left after the mean is subtracted, against the true component. The bottom panels show the full decomposition under the chosen stopping rule. Stopping rules: SD below 0.2, S-number 4, Rilling's thresholds, or a fixed 10 sifts. Switch off mirroring to see the envelopes swing at the ends, turn on the burst to see mode mixing, and bring the tones within a frequency ratio of about 0.7 to see them extracted as one IMF."
      controls={
        <>
          <ParamSlider label="sift" param={step} withArrows format={(v) => `${v}`} />
          <ParamChoice
            label="sifting for"
            value={which}
            onChange={setWhich}
            options={[
              { value: '0', label: 'IMF 1' },
              { value: '1', label: 'IMF 2' },
              { value: '2', label: 'IMF 3' },
            ]}
          />
          <ParamChoice
            label="stopping rule"
            value={ruleName}
            onChange={setRuleName}
            options={[
              { value: 'sd', label: 'SD' },
              { value: 'snumber', label: 'S-number' },
              { value: 'rilling', label: 'Rilling' },
              { value: 'fixed', label: 'fixed' },
            ]}
          />
          <ParamSwitch label="mirror extrema at the ends" checked={mirror} onChange={setMirror} />
          <ParamSwitch label="faster tone as a burst" checked={burst} onChange={setBurst} />
        </>
      }
      readout={
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
          <XYChart
            series={spectrum}
            segments={[
              { from: [f1.value, 0], to: [f1.value, a1.value] },
              { from: [f2.value, 0], to: [f2.value, a2.value] },
            ]}
            handles={handles}
            xRange={[0, 0.15]}
            yRange={[0, 1.5]}
            xLabel="frequency (cycles/sample)"
            yLabel="amplitude"
            height={260}
          />
        </div>
        <div className="min-w-0 space-y-1">
          <div className="text-center text-xs text-muted-foreground">
            {s ? `sift ${k} of IMF ${result.j + 1}: extrema, envelopes and their mean` : 'no extrema left to sift'}
          </div>
          <XYChart series={siftSeries} xLabel="sample" xRange={[0, N - 1]} height={260} />
        </div>
      </div>
      <div className="min-w-0 space-y-1">
        <div className="text-center text-xs text-muted-foreground">candidate IMF: h minus the mean</div>
        <XYChart series={candidateSeries} xRange={[0, N - 1]} height={170} />
      </div>
      <div className="grid gap-2 sm:grid-cols-2">
        {parts.map((p, i) => (
          <div key={p.name} className="min-w-0">
            <div className="text-center text-xs text-muted-foreground">{p.name}</div>
            <XYChart
              series={[line(p.name, p.y, i === parts.length - 1 ? { emphasis: true } : { slot: 0 })]}
              xRange={[0, N - 1]}
              height={150}
            />
          </div>
        ))}
      </div>
    </Interactive>
  )
}
