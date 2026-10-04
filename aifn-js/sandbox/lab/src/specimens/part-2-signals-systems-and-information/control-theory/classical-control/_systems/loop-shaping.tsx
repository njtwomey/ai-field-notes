/**
 * "Root locus, Bode and Nyquist": one open loop K·L(s) given by draggable poles and zeros, read three ways at once:
 * the root locus with the closed-loop poles at K, the Bode plot with its margins, and the Nyquist plot with its
 * encirclements of −1 (core `rootLocus`, `bode`, `margins`, `nyquist`, `routhArray`).
 */
import { useMemo, useState } from 'react'
import { bode, margins, nyquist, rootLocus, routhArray, zerosPolesGain } from 'aifn/systems'
import { toFlat, type ComplexNumber } from 'aifn/foundation/tensor'
import { Figure } from '@lab/layout'
import { MARKER_SHAPES } from '@lab/design/palette'
import { slider, useFigureState, variants } from '@lab/state'
import { Annotation, Curve, Handle, Plot, Plots, Points, Readout, Segments, formatNumber, useAxis } from '@lab/viz'

const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(v) : '∞')
type C = ComplexNumber

const LOOPS: Record<string, { label: string; p: C[]; z: C[]; logK: number }> = {
  third: {
    label: '1/(s(s + 1)(s + 2))',
    p: [
      { re: 0, im: 0 },
      { re: -1, im: 0 },
      { re: -2, im: 0 },
    ],
    z: [],
    logK: 0.3,
  },
  resonant: {
    label: '(s + 3)/(s(s² + 0.4s + 4))',
    p: [
      { re: 0, im: 0 },
      { re: -0.2, im: 1.99 },
      { re: -0.2, im: -1.99 },
    ],
    z: [{ re: -3, im: 0 }],
    logK: 0,
  },
  unstable: {
    label: '(s + 2)/((s − 1)(s + 4))',
    p: [
      { re: 1, im: 0 },
      { re: -4, im: 0 },
    ],
    z: [{ re: -2, im: 0 }],
    logK: 0.5,
  },
}
const gain = (k: number) => ({ logK: slider(-2, 2.5, k, { label: 'log₁₀ gain K', step: 0.02 }) })
const LOOP = variants(
  Object.fromEntries(Object.entries(LOOPS).map(([k, v]) => [k, { label: v.label, params: gain(v.logK) }])) as Record<
    'third' | 'resonant' | 'unstable',
    { label: string; params: ReturnType<typeof gain> }
  >,
  { label: '1 · open loop', choiceLabel: 'L(s) to start from' },
)
const W = Array.from({ length: 500 }, (_, i) => 10 ** (-2 + (4 * i) / 499))

export function LoopShapingSpecimen() {
  const state = useFigureState({ loop: LOOP })
  const key = state.loop.key as keyof typeof LOOPS
  const K = 10 ** state.loop.values.logK
  const [edit, setEdit] = useState<{ key: string; p: C[]; z: C[] } | null>(null)
  const pz = useMemo(() => (edit && edit.key === key ? edit : { key, p: LOOPS[key].p, z: LOOPS[key].z }), [edit, key])
  const move = (which: 'z' | 'p', i: number, [x, y]: readonly [number, number]) => {
    const list = (which === 'z' ? pz.z : pz.p).map((c) => ({ ...c }))
    const r0 = list[i]
    if (Math.abs(r0.im) < 1e-9) list[i] = { re: x, im: 0 }
    else {
      const j = list.findIndex((c, k) => k !== i && Math.abs(c.re - r0.re) < 1e-9 && Math.abs(c.im + r0.im) < 1e-9)
      const im = Math.max(0.05, Math.abs(y))
      list[i] = { re: x, im }
      if (j >= 0) list[j] = { re: x, im: -im }
    }
    setEdit({ key, p: which === 'p' ? list : pz.p, z: which === 'z' ? list : pz.z })
  }
  const L1 = useMemo(() => zerosPolesGain(pz.z, pz.p, 1), [pz])
  const LK = useMemo(() => zerosPolesGain(pz.z, pz.p, K), [pz, K])
  const locus = useMemo(() => rootLocus(L1, { points: 240 }), [L1])
  const atK = useMemo(() => rootLocus(L1, { gains: [K] }).branches.map((b) => b[1]), [L1, K])
  const b = useMemo(() => bode(LK, W), [LK])
  const m = useMemo(() => margins(LK), [LK])
  const ny = useMemo(() => nyquist(LK), [LK])
  const routh = useMemo(() => {
    // The closed-loop characteristic polynomial den + K·num, from the locus' own closed-loop poles.
    const poly = atK.reduce<{ re: number; im: number }[]>(
      (acc, r) => {
        const out: { re: number; im: number }[] = acc.map(() => ({ re: 0, im: 0 })).concat([{ re: 0, im: 0 }])
        acc.forEach((a, i) => {
          out[i].re += a.re
          out[i].im += a.im
          out[i + 1].re -= a.re * r.re - a.im * r.im
          out[i + 1].im -= a.re * r.im + a.im * r.re
        })
        return out
      },
      [{ re: 1, im: 0 }],
    )
    return routhArray(poly.map((c) => c.re))
  }, [atK])
  const branches = useMemo(
    () => ({
      x: locus.branches.flatMap((br) => [...br.map((c) => c.re), NaN]),
      y: locus.branches.flatMap((br) => [...br.map((c) => c.im), NaN]),
    }),
    [locus],
  )
  const asym = locus.asymptotes.angles.map((a) => ({
    from: [locus.asymptotes.centroid, 0] as const,
    to: [locus.asymptotes.centroid + 8 * Math.cos(a), 8 * Math.sin(a)] as const,
  }))
  const nyRe = useMemo(() => toFlat(ny.re).map((v) => Math.max(-12, Math.min(12, v))), [ny])
  const nyIm = useMemo(() => toFlat(ny.im).map((v) => Math.max(-12, Math.min(12, v))), [ny])
  const db = useMemo(() => toFlat(b.magnitudeDb), [b])
  const ph = useMemo(() => toFlat(b.phase), [b])
  const draggable = (which: 'z' | 'p') =>
    (which === 'z' ? pz.z : pz.p).map((c, i) => ({ c, i })).filter(({ c }) => c.im >= -1e-9)
  const re = useAxis({ label: 'Re s', range: [-6, 2.5] })
  const im = useAxis({ label: 'Im s', range: [-4, 4] })
  const w = useAxis({ label: 'ω (rad/s)', log: true, range: [W[0], W[W.length - 1]] })
  const mag = useAxis({ label: '|KL| (dB)', range: [-80, 60] })
  const pha = useAxis({ label: '∠KL (°)', hold: 'union', key })
  const nre = useAxis({ label: 'Re KL(iω)', range: [-4, 2] })
  const nim = useAxis({ label: 'Im KL(iω)', range: [-3, 3] })
  const stable = routh.rightHalfPlane === 0 && routh.imaginaryAxis === 0
  return (
    <Figure
      title="Root locus, Bode and Nyquist"
      purpose="Three readings of one feedback loop: where the closed-loop poles sit as the gain grows, how much gain and phase the open loop has to spare, and how many times its Nyquist plot circles −1. All three agree on stability."
      defaultSize="XL"
      state={state}
      readouts={{
        'closed loop at K': (
          <>
            <Readout label="K" value={fmt(K)} />
            <Readout
              label="closed-loop poles"
              value={atK
                .map(
                  (c) => `${fmt(c.re)}${Math.abs(c.im) > 1e-9 ? `${c.im < 0 ? '−' : '+'}${fmt(Math.abs(c.im))}i` : ''}`,
                )
                .join(', ')}
            />
            <Readout label="Routh: right-half-plane poles" value={routh.rightHalfPlane} />
            <Readout label="verdict" value={stable ? 'stable' : 'unstable'} />
          </>
        ),
        nyquist: (
          <>
            <Readout label="open-loop unstable P" value={ny.openLoopUnstable} />
            <Readout label="clockwise encirclements N" value={ny.encirclements} />
            <Readout label="closed-loop unstable Z = N + P" value={ny.closedLoopUnstable} />
          </>
        ),
        margins: (
          <>
            <Readout label="gain margin" value={`${fmt(m.gainMarginDb)} dB`} />
            <Readout label="phase margin" value={`${fmt(m.phaseMargin)}°`} />
            <Readout label="crossing gains" value={locus.crossings.map((c) => fmt(c.gain)).join(', ') || 'none'} />
          </>
        ),
      }}
      caption="Drag any open-loop pole (cross) or zero (circle) in the upper half-plane of the root-locus panel by its ink mark of the same shape; its conjugate follows and real ones slide along the axis. The grey curves are the root locus for K from 0 to ∞, dashed rays its asymptotes; the ink dots are the closed-loop poles at the gain K on the slider. The Bode panels show K·L(iω) with 0 dB and −180° marked; the Nyquist panel shows K·L along the indented contour (clipped to ±12) and the critical point −1. A closed-loop pole crossing into the right half-plane on the locus is the gain at which the Bode margins reach zero and the Nyquist plot first passes through −1. Changing the starting loop restores its poles and zeros."
    >
      <Plots cols={3} widths={[1.1, 1, 1]}>
        <Plot x={re} y={im} title="root locus">
          <Annotation x={0} />
          <Curve name="locus, K from 0 to ∞" x={branches.x} y={branches.y} muted />
          <Segments name="asymptotes" segments={asym} dashed muted />
          <Points
            name="open-loop poles"
            x={pz.p.map((c) => c.re)}
            y={pz.p.map((c) => c.im)}
            slot={0}
            shape={4}
            size={11}
          />
          {pz.z.length > 0 && (
            <Points
              name="open-loop zeros"
              x={pz.z.map((c) => c.re)}
              y={pz.z.map((c) => c.im)}
              slot={1}
              shape={0}
              size={9}
            />
          )}
          <Points name="closed-loop poles at K" x={atK.map((c) => c.re)} y={atK.map((c) => c.im)} emphasis live />
          {draggable('p').map(({ c, i }) => (
            <Handle
              key={`p${i}`}
              kind="point"
              symbol={MARKER_SHAPES[4]}
              at={[c.re, c.im]}
              onDrag={(v) => move('p', i, v)}
            />
          ))}
          {draggable('z').map(({ c, i }) => (
            <Handle key={`z${i}`} kind="point" at={[c.re, c.im]} onDrag={(v) => move('z', i, v)} />
          ))}
        </Plot>
        <Plots rows={2} hoverGroup>
          <Plot x={w} y={mag} legend={false}>
            <Annotation y={0} />
            <Curve name="|KL| (dB)" x={W} y={db} slot={0} />
            {Number.isFinite(m.phaseCrossover) && <Annotation x={m.phaseCrossover} dashed />}
          </Plot>
          <Plot x={w} y={pha} legend={false}>
            <Annotation y={-180} />
            <Curve name="∠KL (°)" x={W} y={ph} slot={1} />
            {Number.isFinite(m.gainCrossover) && <Annotation x={m.gainCrossover} dashed />}
          </Plot>
        </Plots>
        <Plot x={nre} y={nim} title="Nyquist">
          <Curve name="KL along the contour" x={nyRe} y={nyIm} slot={0} />
          <Points name="−1" x={[-1]} y={[0]} tone="destructive" size={9} />
        </Plot>
      </Plots>
    </Figure>
  )
}
