import { useMemo, useState } from 'react'
import { Heatmap, Interactive, ParamChoice, ParamSlider, Readout, formatNumber } from '@/components/viz'

type Kind = 'gpipe' | '1f1b'
const KINDS = [
  { value: 'gpipe', label: 'GPipe' },
  { value: '1f1b', label: '1F1B' },
] as const

type Op = { backward: boolean; mb: number }

// A backward pass takes about twice as long as a forward pass: it computes gradients for inputs and for weights.
const T_F = 1
const T_B = 2

/** Order of operations on each stage. GPipe: all forwards, then all backwards. 1F1B: warm up, alternate, drain. */
function orders(p: number, m: number, kind: Kind): Op[][] {
  return Array.from({ length: p }, (_, s) => {
    const f = (mb: number): Op => ({ backward: false, mb })
    const b = (mb: number): Op => ({ backward: true, mb })
    if (kind === 'gpipe')
      return [...Array.from({ length: m }, (_, j) => f(j)), ...Array.from({ length: m }, (_, j) => b(j))]
    const warm = Math.min(p - s - 1, m)
    const out: Op[] = Array.from({ length: warm }, (_, j) => f(j))
    let nf = warm
    let nb = 0
    while (nf < m) out.push(f(nf++), b(nb++))
    while (nb < m) out.push(b(nb++))
    return out
  })
}

/** List scheduling: each op starts when its stage is free and its dependency on the neighbouring stage has finished. */
function schedule(p: number, m: number, kind: Kind) {
  const ops = orders(p, m, kind)
  const next = new Array<number>(p).fill(0)
  const free = new Array<number>(p).fill(0)
  const done = new Map<string, number>()
  const key = (backward: boolean, mb: number, s: number) => `${backward ? 'B' : 'F'}${mb}@${s}`
  const placed: { s: number; start: number; op: Op }[] = []
  while (next.some((i, s) => i < ops[s].length)) {
    for (let s = 0; s < p; s++) {
      if (next[s] >= ops[s].length) continue
      const op = ops[s][next[s]]
      const dep = op.backward
        ? s < p - 1
          ? key(true, op.mb, s + 1)
          : key(false, op.mb, s)
        : s > 0
          ? key(false, op.mb, s - 1)
          : null
      if (dep !== null && !done.has(dep)) continue
      const start = Math.max(free[s], dep === null ? 0 : done.get(dep)!)
      const end = start + (op.backward ? T_B : T_F)
      done.set(key(op.backward, op.mb, s), end)
      free[s] = end
      next[s]++
      placed.push({ s, start, op })
    }
  }
  const total = Math.max(...free)
  // +1 forward, −1 backward, 0 idle, one column per forward-pass time unit.
  const z = Array.from({ length: p }, () => new Array<number>(total).fill(0))
  for (const { s, start, op } of placed)
    for (let t = start; t < start + (op.backward ? T_B : T_F); t++) z[s][t] = op.backward ? -1 : 1
  // Peak number of micro-batches whose activations stage 0 holds: forwards done minus backwards done.
  let live = 0
  let peak = 0
  for (const op of ops[0]) {
    live += op.backward ? -1 : 1
    peak = Math.max(peak, live)
  }
  return { z, total, peak }
}

export function PipelineSchedule() {
  const [p, setP] = useState(4)
  const [m, setM] = useState(8)
  const [kind, setKind] = useState<Kind>('1f1b')
  const { z, total, peak } = useMemo(() => schedule(p, m, kind), [p, m, kind])
  const x = useMemo(() => Array.from({ length: total }, (_, t) => t), [total])
  const y = useMemo(() => Array.from({ length: p }, (_, s) => s + 1), [p])
  const idle = z.flat().filter((v) => v === 0).length / (p * total)

  return (
    <Interactive
      title="Pipeline schedules and the bubble"
      caption="Each row is one pipeline stage and each column one forward-pass time. Red cells are forward passes and blue cells backward passes, which take twice as long. Grey cells are the bubble, where a stage waits for its neighbour. GPipe and 1F1B have the same bubble, (p − 1)/(m + p − 1) of the time, but 1F1B starts backward passes early, so each stage holds the activations of at most p micro-batches instead of m."
      controls={
        <>
          <ParamChoice label="schedule" value={kind} onChange={setKind} options={KINDS} />
          <ParamSlider label="stages p" value={p} onChange={setP} min={2} max={8} step={1} />
          <ParamSlider label="micro-batches m" value={m} onChange={setM} min={1} max={16} step={1} />
        </>
      }
      readout={
        <>
          <Readout label="total time" value={`${total} forward units`} />
          <Readout label="idle fraction" value={formatNumber(idle)} />
          <Readout label="(p − 1)/(m + p − 1)" value={formatNumber((p - 1) / (m + p - 1))} />
          <Readout label="activations held by stage 1" value={`${peak} micro-batches`} />
        </>
      }
    >
      <Heatmap
        x={x}
        y={y}
        z={z}
        xLabel="time"
        yLabel="stage"
        scale="diverging"
        range={[-1, 1]}
        valueLabel="forward (+1) / backward (−1)"
        height={60 + 32 * p}
      />
    </Interactive>
  )
}
