import { useMemo, useState } from 'react'
import { Heatmap, Interactive, ParamSlider, Readout } from 'aifn-render'

const N = 10
const POS = Array.from({ length: N }, (_, i) => i)
const RANGE: [number, number] = [0, 1]

/** 1 where query i may attend to key j. */
function mask(kind: 'bidirectional' | 'causal' | 'prefix', prefix: number): number[][] {
  return POS.map((i) =>
    POS.map((j) => {
      if (kind === 'bidirectional') return 1
      if (kind === 'causal') return j <= i ? 1 : 0
      return j < prefix || j <= i ? 1 : 0
    }),
  )
}

const allowed = (m: number[][]) => m.flat().reduce((a, b) => a + b, 0)

function Panel({ title, z }: { title: string; z: number[][] }) {
  return (
    <div className="flex min-w-0 flex-col gap-1">
      <span className="text-center text-xs text-muted-foreground">{title}</span>
      <Heatmap x={POS} y={POS} z={z} range={RANGE} xLabel="key j" yLabel="query i" valueLabel="allowed" height={240} />
    </div>
  )
}

/** Attention masks of encoder, causal decoder and prefix language model, with an adjustable prefix. */
export function MaskPatterns() {
  const [prefix, setPrefix] = useState(4)
  const bi = useMemo(() => mask('bidirectional', 0), [])
  const causal = useMemo(() => mask('causal', 0), [])
  const pre = useMemo(() => mask('prefix', prefix), [prefix])
  return (
    <Interactive
      title="Attention masks of the three families"
      caption={`Each panel is a ${N}-token sequence; a filled cell means query position i may attend to key position j. An encoder attends everywhere. A causal decoder attends to itself and earlier positions. A prefix language model attends bidirectionally within the first P positions and causally after them. P = 0 gives the causal mask and P = ${N} the bidirectional one. An encoder–decoder over the concatenation [source; target] has the prefix pattern with P equal to the source length, computed by two separate stacks.`}
      controls={
        <ParamSlider
          label="prefix length P"
          value={prefix}
          onChange={setPrefix}
          min={0}
          max={N}
          step={1}
          withArrows
          format={(v) => v.toFixed(0)}
        />
      }
      readout={
        <>
          <Readout label="allowed pairs, encoder" value={`${allowed(bi)} of ${N * N}`} />
          <Readout label="causal decoder" value={`${allowed(causal)}`} />
          <Readout label="prefix LM" value={`${allowed(pre)}`} />
          <Readout label="tokens with a loss, prefix LM" value={`${N - Math.max(prefix, 1)}`} />
        </>
      }
    >
      <div className="grid gap-4 sm:grid-cols-3">
        <Panel title="bidirectional (encoder)" z={bi} />
        <Panel title="causal (decoder)" z={causal} />
        <Panel title={`prefix, P = ${prefix}`} z={pre} />
      </div>
    </Interactive>
  )
}
