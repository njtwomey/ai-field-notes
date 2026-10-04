import { useMemo } from 'react'
import { Figure, int, Plot, Raster, Readout, useAxis, useFigureState } from 'aifn-render'

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
  const xAxis = useAxis({ label: 'key j' })
  const yAxis = useAxis({ label: 'query i' })
  return (
    <div className="flex min-w-0 flex-col gap-1">
      <span className="text-center text-xs text-muted-foreground">{title}</span>
      <Plot x={xAxis} y={yAxis} height={240}>
        <Raster x={POS} y={POS} z={z} range={RANGE} valueLabel={'allowed'} />
      </Plot>
    </div>
  )
}

/** Attention masks of encoder, causal decoder and prefix language model, with an adjustable prefix. */
export function MaskPatterns() {
  const state = useFigureState({
    prefix: int(4, { min: 0, max: N, step: 1, label: 'prefix length P', format: (v) => v.toFixed(0) }),
  })
  const bi = useMemo(() => mask('bidirectional', 0), [])
  const causal = useMemo(() => mask('causal', 0), [])
  const pre = useMemo(() => mask('prefix', state.prefix), [state.prefix])
  return (
    <Figure
      title="Attention masks of the three families"
      state={state}
      caption={`Each panel is a ${N}-token sequence; a filled cell means query position i may attend to key position j. An encoder attends everywhere. A causal decoder attends to itself and earlier positions. A prefix language model attends bidirectionally within the first P positions and causally after them. P = 0 gives the causal mask and P = ${N} the bidirectional one. An encoder–decoder over the concatenation [source; target] has the prefix pattern with P equal to the source length, computed by two separate stacks.`}

      readouts={
        <>
          <Readout label="allowed pairs, encoder" value={`${allowed(bi)} of ${N * N}`} />
          <Readout label="causal decoder" value={`${allowed(causal)}`} />
          <Readout label="prefix LM" value={`${allowed(pre)}`} />
          <Readout label="tokens with a loss, prefix LM" value={`${N - Math.max(state.prefix, 1)}`} />
        </>
      }
    >
      <div className="grid gap-4 sm:grid-cols-3">
        <Panel title="bidirectional (encoder)" z={bi} />
        <Panel title="causal (decoder)" z={causal} />
        <Panel title={`prefix, P = ${state.prefix}`} z={pre} />
      </div>
    </Figure>
  )
}
