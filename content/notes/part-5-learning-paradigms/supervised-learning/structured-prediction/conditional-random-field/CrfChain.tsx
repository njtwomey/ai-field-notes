import { useMemo, useState } from 'react'
import {
  Heatmap,
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  formatNumber,
  useParam,
  type HeatmapOverlay,
} from 'aifn-render'
import { rng } from '@/lib/math'
import { forwardBackward, rankOne, viterbi, type Mat } from '../_shared/chain-crf'

const N = 6
const POSITIONS = Array.from({ length: N }, (_, i) => i + 1)

type Labels = '2' | '3'
type Rank = 'full' | 'one'

/**
 * A linear-chain CRF over six positions. Node potentials ψ_n come from fixed per-position evidence scaled by a strength;
 * the edge potential Ψ rewards staying on the same label. The grid shows P(Y_n = y | x); the overlays compare the
 * Viterbi path with decoding each position by its own marginal.
 */
export function CrfChain() {
  const [labels, setLabels] = useState<Labels>('3')
  const [rank, setRank] = useState<Rank>('full')
  const evidence = useParam(1.2, { min: 0, max: 3, step: 0.1 })
  const sticky = useParam(1.5, { min: -2, max: 3, step: 0.1 })
  const seed = useParam(4, { min: 1, max: 20, step: 1 })
  const k = Number(labels)
  const ys = Array.from({ length: k }, (_, i) => i + 1)

  const r = useMemo(() => {
    const g = rng(seed.value)
    // Per-position evidence scores for each label, fixed by the seed; ψ_n(y) = exp(strength · score).
    const scores = POSITIONS.map(() => Array.from({ length: 3 }, () => g.normal()).slice(0, k))
    const psi = scores.map((row) => row.map((s) => Math.exp(evidence.value * s)))
    // Ψ(u, v) = e^s on the diagonal and 1 off it: a positive s favours keeping the label.
    const full: Mat = ys.map((_, u) => ys.map((__, v) => (u === v ? Math.exp(sticky.value) : 1)))
    const one = rankOne(full)
    const edge = Array.from({ length: N - 1 }, () => (rank === 'one' ? one : full))
    const inf = forwardBackward(psi, edge)
    const best = viterbi(psi, edge)
    const byMarginal = inf.marginals.map((m) => m.indexOf(Math.max(...m)))
    return {
      inf,
      best,
      byMarginal,
      logZ: Math.log(inf.z),
      agree: best.path.every((y, i) => y === byMarginal[i]),
    }
    // ys is derived from k.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [k, rank, evidence.value, sticky.value, seed.value])

  // z[i][j] = P(Y_{j+1} = i+1): rows are labels, columns positions.
  const z = ys.map((_, y) => r.inf.marginals.map((m) => m[y]))
  const overlay: HeatmapOverlay[] = [
    {
      name: 'Viterbi path',
      type: 'line',
      x: POSITIONS,
      y: r.best.path.map((y) => y + 1),
      showPoints: true,
      slot: 1,
    },
    {
      name: 'most probable label per position',
      type: 'scatter',
      x: POSITIONS,
      y: r.byMarginal.map((y) => y + 1),
      emphasis: true,
    },
  ]

  return (
    <Interactive
      title="Marginals and the Viterbi path on a chain"
      caption="Each column is a position n and each row a label y; the colour is the marginal P(Yₙ = y | x) from forward–backward. Evidence strength scales the node potentials ψₙ; stickiness sets the diagonal of the edge potential Ψ, so a large value makes neighbours agree. The line is the Viterbi path, the single most probable sequence; the diamonds pick each position's most probable label on its own, and the two can disagree. Switching Ψ to its rank-1 approximation makes the positions independent: the marginals then depend only on each position's own evidence."
      controls={
        <>
          <ParamSlider label="evidence strength" param={evidence} />
          <ParamSlider label="stickiness s (Ψ diagonal = eˢ)" param={sticky} />
          <ParamSlider label="observations (seed)" param={seed} format={(v) => String(v)} withArrows />
          <ParamChoice
            label="labels |𝒴|"
            value={labels}
            onChange={setLabels}
            options={[
              { value: '2', label: '2' },
              { value: '3', label: '3' },
            ]}
          />
          <ParamChoice
            label="edge potential Ψ"
            value={rank}
            onChange={setRank}
            options={[
              { value: 'full', label: 'full rank' },
              { value: 'one', label: 'rank 1' },
            ]}
          />
        </>
      }
      readout={
        <>
          <Readout label="log Z" value={formatNumber(r.logZ)} />
          <Readout label="P(Viterbi path | x)" value={formatNumber(Math.exp(r.best.score - r.logZ))} />
          <Readout label="Viterbi vs per-position decoding" value={r.agree ? 'agree' : 'disagree'} />
        </>
      }
    >
      <Heatmap
        x={POSITIONS}
        y={ys}
        z={z}
        range={[0, 1]}
        xLabel="position n"
        yLabel="label y"
        valueLabel="P(Yₙ = y | x)"
        overlay={overlay}
        height={280}
      />
    </Interactive>
  )
}
