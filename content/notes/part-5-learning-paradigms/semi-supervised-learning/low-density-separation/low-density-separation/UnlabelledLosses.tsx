import { useState } from 'react'
import { Curve, Figure, formatNumber, Handle, Plot, Points, Readout, useAxis } from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { binaryEntropy, sigmoid } from 'aifn-compute/numerics/special'
import { symmetricHinge, symmetricSigmoid } from 'aifn-methods/learning/semi-supervised'

const Z = toFlat(linspace(-3, 3, 241))
/** The loss each method puts on one unlabelled point with score z, each scaled to 1 at z = 0. */
const LOSSES = [
  {
    name: 'entropy of σ(z), bits',
    short: 'entropy (bits)',
    f: (z: number) => binaryEntropy(sigmoid(z), 2),
    style: { emphasis: true } as const,
  },
  {
    name: 'hard pseudo-label loss −log₂ max(σ(z), 1 − σ(z))',
    short: 'pseudo-label loss',
    f: (z: number) => -Math.log2(Math.max(sigmoid(z), 1 - sigmoid(z))),
    style: { slot: 0 } as const,
  },
  {
    name: 'symmetric hinge max(0, 1 − |z|)',
    short: 'symmetric hinge',
    f: (z: number) => symmetricHinge(z),
    style: { slot: 1 } as const,
  },
  {
    name: 'symmetric sigmoid exp(−3z²)',
    short: 'symmetric sigmoid',
    f: (z: number) => symmetricSigmoid(z),
    style: { slot: 2 } as const,
  },
]
const CURVES = LOSSES.map((l) => Z.map(l.f))

/** The per-point losses that entropy minimisation, pseudo-labelling and the transductive SVM put on unlabelled points. */
export function UnlabelledLosses() {
  const [z, setZ] = useState(0.5)
  const xa = useAxis({ label: 'score z of an unlabelled point', range: [-3, 3] })
  const ya = useAxis({ label: 'loss on the point', range: [0, 1.08] })
  return (
    <Figure
      title="Four losses on an unlabelled point"
      caption="The loss each method charges one unlabelled point as a function of its score z: the logit of a logistic model for the entropy and the pseudo-label loss, the decision value f(x) of an SVM for the two symmetric losses. Each is largest, 1, at z = 0, which is on the boundary, and falls as the point moves away from the boundary on either side; none depends on which side. Drag the vertical line to read the four losses at one score. The hinge is exactly zero beyond |z| = 1, so a point outside the margin exerts no force; the entropy and the pseudo-label loss never reach zero and keep pushing, more weakly, at every distance."
      readouts={
        <>
          {LOSSES.map((l) => (
            <Readout key={l.name} label={l.short} value={formatNumber(l.f(z))} />
          ))}
        </>
      }
    >
      <Plot x={xa} y={ya} height={320}>
        {LOSSES.map((l, k) => (
          <Curve key={l.name} name={l.name} x={Z} y={CURVES[k]} {...l.style} />
        ))}
        {LOSSES.map((l, k) => (
          <Points key={`p${k}`} name={l.name} x={[z]} y={[l.f(z)]} {...l.style} size={8} live />
        ))}
        <Handle kind="x" at={z} label="score z" onDrag={(v) => setZ(Math.max(-3, Math.min(3, v)))} />
      </Plot>
    </Figure>
  )
}
