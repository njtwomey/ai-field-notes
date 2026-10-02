import { useState } from 'react'
import { Diagram } from 'aifn-render'
import { factor, link, variable } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'
import { Interactive, ParamChoice } from 'aifn-render'

type Target = 'x1' | 'x2'

/**
 * The bivariate Gaussian as a factor graph: unary factors φ₁, φ₂ and the pair factor ψ₁₂. Choosing a variable outlines
 * the factors that touch it, whose product is its full conditional, and shades its Markov blanket, the other variable.
 */
function spec(target: Target): DiagramSpec {
  const mine = target === 'x1' ? ['phi1', 'psi', 'x1'] : ['psi', 'phi2', 'x2']
  return {
    unit: 46,
    nodes: [
      factor('phi1', 0, 0, '$\\phi_1$', 'n'),
      factor('phi2', 4, 0, '$\\phi_2$', 'n'),
      variable('x1', 0, 1.8, '$x_1$', { filled: target === 'x2' }),
      factor('psi', 2, 1.8, '$\\psi_{12}$', 's'),
      variable('x2', 4, 1.8, '$x_2$', { filled: target === 'x1' }),
    ],
    edges: [link('phi1', 'x1', false), link('phi2', 'x2', false), link('x1', 'psi', false), link('psi', 'x2', false)],
    groups: [
      { id: 'conditional', around: mine, pad: 0.6, dashed: true, tone: 2, label: 'factors in the full conditional' },
    ],
  }
}

export function GibbsFactorGraph() {
  const [target, setTarget] = useState<Target>('x1')
  const other = target === 'x1' ? 'x₂' : 'x₁'
  const self = target === 'x1' ? 'x₁' : 'x₂'
  return (
    <Interactive
      title="The bivariate Gaussian as a factor graph"
      caption={`The joint density is φ₁(x₁) φ₂(x₂) ψ₁₂(x₁, x₂): one factor for each diagonal entry of the precision matrix and one for the off-diagonal entry. The dashed outline holds the factors that touch ${self}; their product is the full conditional of ${self}, because every other factor is constant in ${self}. The shaded variable, ${other}, is ${self}'s Markov blanket: the only variable the conditional depends on.`}
      controls={
        <ParamChoice
          label="full conditional of"
          value={target}
          onChange={setTarget}
          options={[
            { value: 'x1', label: 'x₁' },
            { value: 'x2', label: 'x₂' },
          ]}
        />
      }
    >
      <Diagram
        spec={spec(target)}
        ariaLabel="Factor graph with variables x1 and x2, unary factors phi1 and phi2, and a pair factor psi12 between them"
      />
    </Interactive>
  )
}
