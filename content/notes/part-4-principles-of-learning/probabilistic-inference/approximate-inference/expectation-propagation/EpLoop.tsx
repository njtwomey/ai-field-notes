import { Diagram } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'
import { Interactive } from 'aifn-render'

const SPEC: DiagramSpec = {
  unit: 40,
  nodes: [
    {
      id: 'q',
      x: 0,
      y: 0,
      w: 3.6,
      h: 1.2,
      tone: 'neutral',
      label: 'approximation\n$q \\propto p_0\\prod_j \\tilde t_j$',
    },
    { id: 'cav', x: 6, y: 0, w: 3.6, h: 1.2, tone: 0, label: 'cavity\n$q^{\\setminus i} \\propto q/\\tilde t_i$' },
    {
      id: 'tilt',
      x: 6,
      y: 3.2,
      w: 3.6,
      h: 1.2,
      tone: 1,
      label: 'tilted\n$\\hat p_i \\propto q^{\\setminus i}\\, t_i$',
    },
    { id: 'proj', x: 0, y: 3.2, w: 3.6, h: 1.2, tone: 2, label: 'project\n$q^{\\text{new}}$: moments of $\\hat p_i$' },
    { id: 't', x: 10.4, y: 3.2, w: 1.8, h: 0.9, tone: 'ink', label: 'factor $t_i$' },
  ],
  edges: [
    { from: 'q:e', to: 'cav:w', label: 'remove site $i$' },
    { from: 'cav:s', to: 'tilt:n', label: 'multiply' },
    { from: 't:w', to: 'tilt:e' },
    { from: 'tilt:w', to: 'proj:e', label: 'match $\\expect[\\tvec]$' },
    { from: 'proj:n', to: 'q:s', label: '$\\tilde t_i \\propto q^{\\text{new}}/q^{\\setminus i}$', labelSide: 'right' },
  ],
}

/** The EP loop for one site: cavity, tilted distribution, projection and site update. */
export function EpLoop() {
  return (
    <Interactive
      title="One EP update"
      caption="Remove site i from the approximation to get the cavity; multiply the cavity by the exact factor tᵢ to get the tilted distribution; project that back onto the family by matching expected sufficient statistics; store the ratio of the projection to the cavity as the new site. Every other site is untouched. EP repeats this for i = 1, …, n until no site changes."
    >
      <Diagram
        spec={SPEC}
        ariaLabel="Cycle: approximation q, remove site i to get the cavity, multiply by factor t_i to get the tilted distribution, project by matching moments, divide by the cavity to get the new site, back to q"
      />
    </Interactive>
  )
}
