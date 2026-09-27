import { Diagram } from '@/components/diagram/Diagram'
import { factor, link, variable } from '@/components/diagram/components'
import type { DiagramSpec } from '@/components/diagram/types'
import { Interactive } from '@/components/viz'

const line = (a: string, b: string, label?: string) =>
  link(a, b, false, label ? { label, labelSide: 'left', labelRotate: false } : {})

const SPEC: DiagramSpec = {
  unit: 42,
  nodes: [
    variable('m', 2.6, -1.4, '$m$'),
    factor('ps', 2.6, 0, '$\\Gauss(m, 1/\\tau)$', 'e'),
    variable('s', 2.6, 1.4, '$s_i$'),
    factor('nj', 0.8, 2.8, '$\\Gauss(s_i, 1/\\tau_J)$', 'w'),
    variable('sj', 0.8, 4.2, '$s^J_i$'),
    factor('bt', 0.8, 5.6, '$\\theta_{\\ell} < s^J_i < \\theta_{\\ell + 1}$', 'w'),
    variable('th', 2.6, 5.6, '$\\thetavec$'),
    variable('l', 0.8, 7, '$\\ell_i$', { filled: true }),
    factor('nc', 4.4, 2.8, '$\\Gauss(s_i, 1/\\tau_C)$', 'e'),
    variable('sc', 4.4, 4.2, '$s^C_i$'),
    factor('ev', 4.4, 5.6, '$\\Gauss(\\hat c_i, v_i)$', 'e'),
  ],
  edges: [
    line('m', 'ps'),
    line('ps', 's'),
    line('s', 'nj'),
    line('nj', 'sj'),
    line('sj', 'bt'),
    line('th', 'bt'),
    line('bt', 'l'),
    line('s', 'nc'),
    line('nc', 'sc'),
    line('ev', 'sc', 'click evidence'),
  ],
  groups: [
    {
      id: 'I',
      label: 'query–document pairs $i$',
      tone: 'ink',
      around: ['s', 'nj', 'sj', 'bt', 'l', 'nc', 'sc', 'ev'],
      pad: 0.6,
      labelAt: 'bottom-right',
    },
  ],
}

/** The click model: one latent score seen through a judge's label and through a click record. */
export function ClickGraph() {
  return (
    <Interactive
      title="Factor graph of the click model"
      caption="Each query–document pair has a latent score sᵢ. A judge sees it with noise and reports the label whose thresholds bracket it (observed, shaded). Users see it with other noise; their click record enters as a fixed Gaussian factor, the moment-matched Beta posterior of the click rate. The score mean, noise precisions and thresholds are shared across pairs and learned."
    >
      <Diagram
        spec={SPEC}
        ariaLabel="Factor graph: latent score with a judged branch ending in an observed label and a click branch ending in a soft-evidence factor"
      />
    </Interactive>
  )
}
