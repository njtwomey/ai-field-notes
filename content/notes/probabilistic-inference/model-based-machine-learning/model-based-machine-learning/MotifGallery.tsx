import { MathText } from '@/components/content/MathText'
import { Diagram } from '@/components/diagram/Diagram'
import { factor, link, variable } from '@/components/diagram/components'
import type { DiagramSpec } from '@/components/diagram/types'
import { Interactive } from '@/components/viz'

const line = (a: string, b: string) => link(a, b, false)
const dot = (id: string, x: number, y: number) => ({ id, x, y, shape: 'dot' as const, w: 0.18, h: 0.18 })

const MOTIFS: { name: string; caption: string; spec: DiagramSpec }[] = [
  {
    name: 'Gate',
    caption:
      'A selector $c$ switches which factor explains $x$. Used for model selection, knowing versus guessing, and latent classes.',
    spec: {
      nodes: [
        variable('c', 1.6, 0, '$c$'),
        dot('d1', 0.3, 1.3),
        dot('d2', 2.9, 1.3),
        factor('f1', 0.9, 2.0, '$f_1$', 'e'),
        factor('f2', 2.3, 2.0, '$f_2$', 'w'),
        variable('x', 1.6, 3.6, '$x$', { filled: true }),
      ],
      edges: [line('c', 'd1'), line('c', 'd2'), line('f1', 'x'), line('f2', 'x')],
      groups: [
        { id: 'g1', label: '$c = 1$', tone: 0, dashed: true, around: ['f1', 'd1'], pad: 0.35, labelAt: 'top-left' },
        { id: 'g2', label: '$c = 2$', tone: 1, dashed: true, around: ['f2', 'd2'], pad: 0.35, labelAt: 'top-right' },
      ],
    },
  },
  {
    name: 'Mixture',
    caption:
      'Every item $n$ has its own selector $z_n$; the components $\\thetavec_k$ sit in a plate of their own. Asthma classes, communities of workers.',
    spec: {
      nodes: [
        factor('pz', 0, 0, '', 'n'),
        variable('z', 0, 1.3, '$z_n$'),
        variable('x', 0, 2.8, '$x_n$', { filled: true }),
        factor('f', 1.5, 2.8, '', 'n'),
        variable('th', 3.1, 2.8, '$\\thetavec_k$'),
      ],
      edges: [line('pz', 'z'), line('z', 'f'), line('f', 'x'), line('th', 'f')],
      groups: [
        { id: 'N', label: '$n$', tone: 'ink', around: ['z', 'x', 'f'], pad: 0.35, labelAt: 'bottom-right' },
        { id: 'K', label: '$k$', tone: 'ink', around: ['th'], pad: 0.3, labelAt: 'bottom-right' },
      ],
    },
  },
  {
    name: 'Shared prior across a plate',
    caption:
      'Per-user weights $w_u$ drawn around a population mean $m$: a new user borrows strength from everyone else. Community priors, reviewer biases.',
    spec: {
      nodes: [
        factor('pm', 1.5, 0, '', 'n'),
        variable('m', 1.5, 1.2, '$m$'),
        factor('fw', 1.5, 2.4, '', 'e'),
        variable('w', 1.5, 3.6, '$w_u$'),
        factor('fy', 1.5, 4.8, '', 'e'),
        variable('y', 1.5, 6, '$y_{ui}$', { filled: true }),
      ],
      edges: [line('pm', 'm'), line('m', 'fw'), line('fw', 'w'), line('w', 'fy'), line('fy', 'y')],
      groups: [
        { id: 'I', label: '$i$', tone: 'ink', around: ['fy', 'y'], pad: 0.3, labelAt: 'bottom-right' },
        {
          id: 'U',
          label: 'users $u$',
          tone: 'ink',
          around: ['fw', 'w', 'fy', 'y'],
          pad: 0.65,
          labelAt: 'bottom-right',
        },
      ],
    },
  },
  {
    name: 'Chain over time',
    caption:
      'Each state depends on the previous one and emits an observation: sensitisation by age, weights that drift, skills over seasons.',
    spec: {
      nodes: [0, 1, 2].flatMap((i) => [
        variable(`s${i}`, i * 2, 0, `$s_${i + 1}$`),
        factor(`e${i}`, i * 2, 1.2, ''),
        variable(`o${i}`, i * 2, 2.4, `$o_${i + 1}$`, { filled: true }),
        ...(i > 0 ? [factor(`t${i}`, i * 2 - 1, 0, '')] : []),
      ]),
      edges: [0, 1, 2].flatMap((i) => [
        line(`s${i}`, `e${i}`),
        line(`e${i}`, `o${i}`),
        ...(i > 0 ? [line(`s${i - 1}`, `t${i}`), line(`t${i}`, `s${i}`)] : []),
      ]),
    },
  },
  {
    name: 'Noisy threshold',
    caption:
      'A Gaussian score plus noise, compared with a threshold, gives a binary or ordinal observation: the probit. Email replies, clicks, star ratings, review scores.',
    spec: {
      nodes: [
        variable('s', 0, 0, '$s$'),
        factor('n', 1.4, 0, '', 'n'),
        variable('t', 2.8, 0, '$\\tilde s$'),
        factor('g', 4.2, 0, '$>$', 'n'),
        variable('th', 4.2, 1.4, '$\\theta$'),
        variable('y', 5.6, 0, '$y$', { filled: true }),
      ],
      edges: [line('s', 'n'), line('n', 't'), line('t', 'g'), line('th', 'g'), line('g', 'y')],
    },
  },
  {
    name: 'Product of latents',
    caption:
      'Two latent vectors meet in an inner product: user and item traits, ability and discrimination. Messages need a variational treatment.',
    spec: {
      nodes: [
        variable('s', 0, 0, '$s_k$'),
        variable('t', 0, 2, '$t_k$'),
        factor('m', 1.4, 1, '$\\times$', 'n'),
        variable('z', 2.8, 1, '$z_k$'),
        factor('sum', 4.2, 1, '$\\Sigma$', 'n'),
        variable('r', 5.6, 1, '$r$', { filled: true }),
      ],
      edges: [line('s', 'm'), line('t', 'm'), line('m', 'z'), line('z', 'sum'), line('sum', 'r')],
      groups: [
        { id: 'K', label: '$k$', tone: 'ink', around: ['s', 't', 'm', 'z'], pad: 0.35, labelAt: 'bottom-right' },
      ],
    },
  },
]

/** The recurring factor-graph motifs of the case studies, side by side. */
export function MotifGallery() {
  return (
    <Interactive
      title="Six motifs that recur across the case studies"
      caption="Circles are variables, squares factors, shaded circles observed. Dashed boxes are gates, labelled with the value of the selector that turns them on; solid boxes are plates."
    >
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {MOTIFS.map((m) => (
          <div key={m.name} className="flex flex-col gap-1">
            <p className="text-xs font-medium">{m.name}</p>
            <Diagram spec={{ ...m.spec, unit: 36 }} ariaLabel={`Factor-graph motif: ${m.name}`} />
            <p className="text-xs text-muted-foreground">
              <MathText text={m.caption} />
            </p>
          </div>
        ))}
      </div>
    </Interactive>
  )
}
