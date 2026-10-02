import { Diagram } from 'aifn-render'
import type { DiagramSpec } from 'aifn-render'
import { Interactive } from 'aifn-render'

type N = DiagramSpec['nodes'][number]
const v = (id: string, x: number, y: number, label: string, extra: Partial<N> = {}): N => ({
  id,
  x,
  y,
  shape: 'circle',
  label,
  tone: 'ink',
  ...extra,
})
const note = (id: string, x: number, label: string): N => ({ id, x, y: 4.3, w: 3.6, shape: 'text', small: true, label })

const spec: DiagramSpec = {
  nodes: [
    v('a_x', 1, 0.4, '$\\Xmat$', { tone: 2 }),
    v('a_y', 0, 2.6, '$\\Ymat$', { filled: true }),
    v('a_z', 2, 2.6, '$\\Zmat$', { filled: true }),
    note('a_t', 1, 'shared GP-LVM:\none latent space'),

    v('b_xy', 5, 0.4, '$\\Xmat^Y$', { tone: 0 }),
    v('b_xs', 6.5, 0.4, '$\\Xmat^s$', { tone: 2 }),
    v('b_xz', 8, 0.4, '$\\Xmat^Z$', { tone: 1 }),
    v('b_y', 5.75, 2.6, '$\\Ymat$', { filled: true }),
    v('b_z', 7.25, 2.6, '$\\Zmat$', { filled: true }),
    note('b_t', 6.5, 'shared and private\nlatent spaces'),

    v('c_x', 12, 0.4, '$\\Xmat$', { shape: 'latent', tone: 2 }),
    v('c_y', 11, 2.6, '$\\Ymat$', { filled: true }),
    v('c_z', 13, 2.6, '$\\Zmat$', { filled: true }),
    v('c_wy', 9.9, 1.2, '$\\wvec^Y$', { w: 0.75, h: 0.75, small: true }),
    v('c_wz', 14.1, 1.2, '$\\wvec^Z$', { w: 0.75, h: 0.75, small: true }),
    note('c_t', 12, 'MRD: one integrated-out space,\nper-view relevance weights'),
  ],
  edges: [
    { from: 'a_x', to: 'a_y', route: 'straight' },
    { from: 'a_x', to: 'a_z', route: 'straight' },
    { from: 'b_xy', to: 'b_y', route: 'straight' },
    { from: 'b_xs', to: 'b_y', route: 'straight' },
    { from: 'b_xs', to: 'b_z', route: 'straight' },
    { from: 'b_xz', to: 'b_z', route: 'straight' },
    { from: 'c_x', to: 'c_y', route: 'straight' },
    { from: 'c_x', to: 'c_z', route: 'straight' },
    { from: 'c_wy', to: 'c_y', route: 'straight' },
    { from: 'c_wz', to: 'c_z', route: 'straight' },
  ],
}

/** Three ways to relate two views through GP-LVM latent spaces. */
export function SharedModels() {
  return (
    <Interactive
      title="From a shared latent space to manifold relevance determination"
      caption="Shaded nodes are the two observed views. Left: one latent space generates both views. Centre: the latent space is split by hand into a shared part and a part private to each view. Right: one latent space, integrated out, with a separate vector of ARD weights for each view; which dimensions are shared and which are private is learned from the weights."
    >
      <Diagram
        spec={spec}
        ariaLabel="Shared GP-LVM, factorised shared and private GP-LVM, and manifold relevance determination"
      />
    </Interactive>
  )
}
