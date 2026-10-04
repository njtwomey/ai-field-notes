import { Diagram, factor, Figure, link, variable } from 'aifn-render'
import type { DiagramNode, DiagramSpec } from 'aifn-render'

const PLAYERS = [0, 1, 2, 3]
const px = (i: number) => i * 1.6
const TEAM_X = [0.8, 4]
const MID = 2.4
const NOTE_X = 8.6

const note = (id: string, y: number, label: string): DiagramNode => ({
  id,
  x: NOTE_X,
  y,
  shape: 'text',
  small: true,
  w: 5.6,
  label,
})

const edge = (from: string, to: string) => link(from, to, false)

const SPEC: DiagramSpec = {
  unit: 40,
  nodes: [
    ...PLAYERS.flatMap((i) => [
      factor(`prior${i}`, px(i), 0, ''),
      variable(`s${i}`, px(i), 1.2, `$s_${i + 1}$`),
      factor(`perf${i}`, px(i), 2.4, ''),
      variable(`p${i}`, px(i), 3.6, `$p_${i + 1}$`),
    ]),
    factor('sum0', TEAM_X[0], 4.8, ''),
    factor('sum1', TEAM_X[1], 4.8, ''),
    variable('t0', TEAM_X[0], 6, '$t_1$'),
    variable('t1', TEAM_X[1], 6, '$t_2$'),
    factor('diff', MID, 7.2, ''),
    variable('d', MID, 8.4, '$d$'),
    factor('out', MID, 9.6, ''),
    note('n0', 0, 'prior and dynamics: $\\Gauss(s_i;\\, \\mu_i,\\, \\sigma_i^2 + \\tau^2)$'),
    note('n1', 1.2, 'skills'),
    note('n2', 2.4, 'performance: $\\Gauss(p_i;\\, s_i,\\, \\beta^2)$'),
    note('n3', 3.6, 'player performances'),
    note('n4', 4.8, 'team sum: $\\indicator[t_j = \\sum_{i \\in A_j} p_i]$'),
    note('n5', 6, 'team performances'),
    note('n6', 7.2, 'difference: $\\indicator[d = t_1 - t_2]$'),
    note('n7', 8.4, 'performance difference'),
    note('n8', 9.6, 'outcome: $\\indicator[d > \\varepsilon]$'),
  ],
  edges: [
    ...PLAYERS.flatMap((i) => [
      edge(`prior${i}`, `s${i}`),
      edge(`s${i}`, `perf${i}`),
      edge(`perf${i}`, `p${i}`),
      edge(`p${i}`, `sum${i < 2 ? 0 : 1}`),
    ]),
    edge('sum0', 't0'),
    edge('sum1', 't1'),
    edge('t0', 'diff'),
    edge('t1', 'diff'),
    edge('diff', 'd'),
    edge('d', 'out'),
  ],
  groups: [
    {
      id: 'team1',
      label: 'team 1',
      tone: 0,
      dashed: true,
      around: ['p0', 'p1', 'sum0'],
      pad: 0.35,
      labelAt: 'bottom-left',
    },
    {
      id: 'team2',
      label: 'team 2',
      tone: 1,
      dashed: true,
      around: ['p2', 'p3', 'sum1'],
      pad: 0.35,
      labelAt: 'bottom-right',
    },
  ],
}

/** The TrueSkill factor graph of one game between two teams of two players, in which team 1 wins. */
export function TrueSkillFactorGraph() {
  return (
    <Figure
      title="The TrueSkill factor graph of one team game"
      caption="Circles are variables and squares are factors. Each row of factors adds one assumption: a Gaussian belief about each skill, a noisy performance around each skill, team performance as the sum of its players' performances, the difference between the teams, and the observed outcome as a constraint on the sign of that difference. Every factor except the last is Gaussian or linear."
    >
      <Diagram
        spec={SPEC}
        ariaLabel="TrueSkill factor graph: skills, performances, team sums, difference and outcome"
      />
    </Figure>
  )
}
