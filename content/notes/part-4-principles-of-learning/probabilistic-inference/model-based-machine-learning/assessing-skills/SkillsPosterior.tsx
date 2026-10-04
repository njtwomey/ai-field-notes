import { useMemo } from 'react'
import {
  Curve,
  Diagram,
  factor,
  Figure,
  formatNumber,
  Handle,
  int,
  link,
  Plot,
  Readout,
  setting,
  slider,
  useAxis,
  useFigureState,
  variable,
} from 'aifn-render'
import type { DiagramEdge, DiagramNode, DiagramSpec } from 'aifn-render'
import { exactPosterior, loopyHistory, type SkillsModel } from '../_shared/skills'

const SWEEPS = 8
const X = Array.from({ length: SWEEPS + 1 }, (_, i) => i)
/** Questions 1 and 2 test one skill each; questions 3 and 4 need both. */
const QUESTIONS = [[0], [1], [0, 1], [0, 1]]

/** Toggle a candidate's answers and watch the skill posteriors, exact and by loopy belief propagation. */
export function SkillsPosterior() {
  const state = useFigureState({
    fourth: setting(false, 'ask Q4 (both)'),
    q1: setting(true, 'Q1 (C#) correct'),
    q2: setting(false, 'Q2 (SQL) correct'),
    q3: setting(false, 'Q3 (both) correct'),
    q4: setting(false, { label: 'Q4 (both) correct', when: (v) => v.fourth === true }),
    guess: slider(0.05, 0.5, 0.2, { step: 0.05, label: 'guess probability', format: (v) => v.toFixed(2) }),
    sweep: int(1, { min: 0, max: SWEEPS, step: 1, label: 'BP sweep', format: (v) => String(v) }),
  })

  const nq = state.fourth ? 4 : 3
  const answers = useMemo(() => [state.q1, state.q2, state.q3, state.q4], [state.q1, state.q2, state.q3, state.q4])
  const key = answers.slice(0, nq).join()
  const { exact, history } = useMemo(() => {
    const model: SkillsModel = {
      nSkills: 2,
      questions: QUESTIONS.slice(0, nq),
      prior: 0.5,
      pKnow: 0.9,
      pGuess: state.guess,
    }
    const a = key.split(',').map((v) => v === 'true')
    return { exact: exactPosterior(model, a), history: loopyHistory(model, a, SWEEPS) }
  }, [key, nq, state.guess])

  const series = useMemo(
    () =>
      [
        { name: 'C# (loopy BP)', x: X, y: history.map((h) => h[0]), slot: 0 },
        { name: 'SQL (loopy BP)', x: X, y: history.map((h) => h[1]), slot: 1 },
        { name: 'C# (exact)', x: X, y: X.map(() => exact[0]), slot: 0, dashed: true },
        { name: 'SQL (exact)', x: X, y: X.map(() => exact[1]), slot: 1, dashed: true },
      ] as const,
    [history, exact],
  )

  const now = history[state.sweep]
  const spec = useMemo(() => graph(now, answers, nq), [now, answers, nq])

  const xAxis = useAxis({ label: 'sweeps of belief propagation', range: [0, SWEEPS] })
  const yAxis = useAxis({ label: 'P(has skill)', range: [0, 1] })
  return (
    <Figure
      title="Skill posteriors from four test answers"
      state={state}
      caption="Switch on the questions the candidate answered correctly. Questions 1 and 2 need one skill each; questions 3 and 4 need both. With three questions the factor graph is a tree and one sweep of belief propagation gives the exact posterior. Adding question 4 closes a loop: loopy belief propagation then converges to slightly wrong values (solid lines) next to the exact ones (dashed). Step through sweeps with the arrows or drag the sweep line. Skill nodes are shaded by their current probability."
      readouts={
        <>
          <Readout label="P(C#) loopy / exact" value={`${formatNumber(now[0])} / ${formatNumber(exact[0])}`} />
          <Readout label="P(SQL) loopy / exact" value={`${formatNumber(now[1])} / ${formatNumber(exact[1])}`} />
        </>
      }
    >
      <div className="grid grid-cols-1 items-center gap-4 md:grid-cols-2">
        <Diagram
          spec={spec}
          ariaLabel="Factor graph of the skills model: two skills, an AND factor, noise factors and observed answers"
        />
        <Plot x={xAxis} y={yAxis} height={280}>
          <Curve {...series[0]} />
          <Curve {...series[1]} />
          <Curve {...series[2]} />
          <Curve {...series[3]} />
          <Handle {...state.handle('sweep', { label: 'sweep' })} />
        </Plot>
      </div>
    </Figure>
  )
}

function graph(p: number[], answers: boolean[], nq: number): DiagramSpec {
  const qx = [0, 1.6, 3.2, 4.8]
  const nodes: DiagramNode[] = [
    factor('p0', 0.8, 0, ''),
    factor('p1', 4, 0, ''),
    variable('s0', 0.8, 1.1, '$s_{\\text{C\\#}}$', { shade: p[0] }),
    variable('s1', 4, 1.1, '$s_{\\text{SQL}}$', { shade: p[1] }),
  ]
  const edges: DiagramEdge[] = [link('p0', 's0', false), link('p1', 's1', false)]
  for (let q = 0; q < nq; q++) {
    const both = QUESTIONS[q].length === 2
    if (both) {
      nodes.push(factor(`and${q}`, qx[q], 2.2, 'AND', q === 2 ? 'w' : 'e'))
      nodes.push(variable(`h${q}`, qx[q], 3.2, `$h_${q + 1}$`, { w: 0.8, h: 0.8 }))
      edges.push(link('s0', `and${q}`, false), link('s1', `and${q}`, false), link(`and${q}`, `h${q}`, false))
    }
    nodes.push(factor(`n${q}`, qx[q], 4.2, ''))
    nodes.push(variable(`a${q}`, qx[q], 5.2, `$a_${q + 1}$`, { filled: true }))
    nodes.push({ id: `t${q}`, x: qx[q], y: 6.1, shape: 'text', small: true, label: answers[q] ? 'right' : 'wrong' })
    edges.push(link(both ? `h${q}` : `s${QUESTIONS[q][0]}`, `n${q}`, false), link(`n${q}`, `a${q}`, false))
  }
  return { unit: 40, nodes, edges }
}
