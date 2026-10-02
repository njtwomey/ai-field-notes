import { useMemo, useState } from 'react'
import { Diagram } from 'aifn-render'
import { factor, link, variable } from 'aifn-render'
import type { DiagramEdge, DiagramNode, DiagramSpec } from 'aifn-render'
import {
  Interactive,
  ParamSlider,
  ParamSwitch,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from 'aifn-render'
import { exactPosterior, loopyHistory, type SkillsModel } from '../_shared/skills'

const SWEEPS = 8
const X = Array.from({ length: SWEEPS + 1 }, (_, i) => i)
/** Questions 1 and 2 test one skill each; questions 3 and 4 need both. */
const QUESTIONS = [[0], [1], [0, 1], [0, 1]]

/** Toggle a candidate's answers and watch the skill posteriors, exact and by loopy belief propagation. */
export function SkillsPosterior() {
  const [answers, setAnswers] = useState([true, false, false, false])
  const [fourth, setFourth] = useState(false)
  const guess = useParam(0.2, { min: 0.05, max: 0.5, step: 0.05 })
  const sweep = useParam(1, { min: 0, max: SWEEPS, step: 1 })

  const nq = fourth ? 4 : 3
  const key = answers.slice(0, nq).join()
  const { exact, history } = useMemo(() => {
    const model: SkillsModel = {
      nSkills: 2,
      questions: QUESTIONS.slice(0, nq),
      prior: 0.5,
      pKnow: 0.9,
      pGuess: guess.value,
    }
    const a = key.split(',').map((v) => v === 'true')
    return { exact: exactPosterior(model, a), history: loopyHistory(model, a, SWEEPS) }
  }, [key, nq, guess.value])

  const series = useMemo(
    (): XYSeries[] => [
      { name: 'C# (loopy BP)', type: 'line', x: X, y: history.map((h) => h[0]), slot: 0 },
      { name: 'SQL (loopy BP)', type: 'line', x: X, y: history.map((h) => h[1]), slot: 1 },
      { name: 'C# (exact)', type: 'line', x: X, y: X.map(() => exact[0]), slot: 0, dashed: true },
      { name: 'SQL (exact)', type: 'line', x: X, y: X.map(() => exact[1]), slot: 1, dashed: true },
    ],
    [history, exact],
  )

  const now = history[sweep.value]
  const spec = useMemo(() => graph(now, answers, nq), [now, answers, nq])
  const toggle = (i: number) => (v: boolean) => setAnswers((a) => a.map((x, j) => (j === i ? v : x)))

  return (
    <Interactive
      title="Skill posteriors from four test answers"
      caption="Tick the questions the candidate answered correctly. Questions 1 and 2 need one skill each; questions 3 and 4 need both. With three questions the factor graph is a tree and one sweep of belief propagation gives the exact posterior. Adding question 4 closes a loop: loopy belief propagation then converges to slightly wrong values (solid lines) next to the exact ones (dashed). Step through sweeps with the arrows or drag the sweep line. Skill nodes are shaded by their current probability."
      controls={
        <>
          <ParamSwitch label="Q1 (C#) correct" checked={answers[0]} onChange={toggle(0)} />
          <ParamSwitch label="Q2 (SQL) correct" checked={answers[1]} onChange={toggle(1)} />
          <ParamSwitch label="Q3 (both) correct" checked={answers[2]} onChange={toggle(2)} />
          <ParamSwitch label="ask Q4 (both)" checked={fourth} onChange={setFourth} />
          {fourth && <ParamSwitch label="Q4 (both) correct" checked={answers[3]} onChange={toggle(3)} />}
          <ParamSlider label="guess probability" param={guess} format={(v) => v.toFixed(2)} />
          <ParamSlider label="BP sweep" param={sweep} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
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
        <XYChart
          series={series}
          xLabel="sweeps of belief propagation"
          yLabel="P(has skill)"
          xRange={[0, SWEEPS]}
          yRange={[0, 1]}
          height={280}
          handles={[{ kind: 'x', at: sweep.value, label: 'sweep', onDrag: (x) => sweep.set(x) }]}
        />
      </div>
    </Interactive>
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
