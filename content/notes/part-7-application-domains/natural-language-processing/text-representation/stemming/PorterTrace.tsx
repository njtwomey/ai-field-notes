import { useMemo, useState } from 'react'
import { Interactive, ParamChoice, Readout } from 'aifn-render'
import { Input } from 'aifn-render'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from 'aifn-render'
import { cvForm, measure, porterStem, type PorterStep } from '../_shared/text'

const PRESETS = ['generalizations', 'oscillators', 'filing', 'hopping', 'argument', 'relational', 'university'] as const

const GROUPS = {
  connect: ['connect', 'connected', 'connecting', 'connection', 'connections'],
  over: ['universe', 'university', 'universal', 'general', 'generous', 'generate'],
  under: ['absorb', 'absorption', 'alumnus', 'alumni', 'europe', 'european'],
  irregular: ['run', 'ran', 'running', 'mouse', 'mice', 'be', 'was'],
} as const
type Group = keyof typeof GROUPS

/** Collapse runs so the form reads [C](VC)^m[V]. */
const collapsed = (w: string) => cvForm(w).replace(/C+/g, 'C').replace(/V+/g, 'V')

/** Porter's algorithm on one word, rule by rule, plus the stems of a group of related words. */
export function PorterTrace() {
  const [word, setWord] = useState('generalizations')
  const [group, setGroup] = useState<Group>('connect')

  const { stem, trace } = useMemo(() => {
    const t: PorterStep[] = []
    const s = porterStem(word.trim(), t)
    return { stem: s, trace: t }
  }, [word])

  const w = word.trim().toLowerCase()
  return (
    <Interactive
      title="Porter's algorithm, one rule at a time"
      caption="Type a word or pick one. Each row is a rule that fired, with the measure m of the stem it left. A step whose longest matching suffix fails its condition does nothing, so 'argument' keeps its 'ment'. The lower table shows the stems of a group of related words: which ones the stemmer joins, and which it wrongly joins or leaves apart."
      controls={
        <>
          <label className="flex flex-col gap-2 text-xs text-muted-foreground">
            word
            <Input
              value={word}
              onChange={(e) => setWord(e.target.value)}
              className="font-mono text-sm text-foreground"
              spellCheck={false}
            />
          </label>
          <ParamChoice
            label="example"
            value={(PRESETS as readonly string[]).includes(w) ? w : ''}
            onChange={setWord}
            options={PRESETS.map((p) => ({ value: p, label: p }))}
          />
          <ParamChoice
            label="word group"
            value={group}
            onChange={setGroup}
            options={[
              { value: 'connect', label: 'one family' },
              { value: 'over', label: 'over-stemming' },
              { value: 'under', label: 'under-stemming' },
              { value: 'irregular', label: 'irregular' },
            ]}
          />
        </>
      }
      readout={
        <>
          <Readout label="C/V form" value={w ? collapsed(w) : '–'} />
          <Readout label="measure m of the word" value={w ? measure(w) : '–'} />
          <Readout label="stem" value={stem || '–'} />
        </>
      }
    >
      <div className="overflow-x-auto rounded-lg border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>step</TableHead>
              <TableHead>rule</TableHead>
              <TableHead>before</TableHead>
              <TableHead>after</TableHead>
              <TableHead>m(after)</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {trace.length === 0 ? (
              <TableRow>
                <TableCell colSpan={5} className="text-xs text-muted-foreground">
                  No rule fired: the word is its own stem.
                </TableCell>
              </TableRow>
            ) : (
              trace.map((t, i) => (
                <TableRow key={i}>
                  <TableCell className="font-mono text-xs">{t.step}</TableCell>
                  <TableCell className="font-mono text-xs">{t.rule}</TableCell>
                  <TableCell className="font-mono text-xs">{t.before}</TableCell>
                  <TableCell className="font-mono text-xs">{t.after}</TableCell>
                  <TableCell className="font-mono text-xs tabular-nums">{measure(t.after)}</TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {GROUPS[group].map((g) => (
          <span key={g} className="inline-flex flex-col items-center rounded-md border px-2 py-1 font-mono text-xs">
            <span>{g}</span>
            <span className="text-muted-foreground">{porterStem(g)}</span>
          </span>
        ))}
      </div>
    </Interactive>
  )
}
