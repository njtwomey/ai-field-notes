import { CircleCheck, CircleX, Terminal } from 'lucide-react'
import { NavLink } from 'react-router'
import { CodeBlock } from '@/components/code/CodeBlock'
import { OutputView } from '@/components/code/OutputView'
import { Badge } from '@/components/ui/badge'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Separator } from '@/components/ui/separator'
import type { ExampleIndex, RunIndex } from '@/generated/contracts'
import { noteUrl } from '@/lib/content'
import { useRun } from '@/lib/generated'
import { cn } from '@/lib/utils'

export function RunIndexNav({ slug, example, active }: { slug: string; example: ExampleIndex; active: string }) {
  return (
    <nav className="text-sm">
      <p className="mb-2 text-xs font-medium tracking-wide text-muted-foreground uppercase">Runs</p>
      <ul className="space-y-0.5">
        {example.runs.map((run) => {
          const Icon = run.exit_code === 0 ? CircleCheck : CircleX
          return (
            <li key={run.name}>
              <NavLink
                to={noteUrl(slug, 'outputs', run.name)}
                className={cn(
                  'flex items-start gap-2 rounded-md px-2 py-1.5 hover:bg-muted',
                  run.name === active ? 'bg-muted text-foreground' : 'text-muted-foreground',
                )}
              >
                <Icon
                  className={cn('mt-0.5 size-3.5 shrink-0', run.exit_code !== 0 && 'text-destructive')}
                  aria-label={run.exit_code === 0 ? 'succeeded' : 'failed'}
                />
                <span>{run.title}</span>
              </NavLink>
            </li>
          )
        })}
      </ul>
    </nav>
  )
}

export function OutputsTab({ run }: { run: RunIndex }) {
  const { data, error, loading } = useRun(run.path)
  const runDir = run.path.replace(/\/run\.json$/, '')
  if (loading) return <p className="text-sm text-muted-foreground">Loading run…</p>
  if (error || !data) return <p className="text-sm text-destructive">{error?.message ?? 'Run not found.'}</p>

  return (
    <div className="space-y-8">
      <div className="space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          <h2 className="text-lg font-semibold tracking-tight">{data.title}</h2>
          <Badge variant={data.exit_code === 0 ? 'secondary' : 'destructive'}>exit {data.exit_code}</Badge>
          <span className="text-xs text-muted-foreground">
            {data.duration_s}s · {data.created.slice(0, 10)} · {data.hash}
          </span>
        </div>
        {data.description && <p className="text-sm text-muted-foreground">{data.description}</p>}
        <CodeBlock code={data.command} lang="bash" className="my-0" />
      </div>
      <Separator />
      {data.outputs.map((o, i) => (
        <OutputView key={i} output={o} runDir={runDir} />
      ))}
      {(data.stdout || data.stderr) && (
        <Collapsible className="rounded-lg border">
          <CollapsibleTrigger className="flex w-full items-center gap-2 px-4 py-2.5 text-sm font-medium">
            <Terminal className="size-4" aria-hidden /> Console output
          </CollapsibleTrigger>
          <CollapsibleContent className="border-t">
            <pre className="max-h-96 overflow-auto p-4 font-mono text-xs">
              {data.stdout}
              {data.stderr && <span className="text-destructive">{data.stderr}</span>}
            </pre>
          </CollapsibleContent>
        </Collapsible>
      )}
    </div>
  )
}
