import { FileCode2, Play } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Link } from 'react-router'
import { CodeBlock } from '@/components/code/CodeBlock'
import type { ExampleIndex } from '@/generated/contracts'
import { noteUrl } from '@/lib/content'
import { cn } from '@/lib/utils'

const sources = import.meta.glob<string>('@python/mlc/examples/**/*.py', { query: '?raw', import: 'default' })

function loadSource(path: string): Promise<string> {
  const key = Object.keys(sources).find((k) => k.endsWith(path))
  return key ? sources[key]() : Promise.resolve(`# ${path} not found in the bundle`)
}

/** Short file label: path inside the example package. */
export function fileLabel(example: ExampleIndex, path: string): string {
  const dir = `python/${example.module.replaceAll('.', '/')}/`
  return path.startsWith(dir) ? path.slice(dir.length) : path
}

export function CodeIndex({
  slug,
  example,
  file,
  onSelect,
}: {
  slug: string
  example: ExampleIndex
  file: string
  onSelect: (path: string) => void
}) {
  return (
    <nav className="space-y-6 text-sm">
      <div>
        <p className="mb-2 text-xs font-medium tracking-wide text-muted-foreground uppercase">Files</p>
        <ul className="space-y-0.5">
          {example.sources.map((path) => (
            <li key={path}>
              <button
                onClick={() => onSelect(path)}
                className={cn(
                  'flex w-full items-center gap-2 rounded-md px-2 py-1 text-left font-mono text-xs hover:bg-muted',
                  path === file ? 'bg-muted text-foreground' : 'text-muted-foreground',
                )}
              >
                <FileCode2 className="size-3.5 shrink-0" aria-hidden />
                {fileLabel(example, path)}
              </button>
            </li>
          ))}
        </ul>
      </div>
      <div>
        <p className="mb-2 text-xs font-medium tracking-wide text-muted-foreground uppercase">Runs</p>
        <ul className="space-y-0.5">
          {example.runs.map((run) => (
            <li key={run.name}>
              <Link
                to={noteUrl(slug, 'outputs', run.name)}
                className="flex items-center gap-2 rounded-md px-2 py-1 text-muted-foreground hover:bg-muted hover:text-foreground"
              >
                <Play className="size-3.5 shrink-0" aria-hidden />
                {run.title}
              </Link>
            </li>
          ))}
        </ul>
      </div>
    </nav>
  )
}

export function CodeTab({ example, file }: { example: ExampleIndex; file: string }) {
  const [code, setCode] = useState<string>()
  useEffect(() => {
    let live = true
    loadSource(file).then((c) => live && setCode(c))
    return () => {
      live = false
    }
  }, [file])

  return (
    <div className="space-y-6">
      <div className="space-y-1">
        <h2 className="text-lg font-semibold tracking-tight">{example.title}</h2>
        <p className="text-sm text-muted-foreground">
          Runnable package <code className="font-mono text-xs">{example.module}</code>. Every command below is executed
          by the build; its outputs are in the Outputs tab.
        </p>
      </div>
      <CodeBlock
        lang="bash"
        filename="Commands"
        code={example.runs.map((r) => `# ${r.title}\n${r.command}`).join('\n\n')}
      />
      {code !== undefined && <CodeBlock code={code} lang="python" filename={fileLabel(example, file)} />}
    </div>
  )
}
