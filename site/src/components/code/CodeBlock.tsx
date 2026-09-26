import { Check, Copy, FileCode2 } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { highlight, resolveLanguage } from './highlight'

export type CodeBlockProps = {
  code: string
  lang?: string
  /** Shown in the header, e.g. a file path. */
  filename?: string
  className?: string
}

export function CodeBlock({ code, lang, filename, className }: CodeBlockProps) {
  const [html, setHtml] = useState<string>()
  const [copied, setCopied] = useState(false)
  const text = code.replace(/\n$/, '')

  useEffect(() => {
    let live = true
    highlight(text, resolveLanguage(lang)).then((h) => live && setHtml(h))
    return () => {
      live = false
    }
  }, [text, lang])

  const copy = async () => {
    await navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  return (
    <div className={cn('not-prose group relative my-6 overflow-hidden rounded-lg border bg-muted/30', className)}>
      {filename && (
        <div className="flex items-center gap-2 border-b px-3 py-1.5 font-mono text-xs text-muted-foreground">
          <FileCode2 className="size-3.5" aria-hidden />
          {filename}
        </div>
      )}
      <Button
        variant="ghost"
        size="icon-sm"
        onClick={copy}
        aria-label="Copy code"
        className="absolute top-1 right-1 opacity-0 group-hover:opacity-100 focus-visible:opacity-100"
      >
        {copied ? <Check /> : <Copy />}
      </Button>
      {html ? (
        <div
          className="overflow-x-auto p-4 text-[13px] leading-relaxed [&_pre]:!bg-transparent"
          dangerouslySetInnerHTML={{ __html: html }}
        />
      ) : (
        <pre className="overflow-x-auto p-4 text-[13px] leading-relaxed">
          <code>{text}</code>
        </pre>
      )}
    </div>
  )
}
