import { AlertTriangle, ArrowLeft, Check, Copy, Home, RotateCcw } from 'lucide-react'
import { useState } from 'react'
import { Link, useRouteError } from 'react-router'
import { Button } from '@/components/ui/button'
import { SiteFooter } from './SiteFooter'
import { SiteHeader } from './SiteHeader'

export function RouteErrorBoundary() {
  const error = useRouteError()
  const [copied, setCopied] = useState(false)

  const errorMessage =
    error instanceof Error
      ? error.message
      : typeof error === 'string'
        ? error
        : 'An unknown application error occurred.'

  const stack = error instanceof Error ? error.stack : undefined

  const copyDetails = () => {
    const text = `${errorMessage}\n\n${stack ?? ''}`
    navigator.clipboard?.writeText(text.trim()).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  return (
    <div className="flex min-h-svh flex-col bg-background text-foreground">
      <SiteHeader />
      <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col items-center justify-center px-4 py-16 text-center">
        <div className="mb-4 inline-flex items-center gap-1.5 rounded-full border border-amber-500/30 bg-amber-500/10 px-3 py-1 text-xs font-mono font-medium text-amber-600 dark:text-amber-400">
          <AlertTriangle className="size-3.5" />
          <span>OB-1 · Caught Exception</span>
        </div>

        <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">Something went off by one</h1>
        <p className="mt-3 max-w-md text-sm text-muted-foreground sm:text-base">
          An unexpected error occurred while rendering this page. You can try reloading, returning to the notes index,
          or inspecting the details below.
        </p>

        <div className="mt-6 flex flex-wrap items-center justify-center gap-2.5">
          <Button onClick={() => window.location.reload()}>
            <RotateCcw className="mr-1.5 size-4" />
            Reload Page
          </Button>
          <Button variant="outline" onClick={() => window.history.back()}>
            <ArrowLeft className="mr-1.5 size-4" />
            Go Back
          </Button>
          <Link
            to="/browse"
            className="inline-flex h-8 items-center justify-center rounded-lg border border-border bg-background px-2.5 text-sm font-medium hover:bg-muted"
          >
            Browse Notes
          </Link>
          <Link
            to="/"
            className="inline-flex h-8 items-center justify-center rounded-lg px-2.5 text-sm font-medium hover:bg-muted text-muted-foreground hover:text-foreground"
          >
            <Home className="mr-1.5 size-4" />
            Home
          </Link>
        </div>

        <div className="mt-10 w-full rounded-xl border border-destructive/20 bg-destructive/5 p-4 text-left">
          <div className="flex items-center justify-between gap-2">
            <span className="font-mono text-xs font-semibold text-destructive">
              {error instanceof Error ? error.name : 'Error'}: {errorMessage}
            </span>
            <Button variant="ghost" size="xs" onClick={copyDetails} className="h-7 text-xs text-muted-foreground">
              {copied ? (
                <>
                  <Check className="mr-1 size-3.5 text-primary" /> Copied
                </>
              ) : (
                <>
                  <Copy className="mr-1 size-3.5" /> Copy Details
                </>
              )}
            </Button>
          </div>

          {stack && (
            <details className="mt-3 text-xs">
              <summary className="cursor-pointer select-none text-muted-foreground hover:text-foreground">
                Stack trace
              </summary>
              <pre className="mt-2 max-h-60 overflow-x-auto rounded-lg bg-background/80 p-3 font-mono text-[11px] leading-relaxed text-muted-foreground border">
                {stack}
              </pre>
            </details>
          )}
        </div>
      </main>
      <SiteFooter />
    </div>
  )
}
