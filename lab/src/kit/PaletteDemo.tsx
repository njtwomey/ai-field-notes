import { chrome, diverging, palette, sequential, type Mode } from 'aifn-render/design'
import { useTheme } from 'aifn-render/design'
import { ThemeToggle } from 'aifn-render/controls'

function Swatches({ colours, labels }: { colours: readonly string[]; labels?: readonly string[] }) {
  return (
    <div className="flex flex-wrap gap-1">
      {colours.map((c, i) => (
        <div key={i} className="flex w-12 flex-col items-center gap-1">
          <div className="h-7 w-full rounded-md ring-1 ring-foreground/10" style={{ background: c }} />
          <span className="font-mono text-[10px] text-muted-foreground">{labels?.[i] ?? i}</span>
        </div>
      ))}
    </div>
  )
}

function Ramp({ colours }: { colours: readonly string[] }) {
  return (
    <div
      className="h-6 w-full max-w-xl rounded-md ring-1 ring-foreground/10"
      style={{ background: `linear-gradient(to right, ${colours.join(', ')})` }}
    />
  )
}

/** The data palette in the current theme, and the other theme's categorical slots for comparison. */
export function PaletteDemo() {
  const { resolved, preference } = useTheme()
  const other: Mode = resolved === 'dark' ? 'light' : 'dark'
  const c = chrome(resolved)
  return (
    <div className="grid gap-6 rounded-xl border bg-card p-4 lg:grid-cols-2">
      <div className="flex flex-col gap-4">
        <div className="flex items-center gap-2 text-sm">
          Theme: <span className="font-mono">{preference}</span> → <span className="font-mono">{resolved}</span>
          <ThemeToggle />
        </div>
        <div className="space-y-1.5">
          <div className="text-xs text-muted-foreground">
            Categorical slots ({resolved}), fixed per entity, never cycled
          </div>
          <Swatches colours={palette.categorical[resolved]} />
        </div>
        <div className="space-y-1.5">
          <div className="text-xs text-muted-foreground">Categorical slots ({other})</div>
          <Swatches colours={palette.categorical[other]} />
        </div>
      </div>
      <div className="flex flex-col gap-4">
        <div className="space-y-1.5">
          <div className="text-xs text-muted-foreground">Sequential: one hue, light to dark</div>
          <Ramp colours={sequential(resolved)} />
        </div>
        <div className="space-y-1.5">
          <div className="text-xs text-muted-foreground">Diverging: signed values, neutral midpoint ({resolved})</div>
          <Ramp colours={diverging(resolved)} />
        </div>
        <div className="space-y-1.5">
          <div className="text-xs text-muted-foreground">Chart chrome ({resolved}); emphasis marks use ink</div>
          <Swatches colours={Object.values(c)} labels={Object.keys(c)} />
        </div>
      </div>
    </div>
  )
}
