import { Link } from 'react-router'
import { browseUrl } from '@/lib/content'

const links = [
  { to: browseUrl(), label: 'Browse' },
  { to: '/tags', label: 'Tags' },
  { to: '/references', label: 'References' },
  { to: '/notation', label: 'Notation' },
]

/** Site-wide footer: who wrote what, where to go next, and the source. */
export function SiteFooter() {
  return (
    <footer className="border-t text-sm text-muted-foreground">
      <div className="flex flex-col gap-4 px-4 py-8 lg:flex-row lg:items-start lg:justify-between lg:px-8">
        <div className="max-w-xl space-y-1">
          <p>
            <span className="font-medium text-foreground">AI Field Notes</span> by{' '}
            <a href="https://www.nialltwomey.com/" className="text-foreground hover:underline">
              Niall Twomey
            </a>
          </p>
          <p>Digitised notes from my research, augmented by Claude for this public format.</p>
        </div>
        <nav aria-label="Footer" className="flex flex-wrap gap-x-5 gap-y-2">
          {links.map((l) => (
            <Link key={l.label} to={l.to} className="hover:text-foreground hover:underline">
              {l.label}
            </Link>
          ))}
          <a href="https://github.com/njtwomey/ai-field-notes" className="hover:text-foreground hover:underline">
            Source
          </a>
        </nav>
      </div>
    </footer>
  )
}
