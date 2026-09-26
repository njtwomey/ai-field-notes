import { Link } from 'react-router'
import { PageContainer } from '@/components/layout/AppShell'

export function NotFoundPage() {
  return (
    <PageContainer>
      <h1 className="text-2xl font-semibold">Not found</h1>
      <p className="mt-2 text-muted-foreground">
        No page here. Try search (⌘K) or go{' '}
        <Link to="/" className="underline">
          home
        </Link>
        .
      </p>
    </PageContainer>
  )
}
