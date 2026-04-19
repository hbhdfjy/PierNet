let installed = false

function resolveElement(target: EventTarget | null): HTMLElement | null {
  if (!target) return null
  if (target instanceof HTMLElement) return target
  if (target instanceof SVGElement) return target.closest('*') as HTMLElement | null
  if (target instanceof Node) return target.parentElement
  return null
}

function normalizeDeltaY(event: WheelEvent): number {
  if (event.deltaMode === WheelEvent.DOM_DELTA_LINE) return event.deltaY * 16
  if (event.deltaMode === WheelEvent.DOM_DELTA_PAGE) return event.deltaY * window.innerHeight * 0.85
  return event.deltaY
}

function isScrollable(el: HTMLElement): boolean {
  const style = window.getComputedStyle(el)
  const overflowY = style.overflowY
  if (!['auto', 'scroll', 'overlay'].includes(overflowY)) return false
  return el.scrollHeight > el.clientHeight + 1
}

function collectScrollableAncestors(target: EventTarget | null): HTMLElement[] {
  const chain: HTMLElement[] = []
  let el = resolveElement(target)

  while (el) {
    if (isScrollable(el)) chain.push(el)
    el = el.parentElement
  }

  const docScroller = document.scrollingElement
  if (docScroller instanceof HTMLElement && isScrollable(docScroller) && !chain.includes(docScroller)) {
    chain.push(docScroller)
  }

  return chain
}

function consumeScroll(el: HTMLElement, deltaY: number): number {
  if (deltaY === 0) return 0

  if (deltaY < 0) {
    const room = Math.max(el.scrollTop, 0)
    if (room <= 0) return 0
    const applied = -Math.min(room, -deltaY)
    el.scrollTop += applied
    return applied
  }

  const maxScrollTop = Math.max(el.scrollHeight - el.clientHeight, 0)
  const room = Math.max(maxScrollTop - el.scrollTop, 0)
  if (room <= 0) return 0
  const applied = Math.min(room, deltaY)
  el.scrollTop += applied
  return applied
}

export function installWheelScrollAssist() {
  if (installed || typeof window === 'undefined' || typeof document === 'undefined') return
  installed = true

  document.addEventListener(
    'wheel',
    event => {
      if (event.defaultPrevented || event.ctrlKey || event.metaKey) return
      if (Math.abs(event.deltaY) < Math.abs(event.deltaX)) return

      const startEl = resolveElement(event.target)
      if (!startEl) return

      const chain = collectScrollableAncestors(startEl)
      if (chain.length === 0) return

      let remaining = normalizeDeltaY(event)
      let handled = false

      for (const el of chain) {
        if (Math.abs(remaining) <= 0.5) break
        const consumed = consumeScroll(el, remaining)
        if (consumed !== 0) {
          remaining -= consumed
          handled = true
        }
      }

      if (handled) event.preventDefault()
    },
    { passive: false },
  )
}
