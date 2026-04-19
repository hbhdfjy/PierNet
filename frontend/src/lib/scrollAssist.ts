let installed = false

const PAGE_SCROLL_SELECTORS = [
  '.training-page__body',
  '.workbench-main-scroll',
  '.page-content',
  '.page-shell',
]

function resolveElement(target: EventTarget | null): HTMLElement | null {
  if (!target) return null
  if (target instanceof HTMLElement) return target
  if (target instanceof SVGElement) return target.closest('*') as HTMLElement | null
  if (target instanceof Node) return target.parentElement
  return null
}

function isPageScroller(el: HTMLElement): boolean {
  return PAGE_SCROLL_SELECTORS.some(selector => el.matches(selector))
}

function isScrollable(el: HTMLElement): boolean {
  const style = window.getComputedStyle(el)
  const overflowY = style.overflowY
  if (!['auto', 'scroll', 'overlay'].includes(overflowY)) return false
  return el.scrollHeight > el.clientHeight + 1
}

function canScroll(el: HTMLElement, deltaY: number): boolean {
  if (deltaY < 0) return el.scrollTop > 0
  if (deltaY > 0) return el.scrollTop + el.clientHeight < el.scrollHeight - 1
  return false
}

function findNearestScrollableAncestor(
  target: EventTarget | null,
  deltaY: number,
  options: { excludePage?: boolean } = {},
): HTMLElement | null {
  let el = resolveElement(target)
  while (el) {
    if (isScrollable(el) && (!options.excludePage || !isPageScroller(el)) && canScroll(el, deltaY)) {
      return el
    }
    el = el.parentElement
  }
  return null
}

function findPageScroller(target: EventTarget | null, deltaY: number): HTMLElement | null {
  let el = resolveElement(target)
  while (el) {
    if (isPageScroller(el) && isScrollable(el) && canScroll(el, deltaY)) {
      return el
    }
    el = el.parentElement
  }

  const docScroller = document.scrollingElement
  if (docScroller instanceof HTMLElement && isScrollable(docScroller) && canScroll(docScroller, deltaY)) {
    return docScroller
  }

  return null
}

function applyScroll(el: HTMLElement | null, deltaY: number): number {
  if (!el || deltaY === 0) return 0
  const before = el.scrollTop
  el.scrollTop += deltaY
  return el.scrollTop - before
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

      let remaining = event.deltaY
      let handled = false

      const componentScroller = findNearestScrollableAncestor(startEl, remaining, { excludePage: true })
      if (componentScroller) {
        const consumed = applyScroll(componentScroller, remaining)
        if (consumed !== 0) {
          remaining -= consumed
          handled = true
        }
      }

      if (Math.abs(remaining) > 0.5) {
        const pageScroller = findPageScroller(startEl, remaining)
        if (pageScroller && pageScroller !== componentScroller) {
          const consumed = applyScroll(pageScroller, remaining)
          if (consumed !== 0) {
            remaining -= consumed
            handled = true
          }
        }
      }

      if (!handled) {
        const fallback = findNearestScrollableAncestor(startEl, event.deltaY)
        if (fallback) {
          const consumed = applyScroll(fallback, event.deltaY)
          if (consumed !== 0) handled = true
        }
      }

      if (handled) event.preventDefault()
    },
    { passive: false },
  )
}
