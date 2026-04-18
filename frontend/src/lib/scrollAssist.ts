let installed = false

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

function findScrollableAncestor(target: EventTarget | null, deltaY: number): HTMLElement | null {
  let el = target instanceof HTMLElement ? target : null

  while (el) {
    if (isScrollable(el) && canScroll(el, deltaY)) return el
    el = el.parentElement
  }

  const root = document.scrollingElement
  return root instanceof HTMLElement && isScrollable(root) && canScroll(root, deltaY) ? root : null
}

export function installWheelScrollAssist() {
  if (installed || typeof window === 'undefined' || typeof document === 'undefined') return
  installed = true

  document.addEventListener(
    'wheel',
    event => {
      if (event.defaultPrevented || event.ctrlKey || event.metaKey) return
      if (Math.abs(event.deltaY) < Math.abs(event.deltaX)) return

      const scrollable = findScrollableAncestor(event.target, event.deltaY)
      if (!scrollable) return

      const before = scrollable.scrollTop
      scrollable.scrollTop += event.deltaY

      if (scrollable.scrollTop !== before) {
        event.preventDefault()
      }
    },
    { passive: false },
  )
}
