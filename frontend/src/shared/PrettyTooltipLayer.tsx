import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'

type TooltipState = {
  id: number
  text: string
  rect: DOMRect
}

type TooltipPosition = {
  left: number
  top: number
}

const VIEWPORT_GAP = 12
const ANCHOR_GAP = 10

function clamp(value: number, min: number, max: number) {
  if (max < min) return min
  return Math.min(Math.max(value, min), max)
}

function tooltipTarget(node: EventTarget | null): HTMLElement | null {
  return node instanceof Element ? node.closest<HTMLElement>('[data-tooltip]') : null
}

export default function PrettyTooltipLayer() {
  const [tooltip, setTooltip] = useState<TooltipState | null>(null)
  const [position, setPosition] = useState<TooltipPosition | null>(null)
  const activeTargetRef = useRef<HTMLElement | null>(null)
  const panelRef = useRef<HTMLDivElement | null>(null)
  const hideTimerRef = useRef<number | null>(null)

  const clearHideTimer = () => {
    if (hideTimerRef.current != null) {
      window.clearTimeout(hideTimerRef.current)
      hideTimerRef.current = null
    }
  }

  const hideSoon = () => {
    clearHideTimer()
    hideTimerRef.current = window.setTimeout(() => {
      activeTargetRef.current = null
      setTooltip(null)
      setPosition(null)
    }, 420)
  }

  const showForTarget = (target: HTMLElement) => {
    const text = target.dataset.tooltip?.trim()
    if (!text) return
    clearHideTimer()
    activeTargetRef.current = target
    const rect = target.getBoundingClientRect()
    setTooltip(current => ({
      id: current ? current.id + 1 : 1,
      text,
      rect,
    }))
    setPosition({
      left: clamp(rect.left, VIEWPORT_GAP, window.innerWidth - VIEWPORT_GAP),
      top: rect.bottom + ANCHOR_GAP,
    })
  }

  useEffect(() => {
    const onPointerOver = (event: PointerEvent) => {
      const target = tooltipTarget(event.target)
      if (target) {
        showForTarget(target)
      }
    }

    const onPointerOut = (event: PointerEvent) => {
      const activeTarget = activeTargetRef.current
      if (!activeTarget) return
      const next = event.relatedTarget
      if (
        next instanceof Node &&
        (activeTarget.contains(next) || panelRef.current?.contains(next))
      ) {
        return
      }
      hideSoon()
    }

    const onFocusIn = (event: FocusEvent) => {
      const target = tooltipTarget(event.target)
      if (target) {
        showForTarget(target)
      }
    }

    const onFocusOut = (event: FocusEvent) => {
      const next = event.relatedTarget
      if (next instanceof Node && panelRef.current?.contains(next)) {
        return
      }
      hideSoon()
    }

    const hideNow = () => {
      clearHideTimer()
      activeTargetRef.current = null
      setTooltip(null)
      setPosition(null)
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        hideNow()
      }
    }

    document.addEventListener('pointerover', onPointerOver)
    document.addEventListener('pointerout', onPointerOut)
    document.addEventListener('focusin', onFocusIn)
    document.addEventListener('focusout', onFocusOut)
    document.addEventListener('keydown', onKeyDown)
    window.addEventListener('resize', hideNow)
    window.addEventListener('scroll', hideNow, true)

    return () => {
      clearHideTimer()
      document.removeEventListener('pointerover', onPointerOver)
      document.removeEventListener('pointerout', onPointerOut)
      document.removeEventListener('focusin', onFocusIn)
      document.removeEventListener('focusout', onFocusOut)
      document.removeEventListener('keydown', onKeyDown)
      window.removeEventListener('resize', hideNow)
      window.removeEventListener('scroll', hideNow, true)
    }
  }, [])

  useLayoutEffect(() => {
    if (!tooltip || !panelRef.current) return

    const panel = panelRef.current
    const panelRect = panel.getBoundingClientRect()
    let left = tooltip.rect.left
    let top = tooltip.rect.bottom + ANCHOR_GAP

    if (left + panelRect.width > window.innerWidth - VIEWPORT_GAP) {
      left = window.innerWidth - panelRect.width - VIEWPORT_GAP
    }
    left = clamp(left, VIEWPORT_GAP, window.innerWidth - panelRect.width - VIEWPORT_GAP)

    if (top + panelRect.height > window.innerHeight - VIEWPORT_GAP) {
      top = tooltip.rect.top - panelRect.height - ANCHOR_GAP
    }
    top = clamp(top, VIEWPORT_GAP, window.innerHeight - panelRect.height - VIEWPORT_GAP)

    setPosition({ left, top })
  }, [tooltip])

  if (!tooltip) {
    return null
  }

  return createPortal(
    <div
      ref={panelRef}
      className="pretty-tooltip-panel"
      style={{
        left: position?.left ?? tooltip.rect.left,
        top: position?.top ?? tooltip.rect.bottom + ANCHOR_GAP,
      }}
      onPointerEnter={clearHideTimer}
      onPointerLeave={hideSoon}
      role="tooltip"
    >
      {tooltip.text}
    </div>,
    document.body,
  )
}
