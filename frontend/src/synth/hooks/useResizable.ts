import { useState, useCallback, useRef, useEffect } from 'react'

interface UseResizableOptions {
  defaultWidth: number
  minWidth: number
  maxWidth: number
  storageKey?: string
}

export function useResizable({ defaultWidth, minWidth, maxWidth, storageKey }: UseResizableOptions) {
  const [width, setWidth] = useState<number>(() => {
    if (storageKey) {
      const stored = localStorage.getItem(storageKey)
      if (stored) {
        const n = parseInt(stored, 10)
        if (!isNaN(n) && n >= minWidth && n <= maxWidth) return n
      }
    }
    return defaultWidth
  })

  // 用 ref 追踪拖拽状态，避免 stale closure 问题
  const dragging = useRef(false)
  const startX = useRef(0)
  const startW = useRef(0)
  // 把 minWidth/maxWidth/storageKey 也存进 ref，供 event handler 直接读取
  const optionsRef = useRef({ minWidth, maxWidth, storageKey })
  useEffect(() => {
    optionsRef.current = { minWidth, maxWidth, storageKey }
  }, [minWidth, maxWidth, storageKey])

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    dragging.current = true
    startX.current = e.clientX
    startW.current = width   // 记录拖拽开始时的宽度
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
  }, [width])

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (!dragging.current) return
      const { minWidth: mn, maxWidth: mx } = optionsRef.current
      const delta = e.clientX - startX.current
      const next = Math.min(mx, Math.max(mn, startW.current + delta))
      setWidth(next)
    }

    const onUp = (e: MouseEvent) => {
      if (!dragging.current) return
      dragging.current = false
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
      const { minWidth: mn, maxWidth: mx, storageKey: sk } = optionsRef.current
      const delta = e.clientX - startX.current
      const final = Math.min(mx, Math.max(mn, startW.current + delta))
      // Bug #3 fix: 同步更新 state，确保最终宽度正确
      setWidth(final)
      if (sk) {
        localStorage.setItem(sk, String(final))
      }
    }

    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
  // Bug #2 fix: event handlers 通过 optionsRef 读取最新值，不依赖闭包
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return { width, onMouseDown }
}
