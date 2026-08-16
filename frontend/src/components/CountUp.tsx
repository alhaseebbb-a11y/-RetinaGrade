import { useEffect, useRef, useState } from 'react'
import { animate, useInView } from 'motion/react'

interface CountUpProps {
  value: number
  duration?: number
  decimals?: number
  className?: string
}

export default function CountUp({ value, duration = 1.2, decimals = 0, className }: CountUpProps) {
  const ref = useRef<HTMLSpanElement>(null)
  const inView = useInView(ref, { once: true, margin: '-40px' })
  const [display, setDisplay] = useState('0')

  useEffect(() => {
    if (!inView) return
    const controls = animate(0, value, {
      duration,
      ease: 'easeOut',
      onUpdate: (v) => {
        setDisplay(
          decimals > 0
            ? v.toFixed(decimals)
            : Math.round(v).toLocaleString(),
        )
      },
    })
    return () => controls.stop()
  }, [inView, value, duration, decimals])

  return (
    <span ref={ref} className={className}>
      {display}
    </span>
  )
}
