import { motion } from 'motion/react'
import { Activity, ScanEye } from 'lucide-react'
import type { HealthResponse } from '../lib/types'
import CountUp from './CountUp'

interface HeaderProps {
  health: HealthResponse | null
  metricsQwk: number | null
  backendOnline: boolean
}

export default function Header({ health, metricsQwk, backendOnline }: HeaderProps) {
  return (
    <motion.header
      initial={{ opacity: 0, y: -16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      className="no-print sticky top-0 z-40 border-b border-clinical-100/80 bg-white/70 backdrop-blur-xl"
    >
      <div className="mx-auto flex max-w-6xl items-center justify-between px-5 py-3.5">
        <div className="flex items-center gap-3">
          <div className="relative flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-clinical-500 to-accent-500 shadow-soft">
            <ScanEye className="text-white" size={20} />
            <span className="absolute -right-0.5 -top-0.5 h-2.5 w-2.5 rounded-full bg-accent-400 ring-2 ring-white" />
          </div>
          <div>
            <p className="font-display text-base font-bold leading-tight text-clinical-900">
              Retina<span className="text-accent-600">Grade</span>
            </p>
            <p className="text-[11px] font-medium leading-tight text-clinical-500">
              Diabetic Retinopathy Severity AI
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="hidden items-center gap-2 rounded-full border border-clinical-200 bg-white/80 px-3.5 py-1.5 shadow-sm sm:flex">
            <Activity size={14} className="text-accent-600" />
            <span className="text-xs font-semibold text-clinical-700">QWK</span>
            <span className="font-mono text-xs font-bold text-accent-600">
              {metricsQwk !== null ? <CountUp value={metricsQwk} decimals={3} duration={1.4} /> : '—'}
            </span>
          </div>
          <div className="flex items-center gap-2 rounded-full border border-clinical-200 bg-white/80 px-3 py-1.5 shadow-sm">
            <span
              className={`h-2 w-2 rounded-full ${
                backendOnline ? 'bg-emerald-500 animate-glow-pulse' : 'bg-red-400'
              }`}
            />
            <span className="text-xs font-semibold text-clinical-700">
              {backendOnline
                ? health?.model.device === 'gpu'
                  ? 'Model · GPU'
                  : 'Model · CPU'
                : 'Backend offline'}
            </span>
          </div>
        </div>
      </div>
    </motion.header>
  )
}
