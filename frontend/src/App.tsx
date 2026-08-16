import { useCallback, useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { BrainCircuit, FlaskConical, ScanSearch, Sparkles } from 'lucide-react'
import Header from './components/Header'
import UploadZone from './components/UploadZone'
import ResultsPanel from './components/ResultsPanel'
import ModelDashboard from './components/ModelDashboard'
import ReportExport from './components/ReportExport'
import { getHealth, getMetrics, predictImage } from './lib/api'
import type { HealthResponse, MetricsResponse, PredictResponse } from './lib/types'

interface ToastState {
  type: 'error' | 'info'
  message: string
}

function Toast({ toast }: { toast: ToastState | null }) {
  return (
    <AnimatePresence>
      {toast && (
        <motion.div
          initial={{ opacity: 0, y: 24, scale: 0.96 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: 12, scale: 0.98 }}
          className="fixed bottom-6 left-1/2 z-50 -translate-x-1/2"
        >
          <div
            className={`flex items-center gap-2 rounded-2xl px-5 py-3 text-sm font-semibold shadow-soft ${
              toast.type === 'error'
                ? 'bg-red-600 text-white'
                : 'bg-clinical-900 text-white'
            }`}
          >
            {toast.message}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

export default function App() {
  const [image, setImage] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [tta, setTta] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [backendOnline, setBackendOnline] = useState(false)
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null)
  const [toast, setToast] = useState<ToastState | null>(null)
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const resultsRef = useRef<HTMLDivElement>(null)

  const showToast = useCallback((t: ToastState) => {
    setToast(t)
    if (toastTimer.current) clearTimeout(toastTimer.current)
    toastTimer.current = setTimeout(() => setToast(null), 4200)
  }, [])

  useEffect(() => {
    const onToast = (e: Event) => {
      const detail = (e as CustomEvent<ToastState>).detail
      if (detail) showToast(detail)
    }
    window.addEventListener('app-toast', onToast)
    return () => window.removeEventListener('app-toast', onToast)
  }, [showToast])

  useEffect(() => {
    let alive = true
    const load = async () => {
      try {
        const h = await getHealth()
        if (!alive) return
        setHealth(h)
        setBackendOnline(h.status === 'ok')
        if (h.status === 'ok') {
          const m = await getMetrics()
          if (alive) setMetrics(m)
        }
      } catch {
        if (alive) setBackendOnline(false)
      }
    }
    load()
    const interval = setInterval(load, 8000)
    return () => {
      alive = false
      clearInterval(interval)
    }
  }, [])

  const handleImage = useCallback((file: File | null) => {
    setImage(file)
    setResult(null)
    setError(null)
    if (previewUrl) URL.revokeObjectURL(previewUrl)
    setPreviewUrl(file ? URL.createObjectURL(file) : null)
  }, [previewUrl])

  const runPrediction = useCallback(async () => {
    if (!image) return
    if (!backendOnline) {
      setError('Backend offline — start the FastAPI service, then try again.')
      return
    }
    setLoading(true)
    setError(null)
    try {
      const res = await predictImage(image, tta)
      setResult(res)
      requestAnimationFrame(() => {
        setTimeout(() => {
          resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
        }, 80)
      })
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Prediction failed'
      setError(msg)
      showToast({ type: 'error', message: msg })
    } finally {
      setLoading(false)
    }
  }, [image, tta, backendOnline, showToast])

  return (
    <div className="bg-aurora min-h-screen">
      <Header health={health} metricsQwk={metrics?.qwk ?? null} backendOnline={backendOnline} />

      <main className="mx-auto max-w-6xl px-5 pb-24 pt-14 sm:pt-16">
        {/* Hero */}
        <motion.section
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55, ease: [0.16, 1, 0.3, 1] }}
          className="text-center"
        >
          <span className="inline-flex items-center gap-2 rounded-full border border-clinical-200 bg-white/70 px-4 py-1.5 text-xs font-semibold text-clinical-600 shadow-sm backdrop-blur">
            <Sparkles size={13} className="text-accent-500" />
            EfficientNet-B3 · CORAL ordinal · QWK 0.8718 on the test set
          </span>
          <h1 className="mx-auto mt-5 max-w-3xl font-display text-4xl font-extrabold leading-[1.1] text-clinical-900 sm:text-5xl">
            Grade diabetic retinopathy from a{' '}
            <span className="bg-gradient-to-r from-clinical-600 to-accent-500 bg-clip-text text-transparent">
              single fundus image
            </span>
          </h1>
          <p className="mx-auto mt-4 max-w-xl text-base text-clinical-600">
            Upload a retinal photograph and get an instant, explainable severity grade —
            No DR to Proliferative DR — with per-class confidence.
          </p>
        </motion.section>

        {/* Upload + controls */}
        <motion.section
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
          className="mx-auto mt-10 max-w-3xl"
        >
          <UploadZone
            image={image}
            previewUrl={previewUrl}
            loading={loading}
            error={error}
            onImage={handleImage}
          />

          <div className="mt-5 flex flex-col items-center justify-between gap-4 rounded-2xl border border-clinical-100 bg-white/70 px-5 py-4 shadow-card backdrop-blur sm:flex-row">
            {/* TTA toggle */}
            <label className="flex cursor-pointer items-center gap-3">
              <button
                type="button"
                role="switch"
                aria-checked={tta}
                onClick={() => setTta((v) => !v)}
                className={`relative h-6 w-11 shrink-0 rounded-full transition-colors duration-300 ${
                  tta ? 'bg-accent-500' : 'bg-clinical-200'
                }`}
              >
                <span
                  className={`absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition-all duration-300 ${
                    tta ? 'left-[22px]' : 'left-0.5'
                  }`}
                />
              </button>
              <span>
                <span className="block text-sm font-semibold text-clinical-800">
                  Test-time augmentation
                </span>
                <span className="block text-xs text-clinical-500">
                  Average 4 flips for more stable predictions
                </span>
              </span>
            </label>

            <button
              type="button"
              onClick={runPrediction}
              disabled={!image || loading || !backendOnline}
              className="group relative inline-flex min-w-[210px] items-center justify-center gap-2.5 overflow-hidden rounded-xl bg-gradient-to-r from-clinical-600 to-accent-500 px-7 py-3.5 text-sm font-bold text-white shadow-soft transition hover:shadow-glow active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? (
                <>
                  <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" />
                  Analyzing…
                </>
              ) : (
                <>
                  <ScanSearch size={17} className="transition-transform group-hover:scale-110" />
                  Analyze image
                </>
              )}
            </button>
          </div>
        </motion.section>

        {/* Results */}
        <div ref={resultsRef} className="mt-14 scroll-mt-24">
          <AnimatePresence mode="wait">
            {result && (
              <motion.div
                key="result"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <div className="mb-5 flex items-center justify-between">
                  <p className="text-sm font-semibold text-clinical-500">
                    <BrainCircuit size={15} className="mr-1.5 inline text-accent-600" />
                    Result for <span className="font-bold text-clinical-800">{result.filename}</span>
                  </p>
                  <ReportExport result={result} />
                </div>
                <ResultsPanel
                  result={result}
                  onReset={() => handleImage(null)}
                />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Dashboard */}
        <ModelDashboard metrics={metrics} loading={!metrics && backendOnline} />

        {/* How it works */}
        <motion.section
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-60px' }}
          transition={{ duration: 0.5 }}
          className="no-print mt-16 grid gap-4 sm:grid-cols-3"
        >
          {[
            { icon: FlaskConical, title: 'Upload', text: 'Drop a standard fundus photograph (JPEG/PNG).' },
            { icon: BrainCircuit, title: 'AI grades it', text: 'EfficientNet-B3 extracts features; the CORAL head outputs 5 severity probabilities.' },
            { icon: ScanSearch, title: 'Review & export', text: 'See confidence per grade, and export a report or JSON for your records.' },
          ].map((step, i) => (
            <motion.div
              key={step.title}
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: i * 0.1 }}
              className="rounded-3xl border border-clinical-100 bg-white p-6 shadow-card"
            >
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-clinical-100 to-accent-400/20 text-accent-600">
                <step.icon size={19} />
              </div>
              <h3 className="mt-4 font-display text-base font-bold text-clinical-900">
                {i + 1}. {step.title}
              </h3>
              <p className="mt-1.5 text-sm leading-relaxed text-clinical-500">{step.text}</p>
            </motion.div>
          ))}
        </motion.section>
      </main>

      <footer className="no-print border-t border-clinical-100 bg-white/60 py-6 text-center text-xs text-clinical-400 backdrop-blur">
        RetinaGrade · EfficientNet-B3 CORAL ordinal classifier · research/demo only, not a
        medical device · test QWK 0.8718
      </footer>

      <Toast toast={toast} />
    </div>
  )
}
