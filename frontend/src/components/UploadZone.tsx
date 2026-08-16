import { useState } from 'react'
import { motion, AnimatePresence } from 'motion/react'
import { CheckCircle2, Eye, X } from 'lucide-react'

interface UploadZoneProps {
  image: File | null
  previewUrl: string | null
  loading: boolean
  error: string | null
  onImage: (file: File | null) => void
}

export default function UploadZone({ image, previewUrl, loading, error, onImage }: UploadZoneProps) {
  const [dragging, setDragging] = useState(false)

  const handleFiles = (files: FileList | null) => {
    if (!files || files.length === 0) return
    const file = files[0]
    if (!/^image\/(jpeg|png|webp|bmp)$/.test(file.type)) {
      onImage(null)
      window.dispatchEvent(
        new CustomEvent('app-toast', {
          detail: { type: 'error', message: 'Please upload a JPEG, PNG, WEBP or BMP image.' },
        }),
      )
      return
    }
    onImage(file)
  }

  return (
    <div
      className="relative"
      onDragOver={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        handleFiles(e.dataTransfer.files)
      }}
    >
      <motion.div
        initial={{ opacity: 0, y: 18 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        className={`relative overflow-hidden rounded-3xl border-2 border-dashed bg-white/80 backdrop-blur transition-all ${
          dragging ? 'dropzone-active' : 'border-clinical-200'
        } p-8 shadow-card sm:p-10`}
      >
        <AnimatePresence mode="wait">
          {previewUrl ? (
            <motion.div
              key="preview"
              initial={{ opacity: 0, scale: 0.96 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.98 }}
              transition={{ duration: 0.35 }}
              className="flex flex-col items-center gap-6 sm:flex-row"
            >
              <div className="relative shrink-0">
                <div className="absolute -inset-1 rounded-2xl bg-gradient-to-tr from-clinical-200 to-accent-400/40 blur-sm" />
                <img
                  src={previewUrl}
                  alt="Fundus image preview"
                  className="relative h-44 w-44 rounded-2xl border border-white object-cover shadow-md sm:h-52 sm:w-52"
                />
                <span className="absolute bottom-2 left-2 flex items-center gap-1 rounded-full bg-clinical-900/80 px-2.5 py-1 text-[10px] font-semibold text-white backdrop-blur">
                  <Eye size={11} /> Fundus scan
                </span>
              </div>

              <div className="min-w-0 flex-1 text-center sm:text-left">
                <p className="truncate text-lg font-semibold text-clinical-900">{image?.name}</p>
                <p className="mt-1 text-sm text-clinical-500">
                  {image ? `${(image.size / 1024 / 1024).toFixed(2)} MB · ready to analyze` : ''}
                </p>
                <div className="mt-5 flex flex-wrap items-center justify-center gap-3 sm:justify-start">
                  <button
                    type="button"
                    disabled={loading}
                    onClick={() => {
                      const input = document.createElement('input')
                      input.type = 'file'
                      input.accept = 'image/*'
                      input.onchange = () => handleFiles(input.files)
                      input.click()
                    }}
                    className="inline-flex items-center gap-2 rounded-xl border border-clinical-200 bg-white px-4 py-2 text-sm font-semibold text-clinical-700 shadow-sm transition hover:border-clinical-300 hover:bg-clinical-50 disabled:opacity-50"
                  >
                    <CheckCircle2 size={15} /> Change image
                  </button>
                  <button
                    type="button"
                    disabled={loading}
                    onClick={() => onImage(null)}
                    className="inline-flex items-center gap-2 rounded-xl px-4 py-2 text-sm font-medium text-clinical-500 transition hover:text-clinical-700 disabled:opacity-50"
                  >
                    <X size={15} /> Remove
                  </button>
                </div>
              </div>
            </motion.div>
          ) : (
            <motion.div
              key="drop"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="flex flex-col items-center gap-4 py-6 text-center"
            >
              <motion.div
                animate={{ y: [0, -6, 0] }}
                transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
                className="flex h-20 w-20 items-center justify-center rounded-2xl bg-gradient-to-br from-clinical-100 to-accent-400/20 ring-1 ring-clinical-200"
              >
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" className="text-accent-600">
                  <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="1.6" opacity="0.4" />
                  <circle cx="12" cy="12" r="4.5" fill="currentColor" opacity="0.85" />
                </svg>
              </motion.div>
              <div>
                <p className="text-lg font-semibold text-clinical-900">
                  Drag &amp; drop your fundus image
                </p>
                <p className="mt-1 text-sm text-clinical-500">
                  or{' '}
                  <button
                    type="button"
                    className="font-semibold text-accent-600 underline-offset-2 hover:underline"
                    onClick={() => {
                      const input = document.createElement('input')
                      input.type = 'file'
                      input.accept = 'image/*'
                      input.onchange = () => handleFiles(input.files)
                      input.click()
                    }}
                  >
                    browse files
                  </button>{' '}
                  — JPEG, PNG, WEBP up to 10 MB
                </p>
              </div>
              <p className="text-xs text-clinical-400">
                The image is analyzed locally by the EfficientNet-B3 model (GPU-accelerated).
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>

      {error && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-4 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm font-medium text-red-700"
        >
          {error}
        </motion.div>
      )}
    </div>
  )
}
