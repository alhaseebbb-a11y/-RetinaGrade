# Deploy to Render + Vercel

Two-part deployment:

```
Vercel (React frontend, static)  ──fetch /api/*──►  Render (FastAPI + TensorFlow model)
```

- **Frontend** → [Vercel](https://vercel.com) (free). The React SPA is built to static files
  and served worldwide.
- **Backend** → [Render](https://render.com) (free or paid). Runs the FastAPI service and the
  127 MB Keras model on CPU.

The model file is too large for GitHub (100 MB/file limit), so it is hosted on **Hugging Face**
and the backend downloads it once at startup (`MODEL_URL` env var).

---

## 0. Before you start

You need accounts at:
1. [Hugging Face](https://huggingface.co) — to host the model file
2. [Render](https://render.com) — to run the backend
3. [Vercel](https://vercel.com) — to host the frontend

And a GitHub account with this repo pushed to it
(`alhaseebbb-a11y/Diabetic-Retinopathy-Detection-and-Severity-Grading-`).

---

## Step 1 — Upload the model to Hugging Face

```bash
source setenv.sh                       # from the repo root
pip install huggingface_hub
export HF_TOKEN=hf_XXXXXXXXXXXXXXXXXX  # https://huggingface.co/settings/tokens
python backend/push_model_to_hf.py <your-hf-username>/dr-grade-model
```

Copy the download URL it prints — you will paste it into Render as `MODEL_URL`
(step 3). It looks like:

```
https://huggingface.co/<your-hf-username>/dr-grade-model/resolve/main/best_model.keras
```

> Use a **public** repo so the backend can download it without credentials.
> `outputs/best_model.keras` stays out of git — only `outputs/test_metrics.json` is committed.

---

## Step 2 — Push the repo to GitHub

```bash
git add -A
git commit -m "Add deployment support (Vercel + Render)"
git push origin main
```

Confirm these are pushed (the model `.keras` must NOT be in the list):

```
frontend/            frontend React app
backend/             FastAPI service + requirements-render.txt + push_model_to_hf.py
ordinal.py           model helpers (imported by the backend)
outputs/test_metrics.json
```

---

## Step 3 — Deploy the backend on Render

1. [render.com](https://render.com) → **New** → **Web Service** → connect your GitHub repo.
2. Settings:
   - **Root Directory:** `backend`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements-render.txt`
   - **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Instance Type:** Free (risky, see note below) or Starter.
3. **Environment Variables:**
   - `MODEL_URL` = the URL from Step 1
   - `BACKEND_DEVICE` = `cpu`
4. Click **Create Web Service** and wait for the first deploy (~5–10 min on free tier:
   installing TensorFlow takes a while).

When it says **Live**, open the URL it gives you and check:

- `https://<your-app>.onrender.com/api/health` → `{"status":"ok", ...}` (the first call may
  take 1–2 min to download the model; wait, then refresh).
- `https://<your-app>.onrender.com/docs` → Swagger UI. Click **POST /api/predict → Try it out**,
  upload one of the `sample_dataset/` images, and confirm a grade comes back.

> **Free tier caveats:** Render free services sleep after ~15 min idle (first request is slow),
> have 512 MB RAM, and get 750 free hours/month. TensorFlow + the 127 MB model in RAM is tight;
> if you see memory errors, upgrade to Starter (safer) or convert the model to TFLite.

---

## Step 4 — Deploy the frontend on Vercel

1. [vercel.com](https://vercel.com) → **Add New** → **Project** → import the same GitHub repo.
2. Settings:
   - **Root Directory:** `frontend`
   - **Framework Preset:** `Vite`
   - Build Command `npm run build`, Output Directory `dist` (usually auto-detected).
3. **Environment Variable:**
   - `VITE_API_BASE` = `https://<your-app>.onrender.com` (the Render URL from Step 3, **no
     trailing slash**)
4. Click **Deploy**.

When it goes live, open the Vercel URL (`https://<your-project>.vercel.app`):

- The page loads and shows **AI ready** (green dot in the header).
- Upload an eye photo → grade + probability bars appear. The model runs on Render, not Vercel.

---

## How the pieces talk to each other

- In local dev the Vite proxy sends `/api/*` → `localhost:8002`.
- In production the built SPA reads `VITE_API_BASE` and calls
  `https://<backend>.onrender.com/api/*` directly.
- The backend's CORS allows any `*.vercel.app` origin, so the browser permits the call.
- The backend downloads `best_model.keras` from `MODEL_URL` once (into `outputs/`) at startup
  and loads it with TensorFlow CPU.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Model not found ... MODEL_URL is not set` | Set `MODEL_URL` env var on Render and redeploy. |
| Frontend shows **AI not connected** | Check `VITE_API_BASE` on Vercel, and that `/api/health` works in a browser. |
| Browser shows CORS error | Vercel domain must end in `.vercel.app`. If you add a custom domain, add it to `allow_origins` in `backend/app.py`. |
| First prediction is very slow | Free-tier Render sleeps after ~15 min. Just wait — subsequent calls are fast. |
| Render deploy fails with `pip` error | Confirm Build Command is `pip install -r requirements-render.txt` (not the CUDA `requirements.txt`). |
| Memory errors on Render | Upgrade to Starter, or convert the model to TFLite. |

## Updating after retraining

Re-run Step 1 (upload the new model to HF), then on Render: **Deploy → Deploy latest commit**
(or just use **Manual Deploy**), and it re-downloads the new weights.
