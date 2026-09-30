# Cloud Deployment Guide & Live URLs for VARSHAA (RAIN-REPAIR X)

This guide provides the instructions and URL endpoints to deploy VARSHAA to free cloud platforms (**Render**, **Vercel**, and **GitHub Pages**).

---

## 1. Quick Comparison & Target Live Links

| Platform | Type | Deployed Live URL | Best For | Setup Time |
| :--- | :--- | :--- | :--- | :---: |
| **GitHub Pages** | Static & Presentation | `https://anjalijaiswal1011.github.io/VARSHAA/` | SIH Judges, Presentation Slides & UI Demo | 30 seconds |
| **Render** | Full-Stack (Backend + Frontend) | `https://varshaa-operational-platform.onrender.com` | Complete Operational System (FastAPI + AI Engine) | 2 minutes |
| **Vercel** | High-Speed Frontend SPA | `https://varshaa.vercel.app` (or custom subdomain) | Global Edge CDN Dashboard | 1 minute |

---

## 2. Option A: 1-Click Free Deployment on Render (Full-Stack Python + Frontend)

We have included `render.yaml` (Infrastructure-as-Code) at the root of the repository.

### Steps:
1. Push your latest code to GitHub:
   ```bash
   git add .
   git commit -m "feat(deploy): add render blueprint and cloud configs"
   git push origin main
   ```
2. Go to **[dashboard.render.com](https://dashboard.render.com/)** and sign in with your GitHub account.
3. Click **New +** in the top right and select **Blueprint**.
4. Connect the repository: **`Anjalijaiswal1011/VARSHAA`**.
5. Render will automatically read `render.yaml`, build the Docker container (`deployment/Dockerfile.backend`), and provide your live URL:
   - **Live App URL:** `https://varshaa-operational-platform.onrender.com/dashboard`
   - **Live API Docs:** `https://varshaa-operational-platform.onrender.com/docs`
   - **Health Check:** `https://varshaa-operational-platform.onrender.com/api/v1/health`

---

## 3. Option B: GitHub Pages (Automatic via GitHub Actions)

We have created `.github/workflows/deploy-pages.yml` to automatically publish both the **Dashboard** and the **SIH 2026 Presentation Slides** directly on GitHub Pages!

### Steps to activate:
1. Go to your repository settings on GitHub:  
   👉 **`https://github.com/Anjalijaiswal1011/VARSHAA/settings/pages`**
2. Under **Build and deployment > Source**, change the dropdown from *Deploy from a branch* to:
   - **GitHub Actions**
3. Push any commit to `main` (or click **Actions > Deploy Presentation Deck > Run workflow**).
4. Your live link will be instantly accessible at:
   - **Main Portal & Dashboard:** `https://anjalijaiswal1011.github.io/VARSHAA/`
   - **SIH 2026 Slide Deck:** `https://anjalijaiswal1011.github.io/VARSHAA/slides/sih_presentation_deck.html`
   - **Verification Slide:** `https://anjalijaiswal1011.github.io/VARSHAA/slides/model_verification_slide.html`

---

## 4. Option C: Frontend Deployment on Vercel

We have configured `frontend/vercel.json`.

### Steps:
1. Go to **[vercel.com](https://vercel.com/)** and log in with GitHub.
2. Click **Add New... > Project**.
3. Import **`Anjalijaiswal1011/VARSHAA`**.
4. In the configuration dialog:
   - **Root Directory:** Edit and select `frontend`
   - **Framework Preset:** Vite (automatically detected)
5. Click **Deploy**.
6. Within 45 seconds, Vercel gives you your production link:
   - **Live Link:** `https://varshaa.vercel.app` (or your chosen project name).
