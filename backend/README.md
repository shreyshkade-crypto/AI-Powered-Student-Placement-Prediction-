# Runway Backend

A small Flask + SQLAlchemy API that gives the Runway placement platform
(`placement_predictor.html`) real, persistent storage — a student's
profile, resume analysis, and probability history survive a refresh, a
browser restart, or opening the app on a different day.

## 1. Setup

```bash
cd backend
pip install -r requirements.txt
python app.py
```

The API starts at **http://localhost:5000** and creates a local SQLite
file (`placement.db`) automatically — no database server needed to try
it out.

## 2. Connect the frontend

Open `placement_predictor.html` in your browser, click the **cloud-sync
icon** in the top navigation bar, and:

1. Leave the Backend URL as `http://localhost:5000` (or change it if
   you deployed the API elsewhere).
2. Enter a Roll Number (this is the student's login key — no password,
   kept simple for a class project).
3. Click **Connect & load**. If that roll number already has saved
   data, it fills in the Dashboard automatically; if not, a fresh
   profile is created on the server.
4. Click **Save now** any time to push the current profile to the
   server. Logging an Analytics snapshot also saves to the server
   automatically once connected.

If the backend isn't running, the frontend keeps working exactly as
before (browser-only, `localStorage` for snapshots) — it just shows
"Offline" in the sync panel.

## 3. Turn on real AI for the chatbot (optional)

By default, the floating **AI Career Assistant** answers with a simple
rule-based engine (no internet call, works even fully offline). If you
want it to answer with real Claude AI instead — genuinely reading and
reasoning over the student's live profile — do this:

1. Get an API key from the **Claude Console**: go to
   https://console.anthropic.com, sign up / log in, open
   **Settings → API Keys**, and create a new key. (This requires
   adding billing to your Anthropic account — API usage is paid, though
   the cost per chat message here is tiny.)
2. Set it as an environment variable before starting the server:

   **Windows (PowerShell):**
   ```powershell
   $env:ANTHROPIC_API_KEY="sk-ant-your-key-here"
   python app.py
   ```

   **Windows (Command Prompt):**
   ```cmd
   set ANTHROPIC_API_KEY=sk-ant-your-key-here
   python app.py
   ```

   **Mac / Linux:**
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-your-key-here"
   python app.py
   ```
3. Restart the server if it was already running. Once connected, the
   frontend's chat header changes to say it's backed by real Claude AI,
   and every answer is grounded in that student's actual live profile
   (probability, weak areas, career matches, ATS score, roadmap) via a
   system prompt built server-side — the API key itself never touches
   the browser.

If no key is set, the chatbot silently falls back to the rule-based
engine — nothing breaks either way.

## 4. API reference

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/api/students` | Create (or fetch, if it already exists) a student by `roll_number` |
| GET | `/api/students/<roll_number>` | Get full profile + snapshots |
| PUT | `/api/students/<roll_number>` | Update any profile fields |
| GET | `/api/students/<roll_number>/snapshots` | List probability snapshots |
| POST | `/api/students/<roll_number>/snapshots` | Add a snapshot `{date, prob}` |
| DELETE | `/api/students/<roll_number>/snapshots` | Clear snapshot history |
| GET | `/api/stats/<roll_number>` | Anonymous peer stats: student count, average probability, your percentile |
| POST | `/api/ml-train` | Start retraining from `ml/placement_data.csv` (local computer only) |
| GET | `/api/ml-train-status` | Read the running/finished training log (local computer only) |
| POST | `/api/chat` | Ask the AI Career Assistant `{message, context}` → `{reply}` (needs `ANTHROPIC_API_KEY`) |

The ML Lab's **Train model now** button uses these local-only endpoints. It retrains the bundled dataset; student profiles entered in the website are not automatically added to the CSV.

## 5. Notes for your report / viva

- Storage is **SQLite via Python's built-in `sqlite3` module** — no
  SQLAlchemy, no separate database server, and no compiled/native
  dependencies. This was a deliberate choice: SQLAlchemy's newer
  versions ship a compiled Cython extension that gets blocked by
  Windows Application Control / Smart App Control policies on some
  locked-down machines. Plain `sqlite3` has none of that risk.
- This is a **development server** (Flask's built-in one) — fine for a
  class project demo, not for production. A real deployment would use
  something like Gunicorn behind Nginx, and a proper database server
  (MySQL/PostgreSQL) instead of a single SQLite file.
- There's no authentication — the roll number alone acts as the lookup
  key. For a real product you'd add password/OTP-based login.
- CORS is fully open (`CORS(app)`) so the static HTML file can call it
  from any origin. Lock this to your actual frontend's origin before
  deploying anywhere public.
