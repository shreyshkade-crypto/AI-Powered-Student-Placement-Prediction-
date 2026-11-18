# Placement Predictor — Project Folder

Everything is in this one folder. Open this whole folder in **VS Code**
(`File → Open Folder…`), then use the steps below.

```
placement-predictor/
├── placement_predictor.html      ← the website (frontend)
├── backend/
│   ├── app.py                    ← the Flask server
│   └── requirements.txt          ← Python packages it needs
└── README.md                     ← this file
```

## 1. Open the integrated terminal

In VS Code: **Terminal → New Terminal** (or `` Ctrl+` ``).

## 2. Install the backend's dependencies

Run these two commands one after another in the terminal:

```bash
cd backend
pip install -r requirements.txt
```

If `pip` isn't found, try `pip3` instead. If Python itself isn't
installed, install it from python.org first (tick "Add Python to PATH"
during setup on Windows), then reopen the terminal.

## 3. Start the backend

Still inside the `backend` folder, run:

```bash
python app.py
```

You should see something like:

```
 * Running on http://127.0.0.1:5000
```

Leave this terminal running — closing it stops the server. It creates
a `placement.db` SQLite file in the `backend` folder automatically the
first time it runs.

## 4. Open the website

In VS Code's file explorer (left sidebar), right-click
`placement_predictor.html` → **"Reveal in File Explorer"** (or Finder
on Mac), then double-click the file to open it in your browser.

(If you have the **Live Server** VS Code extension installed, you can
instead right-click the file → **"Open with Live Server"** — either
way works fine, the file doesn't need to be served to run.)

## 5. Connect the website to the backend

In the browser tab, click the **cloud-sync icon** (top-right of the
page), type any roll number, and click **"Connect & load"**. You
should see "Connected as \<roll number\> — profile loaded." Your data
now saves to `backend/placement.db` whenever you click **"Save now"**.

## Using two terminals at once (optional but convenient)

You can split the VS Code terminal (click the split-terminal icon) so
one pane keeps `python app.py` running, and you can still type other
commands in the other pane without stopping the server.

## If something doesn't work

- **`ModuleNotFoundError: No module named 'flask'`** → the
  `pip install -r requirements.txt` step didn't complete — re-run it
  and check for errors.
- **`DLL load failed` / "Application Control policy has blocked this
  file"** → this was a real issue on some locked-down Windows machines
  with an older version of this backend that used SQLAlchemy. The
  current backend uses only Python's built-in `sqlite3` module and
  Flask/Flask-Cors — no compiled extensions at all — so this shouldn't
  happen anymore. If you still see it, make sure `requirements.txt`
  only lists Flask and Flask-Cors (no SQLAlchemy).
- **Browser shows "Could not reach the backend"** → make sure the
  `python app.py` terminal is still running and shows no errors.
- **Port 5000 already in use** → close whatever else is using it, or
  edit the last line of `backend/app.py` to use a different port
  (e.g. `app.run(debug=True, port=5050)`) and update the Backend URL
  field in the website's sync panel to match.

Full backend API reference is in `backend/README.md`, along with the
comments at the top of `backend/app.py`.
