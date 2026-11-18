"""
Runway — Placement Platform Backend (SQLite, stdlib + Flask)
============================================================
A small Flask API behind the Runway front end (placement_predictor.html).

What it does
------------
* Student accounts (roll number + password) with bearer-token sessions
* Saves each student's profile, resume text, tests, portfolio, snapshots
* Application tracker (Wishlist -> Applied -> Interview -> Offer)
* TPO / admin console endpoints (protected by ADMIN_PASSWORD): cohort view,
  job postings that every student can see and match against
* Anonymous peer statistics
* Optional real-AI endpoints (chat, company insight, answer review) that call
  the Claude API server-side when ANTHROPIC_API_KEY is set

Uses only Python's built-in `sqlite3` (no SQLAlchemy) so there are no compiled
dependencies that locked-down Windows machines might block.

Quick start
-----------
    pip install -r requirements.txt
    python app.py

Environment variables (all optional)
------------------------------------
    ADMIN_PASSWORD     password for the TPO console (default: admin123 — CHANGE IT)
    ANTHROPIC_API_KEY  enables the real-AI endpoints
    ANTHROPIC_MODEL    defaults to claude-sonnet-5
    RUNWAY_DB          path of the SQLite file (default: placement.db next to app.py)
"""
import os
import json
import hmac
import secrets
import sqlite3
import requests
import subprocess
import sys
import threading
from datetime import datetime, timezone, timedelta
from functools import wraps
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from urllib.parse import urlparse

app = Flask(__name__)
CORS(app)

DB_PATH = os.environ.get('RUNWAY_DB') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'placement.db')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'admin123')
SESSION_DAYS = 30
HASH_METHOD = 'pbkdf2:sha256'

EDITABLE_FIELDS = [
    'name', 'cgpa', 'tenth', 'twelfth', 'aptitude', 'communication', 'programming',
    'projects', 'internships', 'certifications', 'backlogs', 'resume_text', 'ats_score',
    'prob', 'portfolio', 'tests'
]
APP_STAGES = ['Wishlist', 'Applied', 'Interview', 'Offer', 'Rejected']


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
    return db


@app.teardown_appcontext
def close_db(exception=None):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


def add_column_if_missing(conn, table, column, decl):
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS student (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_number TEXT UNIQUE NOT NULL,
            name TEXT,
            cgpa REAL DEFAULT 7.5,
            tenth REAL DEFAULT 82,
            twelfth REAL DEFAULT 78,
            aptitude REAL DEFAULT 62,
            communication INTEGER DEFAULT 6,
            programming INTEGER DEFAULT 6,
            projects INTEGER DEFAULT 2,
            internships INTEGER DEFAULT 0,
            certifications INTEGER DEFAULT 1,
            backlogs INTEGER DEFAULT 0,
            resume_text TEXT,
            ats_score INTEGER,
            prob REAL,
            updated_at TEXT
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS snapshot (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            date TEXT,
            prob REAL,
            FOREIGN KEY(student_id) REFERENCES student(id)
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS session (
            token TEXT PRIMARY KEY,
            student_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(student_id) REFERENCES student(id)
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS application (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            company TEXT NOT NULL,
            role TEXT,
            stage TEXT DEFAULT 'Wishlist',
            jd_text TEXT,
            match REAL,
            created_at TEXT,
            FOREIGN KEY(student_id) REFERENCES student(id)
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS job (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company TEXT NOT NULL,
            role TEXT NOT NULL,
            description TEXT,
            created_at TEXT
        )
    ''')
    add_column_if_missing(conn, 'student', 'prob', 'REAL')
    add_column_if_missing(conn, 'student', 'password_hash', 'TEXT')
    add_column_if_missing(conn, 'student', 'portfolio', 'TEXT')
    add_column_if_missing(conn, 'student', 'tests', 'TEXT')
    conn.commit()
    conn.close()


init_db()


def public_student(row):
    if row is None:
        return None
    d = dict(row)
    d.pop('password_hash', None)
    d.pop('session_created', None)
    d.pop('id', None)
    return d


def auth_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        header = request.headers.get('Authorization', '')
        token = header[7:].strip() if header.startswith('Bearer ') else ''
        if not token:
            return jsonify({'error': 'Authentication required'}), 401
        db = get_db()
        row = db.execute(
            'SELECT s.*, ss.created_at AS session_created FROM session ss '
            'JOIN student s ON s.id = ss.student_id WHERE ss.token=?', (token,)
        ).fetchone()
        if not row:
            return jsonify({'error': 'Invalid or expired session — please reconnect'}), 401
        try:
            created = datetime.fromisoformat(row['session_created'])
            if datetime.now(timezone.utc) - created > timedelta(days=SESSION_DAYS):
                db.execute('DELETE FROM session WHERE token=?', (token,))
                db.commit()
                return jsonify({'error': 'Session expired — please reconnect'}), 401
        except ValueError:
            pass
        g.student = row
        roll = kwargs.get('roll_number')
        if roll is not None and roll != row['roll_number']:
            return jsonify({'error': 'Forbidden'}), 403
        return fn(*args, **kwargs)
    return wrapper


def admin_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        supplied = request.headers.get('X-Admin-Password', '')
        if not hmac.compare_digest(supplied.encode(), ADMIN_PASSWORD.encode()):
            return jsonify({'error': 'Wrong or missing admin password'}), 401
        return fn(*args, **kwargs)
    return wrapper


def json_body():
    return request.get_json(force=True, silent=True) or {}


@app.route('/')
def index():
    return jsonify({
        'status': 'ok',
        'message': "This is the Runway backend API — there's nothing to see at this address directly. "
                   "Open placement_predictor.html in your browser instead; it talks to this server in the background.",
    })


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


@app.route('/api/auth/enter', methods=['POST'])
def auth_enter():
    body = json_body()
    roll = (body.get('roll_number') or '').strip()
    password = body.get('password') or ''
    if not roll:
        return jsonify({'error': 'roll_number is required'}), 400
    if len(password) < 4:
        return jsonify({'error': 'Password must be at least 4 characters'}), 400

    db = get_db()
    row = db.execute('SELECT * FROM student WHERE roll_number=?', (roll,)).fetchone()
    created = False
    if row is None:
        db.execute('INSERT INTO student (roll_number, name, password_hash, updated_at) VALUES (?,?,?,?)',
                   (roll, body.get('name'), generate_password_hash(password, method=HASH_METHOD), now_iso()))
        db.commit()
        created = True
    elif row['password_hash'] is None:
        db.execute('UPDATE student SET password_hash=? WHERE id=?',
                   (generate_password_hash(password, method=HASH_METHOD), row['id']))
        db.commit()
    elif not check_password_hash(row['password_hash'], password):
        return jsonify({'error': 'Wrong password for this roll number'}), 401

    row = db.execute('SELECT * FROM student WHERE roll_number=?', (roll,)).fetchone()
    token = secrets.token_hex(24)
    db.execute('INSERT INTO session (token, student_id, created_at) VALUES (?,?,?)', (token, row['id'], now_iso()))
    db.commit()
    return jsonify({'token': token, 'created': created, 'profile': public_student(row)})


@app.route('/api/auth/logout', methods=['POST'])
@auth_required
def auth_logout():
    header = request.headers.get('Authorization', '')
    token = header[7:].strip()
    db = get_db()
    db.execute('DELETE FROM session WHERE token=?', (token,))
    db.commit()
    return jsonify({'status': 'logged out'})


@app.route('/api/students/<roll_number>', methods=['GET'])
@auth_required
def get_student(roll_number):
    db = get_db()
    data = public_student(g.student)
    snaps = db.execute('SELECT date, prob FROM snapshot WHERE student_id=?', (g.student['id'],)).fetchall()
    data['snapshots'] = [dict(s) for s in snaps]
    return jsonify(data)


@app.route('/api/students/<roll_number>', methods=['PUT'])
@auth_required
def update_student(roll_number):
    db = get_db()
    body = json_body()
    updates = {}
    for k, v in body.items():
        if k not in EDITABLE_FIELDS:
            continue
        if k in ('portfolio', 'tests') and not isinstance(v, str) and v is not None:
            v = json.dumps(v)
        updates[k] = v
    if updates:
        set_clause = ', '.join(f'{k}=?' for k in updates.keys())
        values = list(updates.values()) + [now_iso(), g.student['id']]
        db.execute(f'UPDATE student SET {set_clause}, updated_at=? WHERE id=?', values)
        db.commit()
    row = db.execute('SELECT * FROM student WHERE id=?', (g.student['id'],)).fetchone()
    return jsonify(public_student(row))


@app.route('/api/students/<roll_number>/snapshots', methods=['GET'])
@auth_required
def list_snapshots(roll_number):
    snaps = get_db().execute('SELECT date, prob FROM snapshot WHERE student_id=?', (g.student['id'],)).fetchall()
    return jsonify([dict(s) for s in snaps])


@app.route('/api/students/<roll_number>/snapshots', methods=['POST'])
@auth_required
def add_snapshot(roll_number):
    db = get_db()
    body = json_body()
    date = body.get('date', datetime.now(timezone.utc).strftime('%Y-%m-%d'))
    prob = body.get('prob')
    db.execute('INSERT INTO snapshot (student_id, date, prob) VALUES (?,?,?)', (g.student['id'], date, prob))
    db.commit()
    return jsonify({'date': date, 'prob': prob}), 201


@app.route('/api/students/<roll_number>/snapshots', methods=['DELETE'])
@auth_required
def clear_snapshots(roll_number):
    db = get_db()
    db.execute('DELETE FROM snapshot WHERE student_id=?', (g.student['id'],))
    db.commit()
    return jsonify({'status': 'cleared'})


@app.route('/api/stats/<roll_number>', methods=['GET'])
@auth_required
def peer_stats(roll_number):
    db = get_db()
    my_prob = g.student['prob']
    all_probs = [r['prob'] for r in db.execute('SELECT prob FROM student WHERE prob IS NOT NULL').fetchall()]
    count = len(all_probs)
    if count == 0 or my_prob is None:
        return jsonify({'count': count, 'average': None, 'my_probability': my_prob, 'percentile': None})
    average = round(sum(all_probs) / count, 1)
    below_me = len([p for p in all_probs if p <= my_prob])
    return jsonify({
        'count': count, 'average': average, 'my_probability': my_prob,
        'percentile': round((below_me / count) * 100),
    })


APP_COLS = 'id, company, role, stage, jd_text, match, created_at'


@app.route('/api/students/<roll_number>/applications', methods=['GET'])
@auth_required
def list_applications(roll_number):
    rows = get_db().execute(
        f'SELECT {APP_COLS} FROM application WHERE student_id=? ORDER BY id', (g.student['id'],)).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route('/api/students/<roll_number>/applications', methods=['POST'])
@auth_required
def add_application(roll_number):
    body = json_body()
    company = (body.get('company') or '').strip()
    if not company:
        return jsonify({'error': 'company is required'}), 400
    stage = body.get('stage') if body.get('stage') in APP_STAGES else 'Wishlist'
    db = get_db()
    cur = db.execute(
        'INSERT INTO application (student_id, company, role, stage, jd_text, match, created_at) VALUES (?,?,?,?,?,?,?)',
        (g.student['id'], company, (body.get('role') or '').strip(), stage, body.get('jd_text'), body.get('match'), now_iso()))
    db.commit()
    row = db.execute(f'SELECT {APP_COLS} FROM application WHERE id=?', (cur.lastrowid,)).fetchone()
    return jsonify(dict(row)), 201


@app.route('/api/students/<roll_number>/applications/<int:app_id>', methods=['PUT'])
@auth_required
def update_application(roll_number, app_id):
    db = get_db()
    row = db.execute('SELECT id FROM application WHERE id=? AND student_id=?', (app_id, g.student['id'])).fetchone()
    if not row:
        return jsonify({'error': 'not found'}), 404
    body = json_body()
    updates = {}
    if body.get('stage') in APP_STAGES:
        updates['stage'] = body['stage']
    for k in ('company', 'role'):
        if k in body and isinstance(body[k], str):
            updates[k] = body[k].strip()
    if updates:
        set_clause = ', '.join(f'{k}=?' for k in updates)
        db.execute(f'UPDATE application SET {set_clause} WHERE id=?', list(updates.values()) + [app_id])
        db.commit()
    row = db.execute(f'SELECT {APP_COLS} FROM application WHERE id=?', (app_id,)).fetchone()
    return jsonify(dict(row))


@app.route('/api/students/<roll_number>/applications/<int:app_id>', methods=['DELETE'])
@auth_required
def delete_application(roll_number, app_id):
    db = get_db()
    db.execute('DELETE FROM application WHERE id=? AND student_id=?', (app_id, g.student['id']))
    db.commit()
    return jsonify({'status': 'deleted'})


JOB_COLS = 'id, company, role, description, created_at'


@app.route('/api/jobs', methods=['GET'])
@auth_required
def list_jobs():
    rows = get_db().execute(f'SELECT {JOB_COLS} FROM job ORDER BY id DESC').fetchall()
    return jsonify([dict(r) for r in rows])


@app.route('/api/admin/students', methods=['GET'])
@admin_required
def admin_students():
    rows = get_db().execute(
        'SELECT roll_number, name, cgpa, programming, aptitude, projects, internships, certifications, '
        'backlogs, ats_score, prob, updated_at FROM student ORDER BY roll_number').fetchall()
    return jsonify([dict(r) for r in rows])


@app.route('/api/admin/jobs', methods=['GET'])
@admin_required
def admin_list_jobs():
    rows = get_db().execute(f'SELECT {JOB_COLS} FROM job ORDER BY id DESC').fetchall()
    return jsonify([dict(r) for r in rows])


@app.route('/api/admin/jobs', methods=['POST'])
@admin_required
def admin_add_job():
    body = json_body()
    company = (body.get('company') or '').strip()
    role = (body.get('role') or '').strip()
    if not company or not role:
        return jsonify({'error': 'company and role are required'}), 400
    db = get_db()
    cur = db.execute('INSERT INTO job (company, role, description, created_at) VALUES (?,?,?,?)',
                     (company, role, body.get('description') or '', now_iso()))
    db.commit()
    row = db.execute(f'SELECT {JOB_COLS} FROM job WHERE id=?', (cur.lastrowid,)).fetchone()
    return jsonify(dict(row)), 201


@app.route('/api/admin/jobs/<int:job_id>', methods=['DELETE'])
@admin_required
def admin_delete_job(job_id):
    db = get_db()
    db.execute('DELETE FROM job WHERE id=?', (job_id,))
    db.commit()
    return jsonify({'status': 'deleted'})


ANTHROPIC_API_URL = 'https://api.anthropic.com/v1/messages'
ANTHROPIC_MODEL = os.environ.get('ANTHROPIC_MODEL', 'claude-sonnet-5')


class NoApiKey(Exception):
    pass


def call_claude(system_prompt, user_message, max_tokens=300):
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise NoApiKey()
    resp = requests.post(
        ANTHROPIC_API_URL,
        headers={'x-api-key': api_key, 'anthropic-version': '2023-06-01', 'content-type': 'application/json'},
        json={
            'model': ANTHROPIC_MODEL,
            'max_tokens': max_tokens,
            'system': system_prompt,
            'messages': [{'role': 'user', 'content': user_message}],
        },
        timeout=25,
    )
    resp.raise_for_status()
    data = resp.json()
    text = ''.join(b.get('text', '') for b in data.get('content', []) if b.get('type') == 'text')
    return text.strip() or "Sorry, I couldn't generate a reply."


def ai_response(system_prompt, user_message, max_tokens):
    try:
        return jsonify({'reply': call_claude(system_prompt, user_message, max_tokens)})
    except NoApiKey:
        return jsonify({'error': 'ANTHROPIC_API_KEY is not set on this server. '
                                 'See backend/README.md, then restart the server.'}), 400
    except requests.exceptions.RequestException as e:
        return jsonify({'error': 'Could not reach the Claude API: ' + str(e)}), 502


def clip(value, limit):
    return (value or '').strip()[:limit]


@app.route('/api/chat', methods=['POST'])
@auth_required
def ai_chat():
    body = json_body()
    message = clip(body.get('message'), 1500)
    if not message:
        return jsonify({'error': 'message is required'}), 400
    system_prompt = (
        "You are the AI Career Assistant embedded in a student's placement-readiness dashboard called Runway. "
        "Answer using ONLY the student data given below — don't invent numbers that aren't there. Keep answers "
        "short (2-4 sentences), specific, practical, and encouraging. If the question has nothing to do with "
        "placements, careers, resumes, skills, or interviews, politely redirect back to those topics.\n\n"
        "Student context (JSON):\n" + json.dumps(body.get('context') or {})[:4000]
    )
    return ai_response(system_prompt, message, 300)


@app.route('/api/company-insight', methods=['POST'])
@auth_required
def company_insight():
    body = json_body()
    company = clip(body.get('company'), 80)
    if not company:
        return jsonify({'error': 'company is required'}), 400
    system_prompt = (
        "You help Indian engineering students prepare for campus recruitment. The student names a company. "
        "Give 5-6 short bullet points: what the company is generally known for, the kind of roles freshers "
        "usually get, typical selection stages, and what to focus on. Use only general, widely known "
        "information. If you are not sure about specifics (current cutoffs, package figures, exact rounds), say "
        "so instead of guessing, and tell the student to confirm with the placement cell or the official careers "
        "page. Finish with one line tailoring the advice to the student's weak areas.\n\n"
        "Student context (JSON):\n" + json.dumps(body.get('context') or {})[:3000]
    )
    return ai_response(system_prompt, "Company: " + company, 550)


@app.route('/api/evaluate-answer', methods=['POST'])
@auth_required
def evaluate_answer():
    body = json_body()
    question = clip(body.get('question'), 500)
    answer = clip(body.get('answer'), 3000)
    if not question or not answer:
        return jsonify({'error': 'question and answer are required'}), 400
    system_prompt = (
        "You are a friendly interview coach for a fresher. Review the candidate's answer to the interview "
        "question. Reply in under 130 words with: a score out of 10, two strengths, two specific improvements, "
        "and one improved opening sentence they could use. Be honest, not flattering."
    )
    return ai_response(system_prompt, f"Question: {question}\n\nCandidate's answer: {answer}", 450)


# ---------------------------------------------------------------------
# Trained ML model (created by ml/train_model.py)
# ---------------------------------------------------------------------
ML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ml', 'output')
ML_ROOT = os.path.dirname(os.path.abspath(ML_DIR))
ML_SCRIPT = os.path.join(ML_ROOT, 'train_model.py')
ML_CSV = os.path.join(ML_ROOT, 'placement_data.csv')
_ml_bundle = None
_ml_train_lock = threading.Lock()
_ml_train_state = {'running': False, 'success': None, 'message': 'No training run yet.', 'log': ''}


def run_ml_training():
    """Run the checked-in training pipeline and retain its console output for the ML Lab."""
    global _ml_bundle
    try:
        process = subprocess.Popen(
            [sys.executable, '-u', ML_SCRIPT, ML_CSV], cwd=ML_ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', bufsize=1)
        timeout_timer = threading.Timer(
            300, lambda: process.kill() if process.poll() is None else None)
        timeout_timer.daemon = True
        timeout_timer.start()
        with process.stdout:
            for line in process.stdout:
                with _ml_train_lock:
                    _ml_train_state['log'] += line
                    if line.strip():
                        _ml_train_state['message'] = line.strip()
        code = process.wait()
        timeout_timer.cancel()
        with _ml_train_lock:
            _ml_train_state['running'] = False
            _ml_train_state['success'] = code == 0
            _ml_train_state['message'] = 'Training complete.' if code == 0 else 'Training failed; see the log.'
            if code == 0:
                _ml_bundle = None
    except Exception as exc:
        with _ml_train_lock:
            _ml_train_state['running'] = False
            _ml_train_state['success'] = False
            _ml_train_state['message'] = 'Training failed: ' + str(exc)
            _ml_train_state['log'] += '\n' + str(exc)


def local_training_request():
    """Training writes model artifacts, so allow it only from this computer."""
    if request.remote_addr not in ('127.0.0.1', '::1'):
        return False
    origin = request.headers.get('Origin', '')
    if not origin or origin == 'null':  # local file:// frontend or a local command-line request
        return True
    return urlparse(origin).hostname in ('localhost', '127.0.0.1', '::1')


@app.route('/api/ml-train', methods=['POST'])
def ml_train_start():
    if not local_training_request():
        return jsonify({'error': 'Model training is available only from the local computer.'}), 403
    if not os.path.isfile(ML_SCRIPT) or not os.path.isfile(ML_CSV):
        return jsonify({'error': 'Training script or dataset is missing.'}), 503
    with _ml_train_lock:
        if _ml_train_state['running']:
            return jsonify({'running': True, 'message': _ml_train_state['message']}), 409
        _ml_train_state.update(running=True, success=None, message='Starting training…', log='')
    threading.Thread(target=run_ml_training, daemon=True).start()
    return jsonify({'running': True, 'message': 'Training started from ml/placement_data.csv.'}), 202


@app.route('/api/ml-train-status', methods=['GET'])
def ml_train_status():
    if not local_training_request():
        return jsonify({'error': 'Training status is available only from the local computer.'}), 403
    with _ml_train_lock:
        return jsonify(dict(_ml_train_state))


def load_ml():
    """Load the trained model once. If the file is missing or was saved with a different
    scikit-learn version, retrain automatically from ml/placement_data.csv (takes ~20 s)."""
    global _ml_bundle
    if _ml_bundle is not None:
        return _ml_bundle
    import joblib
    path = os.path.join(ML_DIR, 'model.joblib')
    try:
        _ml_bundle = joblib.load(path)
        return _ml_bundle
    except Exception:
        pass
    csv = os.path.join(ML_DIR, '..', 'placement_data.csv')
    script = os.path.join(ML_DIR, '..', 'train_model.py')
    if os.path.exists(csv) and os.path.exists(script):
        import subprocess, sys
        try:
            subprocess.run([sys.executable, script, csv], cwd=os.path.dirname(script),
                           timeout=300, check=True, capture_output=True)
            _ml_bundle = joblib.load(path)
        except Exception:
            _ml_bundle = None
    return _ml_bundle


def ml_features(bundle, p):
    """Map the app's profile fields onto the features the model was trained on."""
    def num(k, default=0.0):
        try:
            return float(p.get(k, default))
        except (TypeError, ValueError):
            return float(default)
    if bundle['kind'] == 'kaggle':
        row = {
            'ssc_p': num('tenth'), 'hsc_p': num('twelfth'),
            'degree_p': num('cgpa') * 10,          # approximate CGPA(10) -> percentage
            'etest_p': num('aptitude'),
            'workex': 1 if num('internships') > 0 else 0,
        }
    else:
        row = {f: num(f) for f in bundle['features']}
    return [[row[f] for f in bundle['features']]], row


@app.route('/api/ml-info', methods=['GET'])
def ml_info():
    path = os.path.join(ML_DIR, 'metrics.json')
    if not os.path.exists(path):
        return jsonify({'trained': False, 'message': 'Run ml/train_model.py first.'}), 200
    with open(path) as f:
        info = json.load(f)
    info['trained'] = True
    return jsonify(info)


@app.route('/api/ml-predict', methods=['POST'])
def ml_predict():
    bundle = load_ml()
    if bundle is None:
        return jsonify({'error': 'Model not available. Run ml/train_model.py first.'}), 503
    body = request.get_json(force=True) or {}
    import pandas as pd
    vec, row = ml_features(bundle, body)
    cols = bundle['features']
    frame = pd.DataFrame(vec, columns=cols)
    proba = float(bundle['model'].predict_proba(frame)[0][1])
    result = {'probability': round(proba * 100, 1), 'model': bundle['name'],
              'data_source': bundle.get('data_source', 'real'), 'inputs_used': row}

    # "Why this prediction?" - set each feature to its training median, one at a time,
    # and see how much the probability changes (a simple local explanation).
    meds = bundle.get('medians')
    if meds:
        factors = []
        for c in cols:
            alt = frame.copy()
            alt[c] = meds[c]
            delta = proba - float(bundle['model'].predict_proba(alt)[0][1])
            factors.append({'feature': c, 'value': row[c], 'median': round(float(meds[c]), 2),
                            'effect': round(delta * 100, 1)})
        factors.sort(key=lambda f: abs(f['effect']), reverse=True)
        result['factors'] = factors

    sal_path = os.path.join(ML_DIR, 'salary_model.joblib')
    if os.path.exists(sal_path):                      # optional regression model
        try:
            import joblib
            sal = joblib.load(sal_path)
            result['expected_salary'] = round(float(sal['model'].predict(frame)[0]), 2)
            result['salary_model'] = sal['name']
            info_path = os.path.join(ML_DIR, 'metrics.json')
            if os.path.exists(info_path):
                with open(info_path) as f:
                    result['salary_unit'] = (json.load(f).get('salary') or {}).get('unit', '')
        except Exception:
            pass
    return jsonify(result)


@app.route('/api/ml-image/<name>', methods=['GET'])
def ml_image(name):
    """Serve the graphs created by train_model.py (whitelisted names only)."""
    allowed = {'confusion_matrix.png', 'roc_curves.png', 'feature_importance.png',
               'eda_correlation.png', 'eda_class_balance.png', 'eda_distributions.png'}
    if name not in allowed:
        return jsonify({'error': 'not found'}), 404
    from flask import send_from_directory
    return send_from_directory(os.path.abspath(ML_DIR), name)


if __name__ == '__main__':
    if ADMIN_PASSWORD == 'admin123':
        print('NOTE: using the default ADMIN_PASSWORD "admin123" — set the ADMIN_PASSWORD environment variable to change it.')
    app.run(debug=True, port=5000)
