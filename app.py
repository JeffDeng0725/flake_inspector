from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename
from celery import Celery
from image_processing import process_image  # Import the process_image function

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Setup Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Celery configuration
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

def make_celery(app):
    celery = Celery(
        app.import_name,
        broker=app.config['CELERY_BROKER_URL'],
        backend=app.config['CELERY_RESULT_BACKEND']
    )
    celery.conf.update(app.config)
    return celery

celery = make_celery(app)

# Temporary storage for users and scan results
users = {}
scans = []

# Directory for uploaded files
UPLOAD_FOLDER = 'uploads/'
PROCESSED_FOLDER = 'processed/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            user = User(username)
            login_user(user)
            return redirect(url_for('options'))
        flash('Invalid username or password')
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users:
            flash('Username already exists')
            return redirect(url_for('register'))
        users[username] = password
        return redirect(url_for('index'))
    return render_template('register.html')

@app.route('/options')
@login_required
def options():
    return render_template('options.html')

@celery.task
def process_image_task(file_path, output_dir, threshold_range, num_outputs, channel):
    return process_image(file_path, output_dir, threshold_range, num_outputs, channel)

@app.route('/process', methods=['GET', 'POST'])
@login_required
def process():
    if request.method == 'POST':
        if 'files' not in request.files:
            flash('No files part')
            return redirect(request.url)
        files = request.files.getlist('files')
        low_threshold = int(request.form['low_threshold'])
        high_threshold = int(request.form['high_threshold'])
        num_outputs = int(request.form['num_outputs'])
        channel = request.form['channel']
        threshold_range = [low_threshold, high_threshold]
        upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        scan_id = str(uuid.uuid4())
        processed_files = []
        
        total_files = len(files)
        for index, file in enumerate(files):
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file:
                original_filename = secure_filename(file.filename)
                unique_filename = f"{scan_id}_{uuid.uuid4().hex}{os.path.splitext(original_filename)[1]}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(file_path)
                # 使用 Celery 任务来处理图像
                result = process_image_task.delay(file_path, app.config['PROCESSED_FOLDER'], threshold_range, num_outputs, channel)
                processed_files.append({
                    'original': unique_filename,
                    'task_id': result.id  # 存储任务ID以便之后跟踪任务状态
                })

        scans.append({
            'upload_time': upload_time,
            'scan_id': scan_id,
            'processed_files': processed_files
        })
        return jsonify({ 'message': 'Files processing started' }), 200
    return render_template('process.html')

@app.route('/review')
@login_required
def review():
    return render_template('review.html', scans=scans)

@app.route('/details/<scan_id>')
@login_required
def details(scan_id):
    processed_scan = next((scan for scan in scans if scan['scan_id'] == scan_id), None)
    if not processed_scan:
        return "No files found", 404

    # Prepare the file details to be sent to the template
    file_details = processed_scan['processed_files']

    unique_files = {file['original'] for file in file_details}
    unique_files = [(original, [file for file in file_details if file['original'] == original]) for original in unique_files]

    return render_template('detail_view.html', scan_id=scan_id, file_details=file_details, unique_files=unique_files)

@app.route('/processed/<filename>')
@login_required
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=5002)
