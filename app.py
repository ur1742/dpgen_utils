# app.py
from flask import Flask, render_template, request, jsonify
import os
from datetime import datetime

# Всё в одном файле — чтобы не было путаницы
from utils import (
    get_current_hash_dir,
    parse_dpgen_log,
    get_iterations,
    analyze_fp_tasks,
    analyze_model_devi_progress,
    get_current_stage_from_record,
    generate_train_plot
)

app = Flask(__name__)
app.config['WORK_DIR'] = None


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        work_dir = request.form.get('work_dir')
        if os.path.isdir(work_dir):
            app.config['WORK_DIR'] = work_dir
        else:
            return render_template('index.html', error="Directory does not exist!")

    work_dir = app.config['WORK_DIR']
    data = {}
    if work_dir:
        data = collect_dpgen_status(work_dir)

    return render_template('index.html', work_dir=work_dir, data=data)


@app.route('/set_dir', methods=['POST'])
def set_dir():
    work_dir = request.json.get('dir')
    if os.path.isdir(work_dir):
        app.config['WORK_DIR'] = work_dir
        return jsonify({"status": "success", "dir": work_dir})
    else:
        return jsonify({"status": "error", "message": "Invalid directory"}), 400


def collect_dpgen_status(work_dir):
    log_path = os.path.join(work_dir, 'dpgen.log')
    iterations = get_iterations(work_dir)
    loss_data = parse_dpgen_log(log_path)

    current_stage = get_current_stage_from_record(work_dir)
    current_hash = get_current_hash_dir(work_dir)

    fp_analysis = None
    train_plot = None

    if current_hash and current_stage:
        stage_name = current_stage["stage_name"]
        if stage_name == "run_train":
            train_plot = generate_train_plot(current_hash["hash_dir"])
        elif stage_name == "run_fp":
            fp_analysis = analyze_fp_tasks(current_hash["hash_dir"])

    model_devi_data = analyze_model_devi_progress(work_dir)

    return {
        'loss_data': loss_data[-10:],
        'iterations': iterations,
        'work_dir': work_dir,
        'current_hash': current_hash,
        'current_stage': current_stage,
        'fp_analysis': fp_analysis,
        'train_plot': train_plot,
        'model_devi_data': model_devi_data
    }


if __name__ == '__main__':
    # Создаём папку для гистограмм
    hist_dir = os.path.join('static', 'model_devi')
    os.makedirs(hist_dir, exist_ok=True)
    app.run(host='127.0.0.1', port=5000, debug=True)
