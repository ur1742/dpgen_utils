from flask import Flask, render_template, request, jsonify, send_file
import os
from datetime import datetime
import numpy as np
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Всё в одном файле — чтобы не было путаницы
from utils import (
    get_current_hash_dir,
    parse_dpgen_log,
    get_iterations,
    analyze_fp_tasks,
    analyze_model_devi_progress,
    get_current_stage_from_record,
    generate_train_plot,
    find_active_train_dir,
    parse_train_log
)

app = Flask(__name__)
app.config['WORK_DIR'] = None


def generate_all_train_plots():
    """Генерирует все графики обучения при запуске"""
    work_dir = app.config.get('WORK_DIR')
    if not work_dir or not os.path.exists(work_dir):
        return
    
    print("Generating training plots...")
    
    # Создаем директорию для графиков
    static_train_dir = os.path.join('static', 'train_plots')
    os.makedirs(static_train_dir, exist_ok=True)
    
    iterations = get_iterations(work_dir)
    for iteration in iterations:
        train_dir = os.path.join(work_dir, iteration, '00.train')
        if not os.path.exists(train_dir):
            continue
            
        for item in sorted(os.listdir(train_dir)):
            if item.isdigit() and len(item) == 3 and os.path.isdir(os.path.join(train_dir, item)):
                task_dir = os.path.join(train_dir, item)
                lcurve_path = os.path.join(task_dir, 'lcurve.out')
                
                if os.path.exists(lcurve_path):
                    # Простое имя файла: iter_000000_task_000.png
                    image_filename = f"{iteration.replace('.', '_')}_task_{item}.png"
                    image_path = os.path.join(static_train_dir, image_filename)
                    
                    # Генерируем график, если его еще нет
                    if not os.path.exists(image_path):
                        try:
                            print(f"Generating plot for {iteration}/task.{item}")
                            data = np.genfromtxt(lcurve_path, names=True)
                            if data.size > 0:
                                plt.figure(figsize=(8, 6))
                                for name in data.dtype.names[1:-1]:
                                    if len(data[name]) > 1:
                                        plt.plot(data['step'][1:], data[name][1:], label=name)
                                plt.legend()
                                plt.xlabel('Step')
                                plt.ylabel('Loss')
                                plt.xscale('symlog')
                                plt.yscale('log')
                                plt.grid(True, alpha=0.3)
                                plt.title(f"{iteration} - Task {item}")

                                plt.savefig(image_path, format='png', dpi=100, bbox_inches='tight')
                                plt.close()
                                print(f"Saved: {image_filename}")
                        except Exception as e:
                            print(f"Error generating plot for {iteration}/task.{item}: {e}")


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        work_dir = request.form.get('work_dir')
        if os.path.isdir(work_dir):
            app.config['WORK_DIR'] = work_dir
            # Генерируем графики при установке новой директории
            generate_all_train_plots()
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
        # Генерируем графики при установке новой директории
        generate_all_train_plots()
        return jsonify({"status": "success", "dir": work_dir})
    else:
        return jsonify({"status": "error", "message": "Invalid directory"}), 400


@app.route('/train_plot')
def train_plot():
    work_dir = app.config['WORK_DIR']
    if not work_dir:
        return "No work directory set", 400

    current_hash = get_current_hash_dir(work_dir)
    current_stage = get_current_stage_from_record(work_dir)

    if not current_hash or not current_stage:
        return "No active task", 404

    if current_stage["stage_name"] != "run_train":
        return "Not in run_train stage", 404

    active_train_dir = find_active_train_dir(current_hash["hash_dir"])
    if not active_train_dir:
        return "No active training directory found", 404

    lcurve_path = os.path.join(active_train_dir, 'lcurve.out')
    if not os.path.exists(lcurve_path):
        return "lcurve.out not found", 404

    try:
        data = np.genfromtxt(lcurve_path, names=True)
        if data.size == 0:
            return "Empty lcurve.out", 404

        plt.figure(figsize=(10, 6))
        for name in data.dtype.names[1:-1]:
            plt.plot(data['step'][1:], data[name][1:], label=name)
        plt.legend()
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.xscale('symlog')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.title("Training Loss Curve (Live)")

        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        plt.close()

        buffer.seek(0)
        return buffer.getvalue(), 200, {'Content-Type': 'image/png'}
    except Exception as e:
        print(f"Error generating plot: {e}")
        return "Error generating plot", 500


@app.route('/cached_train_plot/<filename>')
def cached_train_plot(filename):
    """Отдает кэшированный график"""
    static_train_dir = os.path.join('static', 'train_plots')
    image_path = os.path.join(static_train_dir, filename)
    
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/png')
    
    return "Plot not found", 404


@app.route('/train_history')
def train_history():
    work_dir = app.config['WORK_DIR']
    if not work_dir:
        return "No work directory set", 400
    
    try:
        train_data = collect_train_history(work_dir)
        return render_template('train_history.html', work_dir=work_dir, train_data=train_data)
    except Exception as e:
        return f"Error: {str(e)}", 500


def collect_train_history(work_dir):
    """Собирает данные о задачах обучения с простыми именами файлов"""
    iterations = get_iterations(work_dir)
    train_history_data = []
    
    for iteration in iterations:
        train_dir = os.path.join(work_dir, iteration, '00.train')
        if not os.path.exists(train_dir):
            continue
            
        # Ищем поддиректории с задачами обучения
        train_tasks = []
        for item in sorted(os.listdir(train_dir)):
            if item.isdigit() and len(item) == 3 and os.path.isdir(os.path.join(train_dir, item)):
                task_dir = os.path.join(train_dir, item)
                lcurve_path = os.path.join(task_dir, 'lcurve.out')
                
                if os.path.exists(lcurve_path):
                    try:
                        # Получаем дату последнего изменения файла
                        mtime = os.path.getmtime(lcurve_path)
                        finish_time = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                        
                        # Пытаемся получить количество точек данных
                        try:
                            data = np.genfromtxt(lcurve_path, names=True)
                            data_points = len(data) if data.size > 0 else 'N/A'
                        except:
                            data_points = 'N/A'
                        
                        # Простое имя файла: iter_000000_task_000.png
                        image_filename = f"{iteration.replace('.', '_')}_task_{item}.png"
                        
                        # Сохраняем информацию для отображения
                        train_tasks.append({
                            'task_id': item,
                            'lcurve_path': lcurve_path,
                            'data_points': data_points,
                            'finish_time': finish_time,
                            'image_filename': image_filename,
                            'iteration': iteration
                        })
                    except Exception as e:
                        print(f"Error processing task {iteration}/{item}: {e}")
        
        if train_tasks:
            train_history_data.append({
                'iteration': iteration,
                'tasks': train_tasks
            })
    
    return train_history_data


@app.route('/current_status')
def current_status():
    work_dir = app.config['WORK_DIR']
    if not work_dir:
        return jsonify({"error": "No work directory set"}), 400

    try:
        data = collect_dpgen_status(work_dir)
        # Возвращаем только данные текущей задачи
        current_task_data = {
            'current_hash': data.get('current_hash'),
            'current_stage': data.get('current_stage'),
            'fp_analysis': data.get('fp_analysis'),
            'show_train_plot': data.get('show_train_plot'),
            'active_train_subdir': data.get('active_train_subdir'),
            'loss_data': data.get('loss_data', [])[-10:] if data.get('loss_data') else [],  # Последние 10 значений
            'train_log_info': data.get('train_log_info')  # Добавляем информацию из train.log
        }
        return jsonify(current_task_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def collect_dpgen_status(work_dir):
    log_path = os.path.join(work_dir, 'dpgen.log')
    iterations = get_iterations(work_dir)
    loss_data = parse_dpgen_log(log_path)

    current_stage = get_current_stage_from_record(work_dir)
    current_hash = get_current_hash_dir(work_dir)

    fp_analysis = None
    show_train_plot = False
    train_log_info = None

    if current_hash and current_stage:
        stage_name = current_stage["stage_name"]
        if stage_name == "run_train":
            show_train_plot = True
            # Читаем train.log только для стадии run_train и из хэш директории
            active_train_dir = find_active_train_dir(current_hash["hash_dir"])
            if active_train_dir:
                train_log_path = os.path.join(active_train_dir, 'train.log')
                train_log_info = parse_train_log(train_log_path)
        elif stage_name == "run_fp":
            fp_analysis = analyze_fp_tasks(current_hash["hash_dir"])

    model_devi_data = analyze_model_devi_progress(work_dir)

    active_dir = find_active_train_dir(current_hash["hash_dir"]) if current_hash else None
    active_subdir = os.path.basename(active_dir) if active_dir else None

    return {
        'loss_data': loss_data[-10:],
        'iterations': iterations,
        'work_dir': work_dir,
        'current_hash': current_hash,
        'current_stage': current_stage,
        'fp_analysis': fp_analysis,
        'show_train_plot': show_train_plot,
        'active_train_subdir': active_subdir,
        'model_devi_data': model_devi_data,
        'train_log_info': train_log_info  # Добавляем информацию из train.log
    }


if __name__ == '__main__':
    # Создаём папки
    hist_dir = os.path.join('static', 'model_devi')
    os.makedirs(hist_dir, exist_ok=True)
    
    train_plots_dir = os.path.join('static', 'train_plots')
    os.makedirs(train_plots_dir, exist_ok=True)
    
    app.run(host='127.0.0.1', port=5000, debug=True)
