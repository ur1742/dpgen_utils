import os
import re
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


# --- Путь для гистограмм ---
HIST_DIR = os.path.join('static', 'model_devi')
os.makedirs(HIST_DIR, exist_ok=True)


# --- Вспомогательные функции ---

def get_iterations(work_dir):
    return sorted([d for d in os.listdir(work_dir) if d.startswith('iter.') and os.path.isdir(os.path.join(work_dir, d))])


def parse_dpgen_log(log_path):
    if not os.path.exists(log_path):
        return []
    loss_data = []
    with open(log_path, 'r') as f:
        for line in f:
            if 'loss:' in line and 'valid' in line:
                match = re.search(r'loss:\s*([0-9.e-]+)', line)
                if match:
                    try:
                        loss = float(match.group(1))
                        loss_data.append(loss)
                    except:
                        pass
    return loss_data


def get_current_hash_dir(work_dir):
    hash_dirs = []
    for d in os.listdir(work_dir):
        dir_path = os.path.join(work_dir, d)
        if os.path.isdir(dir_path) and len(d) == 40 and d.isalnum():
            try:
                stat = os.stat(dir_path)
                hash_dirs.append((dir_path, stat.st_mtime))
            except:
                continue
    if not hash_dirs:
        return None
    latest_dir, mtime = max(hash_dirs, key=lambda x: x[1])
    return {
        "hash_dir": latest_dir,
        "hash_name": os.path.basename(latest_dir),
        "mtime": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    }


def parse_train_log(train_log_path):
    """
    Парсит train.log для получения информации о шагах и ETA
    """
    if not os.path.exists(train_log_path):
        return None
    
    try:
        # Ищем последнюю строку с ETA
        last_eta_line = None
        with open(train_log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if 'DEEPMD INFO' in line and 'eta =' in line and 'batch' in line:
                    last_eta_line = line.strip()
        
        if not last_eta_line:
            return None
        
        # Парсим строку вида: [2025-08-30 22:27:03,177] DEEPMD INFO    batch     900: total wall time = 17.90 s, eta = 6:54:56
        # Извлекаем информацию
        timestamp_match = re.search(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),', last_eta_line)
        batch_match = re.search(r'batch\s+(\d+):', last_eta_line)
        eta_match = re.search(r'eta = (\d+):(\d+):(\d+)', last_eta_line)
        
        if not (timestamp_match and batch_match and eta_match):
            return None
        
        # Получаем текущее время из лога
        log_time_str = timestamp_match.group(1)
        log_time = datetime.strptime(log_time_str, '%Y-%m-%d %H:%M:%S')
        
        # Номер шага
        batch_num = int(batch_match.group(1))
        
        # ETA в часах:минутах:секундах
        eta_hours = int(eta_match.group(1))
        eta_minutes = int(eta_match.group(2))
        eta_seconds = int(eta_match.group(3))
        
        # Вычисляем предполагаемое время завершения
        eta_delta = timedelta(hours=eta_hours, minutes=eta_minutes, seconds=eta_seconds)
        finish_time = log_time + eta_delta
        
        return {
            'batch': batch_num,
            'eta': f"{eta_hours}:{eta_minutes:02d}:{eta_seconds:02d}",
            'log_time': log_time.strftime('%Y-%m-%d %H:%M:%S'),
            'finish_time': finish_time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
    except Exception as e:
        print(f"Error parsing train.log: {e}")
        return None


# --- Стадия из record.dpgen ---

def get_current_stage_from_record(work_dir):
    record_path = os.path.join(work_dir, 'record.dpgen')
    if not os.path.exists(record_path):
        return {
            "iter": "N/A",
            "stage_id": "N/A",
            "stage_name": "record.dpgen not found",
            "stage_label": "Unknown (no record)"
        }

    try:
        with open(record_path, 'r') as f:
            lines = f.readlines()
        if not lines:
            return {"stage_label": "Empty record"}
        last_line = lines[-1].strip().split()
        if len(last_line) != 2:
            return {"stage_label": "Invalid format"}

        iter_idx = int(last_line[0])
        stage_id = int(last_line[1])+1

        STAGE_MAP = {
            0: ("make_train", "Prepare Training"),
            1: ("run_train", "Run Training"),
            2: ("post_train", "Collect Training Results"),
            3: ("make_model_devi", "Prepare Model Deviation"),
            4: ("run_model_devi", "Run Model Deviation"),
            5: ("post_model_devi", "Analyze Model Deviation"),
            6: ("make_fp", "Prepare FP Inputs"),
            7: ("run_fp", "Run FP Tasks"),
            8: ("post_fp", "Analyze FP Results")
        }

        stage_name, stage_label = STAGE_MAP.get(stage_id, (f"unknown_{stage_id}", "Unknown Stage"))

        return {
            "iter": f"iter.{iter_idx:06d}",
            "stage_id": stage_id,
            "stage_name": stage_name,
            "stage_label": stage_label
        }
    except Exception as e:
        return {"stage_label": f"Error: {str(e)}"}


# --- Анализ FP-задач ---

def format_time(seconds):
    if seconds != seconds or seconds < 0:
        return "неизвестно"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours} ч {minutes} мин {secs} с"
    else:
        return f"{minutes} мин {secs} с"


def parse_time_from_string(time_str):
    match = re.match(r"(\d+)m(\d+(?:\.\d+)?)s", time_str)
    if match:
        minutes = float(match.group(1))
        seconds = float(match.group(2))
        return minutes * 60 + seconds
    return None


def analyze_fp_tasks(base_dir):
    tasks = sorted([
        d for d in os.listdir(base_dir)
        if d.startswith("task.000.") and os.path.isdir(os.path.join(base_dir, d))
    ])

    if not tasks:
        return {"error": "No FP tasks found"}

    output_files = ["pw.out", "output", "stdout", "qe.out"]
    finished_keyword = "finished"
    fail_flag = "flag_if_job_task_fail"

    finished = []
    failed = []
    running = []
    pending = []
    durations = []

    for task in tasks:
        task_path = os.path.join(base_dir, task)
        try:
            files = os.listdir(task_path)
        except:
            files = []

        has_fail = any(fail_flag in f for f in files)
        has_finish = any(finished_keyword in f for f in files)

        out_file = None
        content = ""
        for fname in output_files:
            fpath = os.path.join(task_path, fname)
            if os.path.isfile(fpath):
                out_file = fpath
                try:
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    break
                except:
                    continue

        has_output = bool(out_file)

        if has_fail:
            failed.append(task)
        elif has_finish:
            finished.append(task)
            if has_output:
                pwscf_line = re.search(r"PWSCF\s*:\s*\S+\s+CPU\s+\S+\s+WALL", content)
                if pwscf_line:
                    parts = pwscf_line.group(0).split()
                    try:
                        cpu_idx = parts.index("CPU")
                        cpu_time_str = parts[cpu_idx - 1]
                        cpu_seconds = parse_time_from_string(cpu_time_str)
                        if cpu_seconds is not None:
                            durations.append(cpu_seconds)
                    except:
                        pass
        elif has_output:
            running.append(task)
        else:
            pending.append(task)

    total = len(tasks)
    done = len(finished)
    remaining = len(pending) + len(running) + len(failed)

    avg_sec = sum(durations) / len(durations) if durations else 0
    total_remaining_sec = avg_sec * remaining / 2.0 if avg_sec > 0 else 0

    return {
        "total": total,
        "finished": finished,
        "failed": failed,
        "running": running,
        "pending": pending,
        "done": done,
        "remaining": remaining,
        "avg_duration_sec": avg_sec,
        "avg_duration_str": format_time(avg_sec),
        "predicted_remaining_sec": total_remaining_sec,
        "predicted_remaining_str": format_time(total_remaining_sec),
        "durations_count": len(durations),
        "has_data": True
    }


# --- График обучения ---

def generate_train_plot(hash_dir):
    import numpy as np
    from io import BytesIO
    import base64

    train_dir = os.path.join(hash_dir, '00.train')
    lcurve_path = os.path.join(train_dir, 'lcurve.out')
    if not os.path.exists(lcurve_path):
        return None

    try:
        data = np.genfromtxt(lcurve_path, names=True)
        if data.size == 0:
            return None

        plt.figure(figsize=(10, 6))
        for name in data.dtype.names[1:-1]:
            plt.plot(data['step'][1:], data[name][1:], label=name)
        plt.legend()
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.xscale('symlog')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.title("Training Loss Curve")

        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        plt.close()

        buffer.seek(0)
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"image/png;base64,{img_str}"
    except Exception as e:
        print(f"Error: {e}")
        return None


# --- Анализ model_devi ---

def read_model_devi_f(file_path):
    if not os.path.exists(file_path):
        return None
    try:
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 5:
                        data.append(float(parts[4]))
        return np.array(data) if data else None
    except:
        return None


def generate_histogram_image(data, filename_prefix, title="max_devi_f distribution"):
    """
    Генерирует гистограмму, ТОЛЬКО если её ещё нет.
    Имя файла: {filename_prefix}_hist.png — фиксированное, без timestamp.
    """
    # Фиксированное имя файла
    filename = f"{filename_prefix}_hist.png"
    filepath = os.path.join(HIST_DIR, filename)

    # 🔥 Если уже есть — не создаём
    if os.path.exists(filepath):
        return f"/static/model_devi/{filename}"

    # Создаём, только если нет
    try:
        plt.figure(figsize=(6, 4))
        plt.hist(data, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        plt.title(title, fontsize=12)
        plt.xlabel('max_devi_f', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.savefig(filepath, dpi=120, bbox_inches='tight')
        plt.close()

        return f"/static/model_devi/{filename}"
    except Exception as e:
        print(f"Error generating histogram: {e}")
        return None


def analyze_model_devi_progress(work_dir):
    results = []
    iterations = sorted([
        d for d in os.listdir(work_dir)
        if d.startswith('iter.') and os.path.isdir(os.path.join(work_dir, d))
    ])

    for it in iterations:
        model_devi_dir = os.path.join(work_dir, it, '01.model_devi')
        if not os.path.exists(model_devi_dir):
            continue

        task_dirs = sorted([
            d for d in os.listdir(model_devi_dir)
            if d.startswith('task.000.') and os.path.isdir(os.path.join(model_devi_dir, d))
        ])

        for task in task_dirs:
            task_path = os.path.join(model_devi_dir, task)
            model_devi_out = os.path.join(task_path, 'model_devi.out')
            if not os.path.exists(model_devi_out):
                continue

            data = read_model_devi_f(model_devi_out)
            if data is None or len(data) == 0:
                continue

            temp = None
            input_lammps = os.path.join(task_path, 'input.lammps')
            if os.path.exists(input_lammps):
                try:
                    with open(input_lammps, 'r') as f:
                        content = f.read()
                    t_match = re.search(r'variable\s+TEMP\s+equal\s+(\d+\.?\d*)', content)
                    if t_match:
                        temp = float(t_match.group(1))
                except:
                    pass

            title = f"{it}/{task}"
            if temp:
                title += f" (T={temp}K)"
            img_path = generate_histogram_image(
                    data,
                    filename_prefix=f"{it}_{task}".replace('.', '_'),  # iter_000001_task_000_000005
                    title=f"{it}/{task} (T={temp}K)" if temp else f"{it}/{task}"
            )

            results.append({
                'iter': it,
                'task': task,
                'temperature': temp,
                'count': len(data),
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'histogram_image': img_path
            })

    results.sort(key=lambda x: (x['iter'], x['task']))
    return results


def find_active_train_dir(hash_dir):
    print(f"🔍 Поиск активной задачи обучения в: {hash_dir}")
    try:
        items = os.listdir(hash_dir)
    except Exception as e:
        print(f"❌ Не могу прочитать папку: {e}")
        return None

    # Ищем задачи с lcurve.out (для обучения)
    task_dirs = sorted([
        d for d in items
        if d.startswith('task.') and os.path.isdir(os.path.join(hash_dir, d))
    ])
    print(f"📁 Найдены задачи: {task_dirs}")

    for task_dir in task_dirs:
        task_path = os.path.join(hash_dir, task_dir)
        lcurve_path = os.path.join(task_path, 'lcurve.out')
        
        if os.path.exists(lcurve_path):
            try:
                files = os.listdir(task_path)
                has_finished = any('finished' in f.lower() for f in files)
                
                print(f"📁 {task_dir}: lcurve={os.path.exists(lcurve_path)}, finished={has_finished}")
                
                if not has_finished:
                    print(f"✅ Активная задача обучения найдена: {task_path}")
                    return task_path
            except Exception as e:
                print(f"❌ Ошибка при проверке задачи {task_dir}: {e}")
                continue

    print("❌ Активная задача обучения не найдена")
    return None


def find_active_model_devi_dir(hash_dir):
    print(f"🔍 Поиск активной задачи model deviation в: {hash_dir}")
    try:
        items = os.listdir(hash_dir)
    except Exception as e:
        print(f"❌ Не могу прочитать папку: {e}")
        return None

    # Ищем задачи с model_devi.out
    task_dirs = sorted([
        d for d in items
        if d.startswith('task.') and os.path.isdir(os.path.join(hash_dir, d))
    ])
    print(f"📁 Найдены задачи: {task_dirs}")

    # Ищем активные задачи model deviation
    for task_dir in task_dirs:
        task_path = os.path.join(hash_dir, task_dir)
        model_devi_out = os.path.join(task_path, 'model_devi.out')
        
        if os.path.exists(model_devi_out):
            try:
                files = os.listdir(task_path)
                has_finished = any('finished' in f.lower() for f in files)
                
                print(f"📁 {task_dir}: model_devi.out={os.path.exists(model_devi_out)}, finished={has_finished}")
                
                if not has_finished:
                    print(f"✅ Активная задача model deviation найдена: {task_path}")
                    return task_path
            except Exception as e:
                print(f"❌ Ошибка при проверке задачи {task_dir}: {e}")
                continue

    print("❌ Активная задача model deviation не найдена")
    return None


def analyze_current_model_devi(hash_dir):
    """Анализ текущего процесса model deviation в хэш-директории"""
    if not os.path.exists(hash_dir):
        return None
    
    # Ищем активные задачи model deviation (есть model_devi.out, но нет finished файлов)
    active_tasks = []
    
    for item in os.listdir(hash_dir):
        if item.startswith('task.') and os.path.isdir(os.path.join(hash_dir, item)):
            task_path = os.path.join(hash_dir, item)
            model_devi_out = os.path.join(task_path, 'model_devi.out')
            
            if os.path.exists(model_devi_out):
                # Проверяем, есть ли файлы с "finished"
                try:
                    files = os.listdir(task_path)
                    has_finished = any('finished' in f.lower() for f in files)
                    
                    if not has_finished:  # Активная задача
                        active_tasks.append({
                            'task_path': task_path,
                            'model_devi_out': model_devi_out,
                            'task_name': item
                        })
                except:
                    continue
    
    if not active_tasks:
        return None
    
    # Берем первую активную задачу
    task = active_tasks[0]
    
    # Читаем данные model deviation
    data = read_model_devi_f(task['model_devi_out'])
    if data is None or len(data) == 0:
        return None
    
    # Получаем температуру из input.lammps
    temp = None
    input_lammps = os.path.join(task['task_path'], 'input.lammps')
    if os.path.exists(input_lammps):
        try:
            with open(input_lammps, 'r') as f:
                content = f.read()
            t_match = re.search(r'variable\s+TEMP\s+equal\s+(\d+\.?\d*)', content)
            if t_match:
                temp = float(t_match.group(1))
        except:
            pass
    
    # Получаем время начала из времени создания файла
    start_time = None
    try:
        start_time = os.path.getctime(task['model_devi_out'])
    except:
        pass
    
    # Оцениваем ETA на основе количества обработанных шагов
    current_steps = len(data)
    eta_info = None
    
    if start_time and current_steps > 0:
        elapsed_time = time.time() - start_time
        steps_per_second = current_steps / elapsed_time if elapsed_time > 0 else 0
        
        # Предполагаем, что обычно ~10000 шагов (можно настроить)
        total_expected_steps = 10000
        remaining_steps = max(0, total_expected_steps - current_steps)
        
        if steps_per_second > 0:
            eta_seconds = remaining_steps / steps_per_second
            eta_str = format_time(eta_seconds)
            eta_info = {
                'current_steps': current_steps,
                'total_expected': total_expected_steps,
                'eta': eta_str,
                'steps_per_second': round(steps_per_second, 2)
            }
    
    # Возвращаем только сериализуемые данные (без numpy массива data)
    return {
        'task_name': task['task_name'],
        'temperature': temp,
        'count': len(data),
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'eta_info': eta_info
    }
