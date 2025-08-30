# DPGEN Web Monitor 0.3
UNDER DEVELOPMENT

A real-time web interface for monitoring DPGEN tasks. 
This version adapted ONLY for Quantum Espresso fp task

## Features
- 🚀 Tracks current stage via `record.dpgen`
- 🔍 FP task analysis: finished, running, failed, pending
- ⏱️ Time prediction from `pw.out` (CPU time)
- 📊 Histograms of `max_devi_f` from completed `model_devi` tasks
- 📈 Training loss curve from `lcurve.out`
- 🔄 Auto-refresh every 60 seconds

## Installation

Install dependencies:

```
pip install flask matplotlib numpy
```
Pull repository

## How to Run

Just type:

```
python app.py
```

Open in browser: <http://127.0.0.1:5000>

In the web interface:
1. Enter the name of your DPGEN process working directory 
2. Click "Set Directory"

## Project Structure
```
DPGEN_WEB/
├── app.py
├── utils.py
├── templates/
│   └── index.html
├── static/
│   └── model_devi/   # generated histograms
└── README.md
```

## Notes
- Histograms are generated once per task and reused.
- Designed for local use — do not expose to public networks.
- No external tracking or data collection.
- par_qe.json and machine.json - run dpgen with Quantum espresso on local machine with 1 GPU and 16 cores. Number of cores can be chaged in command mpirun. Used system mpirun cos QE was compiled with system mpi not with dpgen one - they different. This parameters files are tuned for my particular task to learn potential for some special ionic salt - you must change parameter.json accordingly to your own task.

## Feedback
Open an issue or pull request on GitHub if you:
- Found a bug
- Want to add a feature
- Have suggestions

Built with care for DPGEN users.
