# DPGEN Web Monitor 0.4

A real-time web interface for monitoring DPGEN tasks.  
This version is adapted for Quantum Espresso FP tasks.

## Features

- 🚀 Tracks current stage via `record.dpgen`  
- 🔍 **FP task analysis**: finished, running, failed, pending  
- ⏱️ Time prediction from `pw.out` (CPU time)  
- 📊 **Live histogram of `max_devi_f`** for the current active `model_devi` task  
- 📈 **Live training loss curve** from `lcurve.out` for the current active training task  
- 📚 **Training History**: View loss curves for all completed training tasks (cached)  
- 🔄 Auto-refresh every 30 seconds for live data, every 5 minutes for the full page

## Installation

Install dependencies:

```bash
pip install flask matplotlib numpy
```

Pull the repository.

## How to Run

1. Run the application:

```bash
python app.py
```

2. Open in browser: `http://127.0.0.1:5000` (local access only by default).

3. In the web interface:  
    - Enter the path to your DPGEN process working directory.  
    - Click "Set Directory".

## Project Structure

```
DPGEN_WEB/  
├── app.py              # Main Flask application  
├── utils.py            # Helper functions for parsing and analysis  
├── templates/  
│   ├── index.html      # Main monitoring page  
│   └── train_history.html # Page for viewing completed training tasks  
├── static/  
│   ├── model_devi/     # Generated histograms for completed model_devi tasks  
│   └── train_plots/    # Cached plots for completed training tasks  
└── README.md
```

## Notes

- **Histograms & Plots**:  
    - Histograms for completed `model_devi` tasks are generated once and reused.  
    - Plots for completed training tasks are generated once and cached in `static/train_plots/`.  
    - Live plots (current training/model deviation) are generated dynamically on each refresh.  
- **Designed for Local Use**: Do not expose this interface to public networks without proper security measures.  
- **No External Tracking**: The application does not collect or transmit any data externally.  
- **Quantum Espresso Specific**: The FP analysis (`pw.out` parsing) is tailored for Quantum Espresso. The provided `param_qe.json` and `machine.json` files are examples for running DPGEN with QE locally (1 GPU, 16 cores, system `mpirun`). You must adapt these parameter files for your specific task and system.

## Feedback

Open an issue or pull request on GitHub if you:  
- Found a bug  
- Want to add a feature  
- Have suggestions

Built with care for DPGEN users.
