# DPGEN Web Monitor

A real-time web interface for monitoring DPGEN tasks.

## Features
- 🚀 Tracks current stage via `record.dpgen`
- 🔍 FP task analysis: finished, running, failed, pending
- ⏱️ Time prediction from `pw.out` (CPU time)
- 📊 Histograms of `max_devi_f` from completed `model_devi` tasks
- 📈 Training loss curve from `lcurve.out`
- 🔄 Auto-refresh every 60 seconds

## Installation
\`\`\`bash
pip install flask matplotlib numpy
\`\`\`

## How to Run
\`\`\`bash
python app.py
\`\`\`

Open in browser: \`http://127.0.0.1:5000\`

In the web interface:
1. Enter your DPGEN working directory (e.g. \`~/CALC/DPGEN/N4_dpgen_qe\`)
2. Click "Set Directory"

## Project Structure
\`\`\`
DPGEN_WEB/
├── app.py
├── utils.py
├── templates/
│   └── index.html
├── static/
│   └── model_devi/   # generated histograms
└── README.md
\`\`\`

## Notes
- Histograms are generated once per task and reused.
- Designed for local use — do not expose to public networks.
- No external tracking or data collection.

## Feedback
Open an issue or pull request on GitHub if you:
- Found a bug
- Want to add a feature
- Have suggestions

Built with care for DPGEN users.
