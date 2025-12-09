#!/bin/bash

# Exit on error
set -e

echo "=================================================="
echo "  Baseline Evaluation - Production Run"
echo "=================================================="

# Create tmux session (survives SSH disconnection)
SESSION_NAME="baseline_eval"

# Kill existing session if any
tmux kill-session -t $SESSION_NAME 2>/dev/null || true

# Create new tmux session
tmux new-session -d -s $SESSION_NAME

# Run main script in tmux
tmux send-keys -t $SESSION_NAME "source venv/bin/activate" C-m
tmux send-keys -t $SESSION_NAME "export TINKER_API_KEY='your-api-key-here'" C-m
tmux send-keys -t $SESSION_NAME "python main.py 2>&1 | tee -a run.log" C-m

echo ""
echo "✓ Evaluation started in tmux session: $SESSION_NAME"
echo ""
echo "Commands:"
echo "  - Attach: tmux attach -t $SESSION_NAME"
echo "  - Detach: Ctrl+B then D"
echo "  - View logs: tail -f results/logs/*.log"
echo "  - View live: tail -f run.log"
echo ""
echo "You can now close your laptop. The job will continue running."
echo ""