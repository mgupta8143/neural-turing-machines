#!/usr/bin/env bash
# Train both models and keep every figure up to date while they run.
#
#   bash overnight.sh
#
# Safe to leave overnight: figures are redrawn every 15 minutes from the latest checkpoints, so
# whatever has finished by morning is already plotted, and if the machine dies you keep the last
# refresh. On Colab, figures and results are also copied to Google Drive when it is mounted.
set -u

SEQUENCES=${SEQUENCES:-1000000}   # per model
REFRESH=${REFRESH:-900}           # seconds between figure refreshes
DRIVE=${DRIVE:-/content/drive/MyDrive/ntm-results}

BATCH=${BATCH:-1}   # the paper trains on one sequence per update

echo "$(date '+%H:%M') starting all three models at batch $BATCH, $SEQUENCES sequences each"
python -u main.py train --model ntm-ff --batch-size "$BATCH" --sequences "$SEQUENCES" > train_ntm-ff.log 2>&1 &
python -u main.py train --model ntm-lstm --batch-size "$BATCH" --sequences "$SEQUENCES" > train_ntm-lstm.log 2>&1 &
python -u main.py train --model lstm --batch-size "$BATCH" --sequences "$SEQUENCES" > train_lstm.log 2>&1 &

refresh_figures() {
  for model in ntm-ff ntm-lstm lstm; do
    [ -f "results/copy/$model/model.pt" ] || continue
    python main.py plot --model "$model" > /dev/null 2>&1
    if [ "${model#ntm}" != "$model" ]; then     # memory figures are NTM only
      python main.py memory --model "$model" --length 20 > /dev/null 2>&1
      python main.py memory --model "$model" --length 40 > /dev/null 2>&1
    fi
  done
  python main.py compare > /dev/null 2>&1
  if [ -d "$(dirname "$DRIVE")" ]; then
    mkdir -p "$DRIVE" && cp -r figures results "$DRIVE"/ 2> /dev/null
  fi
  echo "$(date '+%H:%M') figures refreshed: $(ls figures 2>/dev/null | tr '\n' ' ')"
}

while pgrep -f "main.py train" > /dev/null; do
  sleep "$REFRESH"
  refresh_figures
done

refresh_figures
echo "$(date '+%H:%M') all training finished"
