#!/usr/bin/env bash
# Run train.py jobs on a Colab T4 with the Colab CLI (WSL), one after another, fetching results after each one.
#   bash colab/run_jobs.sh <jobs file> [session]      jobs file: one line of train.py arguments per job (incl. --out runs/...)
# A job whose <out>/metrics.json was already fetched (or exists on the VM) is skipped. Results land in runs_colab/<jobs name>/.
# Free/paid runtimes can be reclaimed: finished jobs are fetched as they end, so a rerun only redoes the lost ones.
set -uo pipefail
JOBS=${1:?jobs file}; S=${2:-vifc}
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH PYTHON_KEYRING_BACKEND=${PYTHON_KEYRING_BACKEND:-keyrings.alt.file.PlaintextKeyring}
OUT=runs_colab/$(basename "$JOBS" .txt); mkdir -p "$OUT"
TMP=$(mktemp -d); TGZ=$TMP/code.tgz; trap 'rm -rf "$TMP"' EXIT
while read -r line; do
  out=$(echo "$line" | grep -oE -- "--out [^ ]+" | cut -d" " -f2)
  [ -n "$out" ] && [ ! -f "$OUT/$out/metrics.json" ] && echo "$line"
done < "$JOBS" > "$TMP/jobs.txt"
[ -s "$TMP/jobs.txt" ] || { echo "all runs already in $OUT"; exit 0; }
tar -czf "$TGZ" train.py requirements.txt data/make_split.py data/split_article.json -C "$TMP" jobs.txt
cell() { printf '%%%%bash\ncd /content/vifc 2>/dev/null || true\n%s\n' "$1" | timeout 600 colab exec -s "$S" 2>&1; }

colab new -s "$S" --gpu T4 < /dev/null 2>&1 | tail -1 || { echo "no T4 session"; exit 1; }
colab upload -s "$S" "$TGZ" /content/vifc_code.tgz 2>&1 | tail -1
cell 'mkdir -p /content/vifc && tar -xzf /content/vifc_code.tgz -C /content/vifc && cd /content/vifc && pip install -q "transformers==5.24.0" pyarrow huggingface_hub 2>&1 | tail -1; nvidia-smi --query-gpu=name --format=csv,noheader'
# Each job: skip if done; status markers are built from two strings so a traceback echoing the cell never matches them.
cell 'cat > run.sh <<"EOF"
while read -r line; do
  [ -z "$line" ] && continue
  out=$(echo "$line" | grep -oE -- "--out [^ ]+" | cut -d" " -f2)
  [ -f "$out/metrics.json" ] && { echo "skip $out"; continue; }
  python train.py $line > "$out.log" 2>&1 || echo "FAILED $out"
done < jobs.txt
EOF
mkdir -p runs; nohup bash run.sh > /content/jobs.log 2>&1 &'

fetch() {
  # a failed tar must not leave an older archive to download (tar exit 1 = a log grew while being read: fine)
  cell "rm -f /content/out.tgz; tar -czf /content/out.tmp runs /content/jobs.log 2>/dev/null; [ \$? -le 1 ] && mv /content/out.tmp /content/out.tgz" >/dev/null
  colab download -s "$S" /content/out.tgz "$OUT/out.tgz" 2>&1 | tail -1 && tar -xzf "$OUT/out.tgz" -C "$OUT"
}
fails=0; got=0
while true; do
  sleep 120
  st=$(cell 'pgrep -f run.sh >/dev/null && echo "ST""ATE=RUN" || echo "ST""ATE=END"; echo done_runs=$(ls runs/*/metrics.json 2>/dev/null | wc -l)')
  echo "$(date +%H:%M) $(echo "$st" | grep -oE 'STATE=[A-Z]+|done_runs=[0-9]+' | tr '\n' ' ')"
  n=$(echo "$st" | grep -oE 'done_runs=[0-9]+' | cut -d= -f2)
  if [ -n "$n" ] && [ "$n" -gt "$got" ]; then fetch && got=$n; fi
  if echo "$st" | grep -q "STATE=END"; then break; fi
  if echo "$st" | grep -q "STATE=RUN"; then fails=0; else
    fails=$((fails+1)); echo "poll failed ($fails/3)"
    [ $fails -ge 3 ] && { echo "giving up polling; results so far in $OUT; session '$S' kept (colab stop -s $S)"; exit 1; }
  fi
done
cell 'grep -aE "FAILED|Traceback|Error" /content/jobs.log runs/*.log | tail -8'
if fetch; then ls "$OUT/runs"; colab stop -s "$S" < /dev/null 2>&1 | tail -1
else echo "download failed; session '$S' kept: retry, then colab stop -s $S"; exit 1; fi
