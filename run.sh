#!/usr/bin/env bash
# Launch the Pong tournament using the project's virtualenv (pongenv),
# so you don't have to remember the interpreter path or activate anything.
#
#   ./tournament/run.sh                          # realpong vs ball_follower
#   ./tournament/run.sh --opponent karpathy_pong # realpong vs karpathy_pong
#   ./tournament/run.sh --games 10               # more games
#
# Any arguments are passed straight through to run_tournament.py.
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$DIR/../pongenv/bin/python"

if [ ! -x "$PY" ]; then
  echo "error: virtualenv python not found at $PY" >&2
  echo "expected the 'pongenv' venv one level up from this folder." >&2
  exit 1
fi

exec "$PY" "$DIR/run_tournament.py" "$@"
