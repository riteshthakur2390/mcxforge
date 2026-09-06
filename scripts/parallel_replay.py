import sys
import logging
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from loguru import logger
logger.remove()
logging.disable(logging.CRITICAL)

import time
from concurrent.futures import ProcessPoolExecutor
from signalforge.backtest.deterministic_replay_engine import DeterministicReplayEngine

def run_one_session(args):
    s_date, s_idx, mode, warmup = args
    eng = DeterministicReplayEngine(context_mode=mode, warmup_bars=warmup)
    _, trades = eng.replay_session(s_date, s_idx)
    return trades

def main():
    eng0 = DeterministicReplayEngine()
    sessions = eng0.available_sessions[:16]
    args = [(s, idx+1, "LEGACY_CONTEXT", 0) for idx, s in enumerate(sessions)]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(run_one_session, args))
    t1 = time.time()
    total_trades = sum(len(r) for r in results)
    print(f"16 sessions in parallel took {t1-t0:.2f}s | Trades: {total_trades}")

if __name__ == "__main__":
    main()
