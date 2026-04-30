import sys, os
sys.path.insert(0, "/opt/anaconda3/lib/python3.12/site-packages")
sys.path.insert(0, os.path.expanduser("~/Library/Python/3.12/lib/python/site-packages"))
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--rounds", type=int, default=3)
parser.add_argument("--skip-cortex", action="store_true")
args = parser.parse_args()

from federation.run_federation import run
run(num_rounds=args.rounds, skip_cortex=args.skip_cortex, skip_upload=False)
