from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import time

from truetrade.scalper.execution import DemoMT5Settings
from truetrade.scalper.history_import import insert_tick_payload, normalize_history_rows
from truetrade.scalper.store import ScalperStore
from truetrade.scalper.timebase import BrokerTimebase