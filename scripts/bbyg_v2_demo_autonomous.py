from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from truetrade.scalper.execution import DemoMT5Settings, ExecutionRejected, ExecutionUncertain
from truetrade.scalper.features import TickFeatureEngine
from truetrade