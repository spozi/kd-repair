"""Test package defaults: keep dataset-preparation progress out of unittest output."""

import os

os.environ.setdefault("KD_PROGRESS", "0")
