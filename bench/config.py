"""Constants shared by the runner, the summary and the chart."""

from pathlib import Path

GPU_PRICE_PER_HOUR = 0.35  # rented RTX 5090, USD
MAX_TOKENS = 256
CONCURRENCIES = (1, 4, 8, 32, 64, 128)
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
