# =========================================
# Project Configuration
# =========================================

SELECTED_LABELS = [
    "yes", "no", "up", "down",
    "left", "right", "on", "off",
    "stop", "go"
]

N_MFCC = 40
FIXED_TIME_STEPS = 32
MAX_PER_CLASS = 300

BATCH_SIZE = 64
LEARNING_RATE = 0.0003
EPOCHS = 25
HIDDEN_SIZE = 192