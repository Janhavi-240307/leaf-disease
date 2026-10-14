"""Root-level Model Evaluation Utility for Potato Leaf Disease Classification.
Evaluates the trained CNN model on the validation dataset across all classes,
generating the confusion matrix heatmap and classification report.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluate import evaluate_model, main

if __name__ == "__main__":
    main()

