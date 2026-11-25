"""Entry point for the shared Gata3 direction evaluation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'src/downstream/geneko'))
from evaluate_gata3_screen_direction import main

if __name__ == '__main__':
    main()
