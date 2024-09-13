import os
import sys
from pathlib import Path
import subprocess

def convert_notebooks(notebooks_dir):
    if notebooks_dir.exists():
        # Convert all notebooks in the notebooks directory to HTML
        for notebook in notebooks_dir.glob('*.ipynb'):
            if notebook.stat().st_size == 0:
                print(f"Skipping empty notebook: {notebook}")
                continue
            try:
                result = subprocess.run(
                    ['poetry', 'run', 'jupyter', 'nbconvert', '--to', 'html', str(notebook)],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(result.stdout)
            except subprocess.CalledProcessError as e:
                print(f"Failed to convert {notebook}: {e.stderr}")
    else:
        print(f"No 'notebooks' directory found at {notebooks_dir}")

if __name__ == "__main__":
    # Define the notebooks directory
    project_root = Path(__file__).resolve().parents[2]
    notebooks_dir = project_root / 'notebooks'

    convert_notebooks(notebooks_dir)