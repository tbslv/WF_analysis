from pathlib import Path
from setuptools import setup, find_packages

# Read requirements from requirements.txt (if present)
req_path = Path(__file__).parent / "requirements_clean.txt"
if req_path.exists():
    with req_path.open() as f:
        install_requires = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]
else:
    install_requires = []

setup(
    name="wf_analysis",
    version="0.1.0",
    description="Widefield imaging analysis pipeline",
    author="Tobias Leva",
    packages=find_packages(include=["wf", "wf.*", "wf_analysis", "wf_analysis.*", "scripts", "scripts.*"]),
    python_requires=">=3.8",
    install_requires=install_requires,
)
