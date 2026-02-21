"""
Setup script para empaquetar el clasificador de especialidades médicas.
"""
import io
import os
import re
from setuptools import setup, find_packages


# Leer versión desde src/__init__.py
def get_version():
    init_path = os.path.join(os.path.dirname(__file__), "src", "__init__.py")
    with open(init_path, encoding="utf-8") as f:
        match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', f.read())
    if match:
        return match.group(1)
    raise RuntimeError("No se encontró __version__ en src/__init__.py")


# Leer README como descripción larga
def get_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, encoding="utf-8") as f:
            return f.read()
    return ""


# Dependencias principales (runtime)
INSTALL_REQUIRES = [
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "scikit-learn>=1.3.0",
    "xgboost>=2.0.0",
    "nltk>=3.8.0",
    "joblib>=1.3.0",
    "fastapi>=0.110.0",
    "uvicorn>=0.27.0",
    "pydantic>=2.5.0",
]

# Dependencias de desarrollo/testing
EXTRAS_REQUIRE = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "httpx>=0.24.0",
        "tox>=4.0.0",
        "flake8>=6.0.0",
        "build>=1.0.0",
    ],
}


setup(
    name="medical-specialty-classifier",
    version=get_version(),
    description="Clasificador de especialidades médicas basado en transcripciones con NLP",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Equipo MLOps UniAndes",
    url="https://github.com/wiflore/mlops-uniandes-project",
    packages=find_packages(exclude=["tests", "tests.*", "notebooks"]),
    package_data={
        "": ["models/*.joblib"],
    },
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    entry_points={
        "console_scripts": [
            "med-train=src.train:train_models",
            "med-api=src.api:app",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
