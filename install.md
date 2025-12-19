# JupyterLab + virtualenv (Homebrew Python on macOS)

## 1) Create a project folder

```bash
cd ~/projects/myproj
```

## 2) Create the virtual environment

Use the exact Python you want (example: **3.14**):

```bash
python3.14 -m venv .venv
```

## 3) Activate the venv

```bash
source .venv/bin/activate
```

## 4) Upgrade pip tooling (recommended)

```bash
python -m pip install --upgrade pip setuptools wheel
```

## 5) Install packages (example: pandas)

```bash
pip install pandas jupyterlab ipykernel polars
```

```bash
python -m ipykernel install --user --name myproj-py314 --display-name "myproj (Python 3.14 venv)"
```

## 7) Start JupyterLab

Still inside the venv (`(.venv)` :

```bash
jupyter lab
```

In JupyterLab:
- **File → New → Notebook**
- Choose kernel: **“myproj (Python 3.14 venv)”**

## 8) Deactivate / delete the venv
Deactivate:
```bash
deactivate
```

## Minimal “copy/paste” version

```bash
cd ~/projects/myproj
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install pandas jupyterlab ipykernel
python -m ipykernel install --user --name myproj-py314 --display-name "myproj (Python 3.14 venv)"
jupyter lab
```
