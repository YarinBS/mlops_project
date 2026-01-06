# mlops_project

MLOps project git repo for the DTU MLOps course.

## Quickstart - Running the scripts

**1. Preprocess Data**
```bash
python src/mlops_project/data.py data/raw data/processed
# Or using invoke:
invoke preprocess-data
```

**2. Train model**
```bash
python src/mlops_project/train.py --lr 0.001 --epochs 20
# Or using invoke:
invoke train
```

**3. Evaluate model**
```bash
python src/mlops_project/evaluate.py models/trained_model.pth
```

**4. Visualize embeddings**
```bash
python src/mlops_project/visualize.py models/trained_model.pth --figure-name embeddings.png
```

**5. View model architecture**
```bash
python src/mlops_project/model.py
```

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
