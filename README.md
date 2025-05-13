# Detecting AI-generated Python code via Machine Learning

as Advanced Machine Learning S25 Project

## Contributors

Dmitry Beresnev / <d.beresnev@innopolis.university>,

Vsevolod Klyushev / <v.klyushev@innopolis.university>

Nikita Yaneev / <n.yaneev@innopolis.university>

## Projects description

Rapid advancement of artificial intelligence led to the widespread use of large language models in code generation, particularly in programming competitions. Although AI-generated code could help participants, it also raised concerns about fairness and originality in contests where human problem solving skills were evaluated. Detecting AI-generated code had to become an important task in ensuring the integrity of programming competitions.

Traditional methods of plagiarism detection were often insufficient for identifying AI-generated code, as LLMs could produce highly varied and syntactically correct solutions that differed from human-written code in subtle but detectable ways. Machine learning (ML) approaches offered a promising solution by analyzing patterns in code structure, style, and other latent features that distinguished machine-generated code from human-written code.

In this project, we explored different ML approaches to detect AI-generated Python code, because this was the most popular programming language. We compared two main strategies:

## Requirements

Code was tested on Windows 11 and Fedora Linux, Python 3.12

All the requirement packages are listed in the file `pyproject.toml`

## Before start

Using [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Optionally setup pre-commit hook:

```bash
uv run pre-commit install
```

and test it:

```bash
uv run pre-commit run --all-files
```

We also highly recommend reading report to fully understand context and purpose of some files and folders.

## Repository structure

```text
├── data                                # Data used in project
├───── ast                              # AST-related files
├──────── ...
├───── generated
├──────── dataset.csv                   # Final dataset for experiments
├───── picture                          # Picture of DTs and RFs
├──────── ...
├───── db_attempts.csv                  # Users' attempts from Accept platform
├───── db_tasks.csv                     # Tasks from Accept Platform
|
├── report
├───── pictures
├──────── *.jpg, *.png
├───── main.pdf                         # Report in PDF format
├───── main.tex
├───── presentation.pdf                 # Presentation in PDF format
|
├── src                                 # Source notebooks and scripts
├───── bot
├──────── main.py                       # Main file for Telegram bot
|
├───── notebooks
├──────── ast_nn.ipynb                              # AST-based MLP solution
├──────── ast_rf.ipynb                              # AST-based RF and DT solutions
├──────── ast_visualize.ipynb                       # Visualization of AST
├──────── build_dataset.ipynb
├──────── codebert_train_with_splits.ipynb          # CodeBERT solution
├──────── deberta_small_train_with_splits.ipynb     # Small DeBERTa solution
├──────── deberta_train_with_splits.ipynb           # DeBERTa solution
├──────── generate.ipynb                            # Notebook for dataset generation
|
├───── scripts                          # Useful scripts for LLM training
├──────── *.py
|
├── .pre-commit-config.yaml
├── .python-version
├── pyproject.toml                      # Formatter and linter settings
├── README.md                           # The top-level README
|
└── uv.lock                             # Information about uv environment
```

## Telegram bot

To start the bot just run

```bash
uv run python src/bot/main.py
```

The original bot is available on @ui_ai_detector_bot

## Contacts

In case of any questions you can contact us via university emails listed at the beginning
