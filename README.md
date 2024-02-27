# Bert-sequence-classification

## Description
Написан pipeline на pytorch-lightning для обучения модели **rubert-tiny** под задачу **sequence classification**. Выполнена корректировка синтаксиса и стиля кода с помощью линтера **wemake-python-styleguide**. Использован pre-commit - инструмент для автоматизации проверки синтаксиса кода, стиля кодирования, проведения тестов перед коммитом в git. Для работы с конфигурацией используется библиотека **OmegaConf**. Для запуска обучения используется [Makefile](Makefile).

## How to start train
For install requirements run: 
```
make install
```
For start train: 
```
make train
```

## Configuration
[configuration file](./configs/config.yaml)


# Project Organization
```
    Bert-sequence-classification/
      ├── configs/
      │   └── config.yaml               <- config for Lightning train
      ├── data/
      │   ├── test.csv                  <- test data part
      │   ├── train.csv                 <- train data part
      │   └── valid.csv                 <- valid data part
      ├── notebooks/
      ├── src/
      │   ├── config.py                 <- OmegaConf cofiguration loader
      │   ├── datamodule.py             <- lightning datamodule
      │   ├── dataset.py                <- classic torch class dataset
      │   ├── lightning_module.py       <- main lightning module
      │   ├── metrics.py                <- used metrics
      │   ├── train_utils.py            <- load objects from lib
      │   └── train.py                  <- main training module
      ├── tests/
      │   └── test_.py                  <- simple test for pre-commit
      ├── .gitignore
      ├── .pre-commit-config.yaml       <- config for pre-commit
      ├── HISTORY.md                    <- history of runs    
      ├── Makefile                      <- makefile for easy launch
      ├── setup.cfg                     <- config for wemake-python-styleguide linter
```