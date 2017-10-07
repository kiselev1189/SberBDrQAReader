Обучение модели:

1. Скачайте файл с моделью embedding слов, готовые модели вы можете найти на сайте http://rusvectores.org/ru/models/ или выполните `scripts/reader/download_w2v.sh`

2. Трансформируйте модель embedding слов в текстовый формат: для этого выполните все ячейке в ноутбуке
   [scripts/reader/BinaryW2VToSpaceSepartor.ipynb](`scripts/reader/BinaryW2VToSpaceSepartor.ipynb`)

3. Разделите файл на обучающую выборку и валидационную

4. Сделайте подготовку данных для обучения: `PYTHONPATH=.:$PYTHONPATH python3 scripts/reader/preprocess.py --tokenizer SimpleTokenizer train.csv output_filename.json`

5. В [scripts/reader/train.sh](`scripts/reader/train.sh`) вы можете найти пример запуска обучения

6. После обучения можете делать сабмит: [scripts/reader/train.sh](`sh create_zip.sh`) положит все необходимые файлы (убедитесь, что среди них есть модель, если вы переименовали модель незабудьте )

7. Также вы можете запустить сессию в интерактивном режиме `PYTHONPATH=.:$PYTHONPATH python3 scripts/reader/interactive.py --model models/20171007-1ce20c3f.mdl`
