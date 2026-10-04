# Raw data

Источник: https://www.kaggle.com/competitions/alfa-bank-pd-credit-history/data

Для просмотра сохранённого результата сырые данные не нужны. В data/evaluation лежат validation-признаки итоговой модели, validation/holdout-прогнозы, метки, ID и baseline-прогнозы. Это производные данные соревнования, не синтетические примеры.

Для анализа исходных данных и нового обучения самостоятельно скачайте ZIP соревнования и положите его как data/raw/alfa-bank-pd-credit-history.zip. В основном блокноте включите RUN_RAW_ANALYSIS=True. Распаковка создаст data/raw/data_for_competition/ с train_data/*.pq, test_data/*.pq, description.xlsx и CSV.
