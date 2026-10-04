# Итоговый кредитный скоринг: LightGBM

## Семейства: общая validation, разный объём train

```
                     train_rows   roc_auc  average_precision  brier_score  log_loss  approval_rate  default_rate_approved  share_defaults_rejected                           source
model                                                                                                                                                                              
LightGBM (итоговый)     2099991  0.745034           0.094264     0.033178  0.139773            0.8               0.022417                 0.492453     Прогноз загруженного Booster
CatBoost                 200000  0.722212           0.088480     0.033316  0.142095            0.8               0.024083                 0.454717  Сохранённые validation-прогнозы
Logistic Regression      200000  0.716588           0.080180     0.033678  0.143552            0.8               0.024167                 0.452830  Сохранённые validation-прогнозы
Random Forest             70000  0.692075           0.076371     0.033491  0.144930            0.8               0.026500                 0.400000  Сохранённые validation-прогнозы
```

## Итоговый LightGBM: validation / holdout

```
                           validation       holdout
rows                     15000.000000  15000.000000
defaults                   530.000000    501.000000
overall_default_rate         0.035333      0.033400
roc_auc                      0.745034      0.734377
average_precision            0.094264      0.089613
brier_score                  0.033178      0.031506
log_loss                     0.139773      0.134940
approval_rate                0.800000      0.806467
approved_count           12000.000000  12097.000000
default_rate_approved        0.022417      0.021493
defaults_approved          269.000000    260.000000
defaults_rejected          261.000000    241.000000
good_rejected             2739.000000   2662.000000
share_defaults_rejected      0.492453      0.481038
```

Порог зафиксирован до открытия holdout: 0.053229853722525304.
Метрики baseline пересчитаны из сохранённых прогнозов; финальный LightGBM загружен из model.txt.
Горизонт flag не подтверждён архивом. Holdout уже открыт. Нового обучения нет.
