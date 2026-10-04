"""Проверка сохранённых весов на реальных validation-признаках, без fit."""
from pathlib import Path
import sys
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inference import load_model, encode_prepared
booster, schema = load_model(ROOT / 'models/lightgbm')
features = pd.read_parquet(ROOT / 'data/evaluation/validation_features.parquet').set_index('id')
predictions = pd.read_parquet(ROOT / 'data/evaluation/validation_predictions.parquet')
p = booster.predict(encode_prepared(features.loc[predictions.id], schema))
np.testing.assert_allclose(p, predictions.model_probability, rtol=1e-6, atol=1e-8)
np.testing.assert_array_equal(features.loc[predictions.id, 'flag'], predictions.flag)
manifest = json.loads((ROOT / 'models/lightgbm/split_manifest.json').read_text())
for name in ['validation', 'holdout']:
    frame = pd.read_parquet(ROOT / f'data/evaluation/{name}_predictions.parquet')
    digest = hashlib.sha256(np.sort(frame.id.to_numpy(dtype='int64')).tobytes()).hexdigest()
    assert digest == manifest['id_hashes'][name]
print(f'OK: {booster.current_iteration()} trees; validation AUC={roc_auc_score(predictions.flag,p):.6f}, AP={average_precision_score(predictions.flag,p):.6f}')
