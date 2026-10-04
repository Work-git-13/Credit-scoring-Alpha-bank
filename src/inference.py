from pathlib import Path
import json
import numpy as np
import lightgbm as lgb

def encode_prepared(features, schema):
    numeric = features[schema['numerical_features']].to_numpy(dtype='float32', copy=True)
    encoded = np.empty((len(features), len(schema['categorical_features'])), dtype='float32')
    for i, col in enumerate(schema['categorical_features']):
        values = features[col].fillna('missing').astype(str)
        encoded[:, i] = values.map(schema['category_maps'][col]).fillna(-1).to_numpy(dtype='float32')
    return np.concatenate([numeric, encoded], axis=1)

def load_model(model_dir):
    model_dir = Path(model_dir)
    with open(model_dir / 'schema.json', encoding='utf-8') as file:
        schema = json.load(file)
    return lgb.Booster(model_file=str(model_dir / 'model.txt')), schema

def predict_prepared(features, model_dir):
    booster, schema = load_model(model_dir)
    return booster.predict(encode_prepared(features, schema))
