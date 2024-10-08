import numpy as np


def might_importance(model_name, model, features):
    if model_name == 'might':
        importances = model.feature_importances_
        feature_names = features.columns
        indices = np.argsort(importances)[::-1]
        print("Feature ranking:")
        for f in range(features.shape[1]):
            print(f"{f + 1}. {feature_names[indices[f]]} ({importances[indices[f]]:.4f})")
    return