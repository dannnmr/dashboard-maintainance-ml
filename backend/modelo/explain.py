import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def ae_feature_contribs(model, Xseq: np.ndarray, feature_names):
    rec = model.predict(Xseq, verbose=0)
    per_feat_mse = ((Xseq - rec) ** 2).mean(axis=1)  # time-avg per feature
    return pd.DataFrame(per_feat_mse, columns=feature_names)

def surrogate_tree(X_valid_aligned: pd.DataFrame, y_binary: np.ndarray, max_depth=3, random_state=42):
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    clf.fit(X_valid_aligned, y_binary)
    importances = pd.Series(clf.feature_importances_, index=X_valid_aligned.columns)
    return clf, importances.sort_values(ascending=False)
