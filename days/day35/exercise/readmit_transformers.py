# day30/readmit_transformers.py  (a real .py file, so joblib and MLflow can import the classes later)
import numpy as np
import pandas as pd
# Mixin order matters: TransformerMixin LEFT of BaseEstimator (check_estimator enforces it).
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class FrequencyEncoder(TransformerMixin, BaseEstimator):
    """Replace each category with its share of the FIT rows. Codes never seen in fit get 0."""

    def __init__(self, min_count=1):
        self.min_count = min_count          # store params as-is in __init__, no logic here

    def fit(self, X, y=None):
        X = validate_data(self, X, dtype=None, ensure_all_finite=False)   # sets n_features_in_, feature_names_in_
        self.freq_ = []
        for col in X.T:
            counts = pd.Series(col).value_counts()
            counts = counts[counts >= self.min_count]
            self.freq_.append((counts / len(col)).to_dict())               # learned state ends with "_"
        return self

    def transform(self, X):
        check_is_fitted(self, "freq_")
        X = validate_data(self, X, dtype=None, ensure_all_finite=False, reset=False)
        return np.column_stack([pd.Series(col).map(f).fillna(0.0).to_numpy(float)
                                for col, f in zip(X.T, self.freq_)])

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "freq_")
        names = getattr(self, "feature_names_in_", [f"x{i}" for i in range(self.n_features_in_)])
        return np.array([f"{n}_freq" for n in names], dtype=object)

    def __sklearn_tags__(self):                 # tell check_estimator what input we accept on purpose
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True        # a missing code just maps to 0
        tags.input_tags.string = True           # DRG codes are strings like "291"
        return tags


class DateFeatures(TransformerMixin, BaseEstimator):
    """Backward-looking date features from (patient_id, admit_date). Learns one thing in fit:
    a cap for days_since_last (a high quantile of the FIT rows), also used to fill first visits."""

    def __init__(self, cap_quantile=0.95):
        self.cap_quantile = cap_quantile

    def _gaps(self, X):
        X = pd.DataFrame(X, columns=["patient_id", "admit_date"]).copy()
        X["admit_date"] = pd.to_datetime(X["admit_date"])
        s = X.sort_values(["patient_id", "admit_date"])
        gap = s.groupby("patient_id")["admit_date"].diff().dt.days      # only looks BACKWARD in time
        prior = s.groupby("patient_id").cumcount()
        return X, gap.reindex(X.index), prior.reindex(X.index)

    def fit(self, X, y=None):
        _, gap, _ = self._gaps(X)
        self.n_features_in_ = 2
        self.cap_ = float(gap.quantile(self.cap_quantile))
        return self

    def transform(self, X):
        check_is_fitted(self, "cap_")
        X, gap, prior = self._gaps(X)
        d = X["admit_date"]
        return pd.DataFrame({
            "admit_dow": d.dt.dayofweek, "weekend_admit": (d.dt.dayofweek >= 5).astype(int),
            "admit_month": d.dt.month, "winter": d.dt.month.isin([12, 1, 2]).astype(int),
            "first_visit": gap.isna().astype(int), "prior_visits": prior,
            "log_days_since_last": np.log1p(gap.fillna(self.cap_).clip(upper=self.cap_)),
        }, index=X.index).to_numpy(float)

    def get_feature_names_out(self, input_features=None):
        return np.array(["admit_dow", "weekend_admit", "admit_month", "winter",
                         "first_visit", "prior_visits", "log_days_since_last"], dtype=object)


def icd_chapter(X):
    """'I50.9' -> 'I'. A named function (not a lambda) so the pipeline can be pickled."""
    return pd.DataFrame(X).apply(lambda s: s.astype(str).str[0])
