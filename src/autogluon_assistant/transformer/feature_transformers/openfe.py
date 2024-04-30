import hashlib
from typing import Any, Dict, Mapping, Tuple

import pandas as pd
from openfe import OpenFE, transform

from .base import BaseFeatureTransformer


class OpenFETransformer(BaseFeatureTransformer):

    identifier = "openfe"

    def __init__(self, n_jobs: int = 1, num_features_to_keep: int = 10, **kwargs) -> None:
        self.n_jobs = n_jobs
        self.num_features_to_keep = num_features_to_keep

        self.ofe = OpenFE()

        self.column_name_decodings: Dict[str, Any] = {}

        self.metadata: Dict[str, Any] = {"transformer": "OpenFE"}

    def _fit_dataframes(self, train_X: pd.DataFrame, train_y: pd.Series, **kwargs) -> None:
        train_y_df = pd.DataFrame(train_y, index=train_X.index)

        # Fill columns of type 'object' with nan values with 'unknonwn'. OpenFE doesn't handle this case.
        object_columns = train_X.select_dtypes(include="object").columns
        train_X[object_columns] = train_X[object_columns].fillna("unknown")

        train_X_renamed = self.encode_column_names(train_X)
        self.features = self.ofe.fit(data=train_X_renamed, label=train_y_df, n_jobs=self.n_jobs)
        train_X = self.decode_column_names(train_X_renamed)

    def _transform_dataframes(self, train_X: pd.DataFrame, test_X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        object_columns = train_X.select_dtypes(include="object").columns
        train_X[object_columns] = train_X[object_columns].fillna("unknown")
        test_X[object_columns] = test_X[object_columns].fillna("unknown")

        train_X_renamed = self.encode_column_names(train_X)
        test_X_renamed = self.encode_column_names(test_X)

        transformed_train_X, transformed_test_X = transform(
            train_X_renamed,
            test_X_renamed,
            self.features[: self.num_features_to_keep],
            n_jobs=self.n_jobs,
        )

        transformed_train_X = self.decode_column_names(transformed_train_X)
        transformed_test_X = self.decode_column_names(transformed_test_X)

        return transformed_train_X, transformed_test_X

    def get_metadata(self) -> Mapping:
        return self.metadata

    def hash_feature_name(self, feature_name) -> str:
        return hashlib.md5(feature_name.encode("utf-8")).hexdigest()[:8]

    def encode_column_names(self, X: pd.DataFrame) -> pd.DataFrame:
        new_column_names = []
        for column in X.columns:
            encoded_column_name = self.hash_feature_name(column)
            self.column_name_decodings[encoded_column_name] = column
            new_column_names.append(encoded_column_name)
        X.columns = new_column_names
        return X

    def decode_column_names(self, X: pd.DataFrame) -> pd.DataFrame:
        X.columns = [self.column_name_decodings.get(column, column) for column in X.columns]
        return X
