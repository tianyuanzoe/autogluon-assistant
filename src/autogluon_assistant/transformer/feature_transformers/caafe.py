import logging
from typing import Mapping, Tuple

import pandas as pd
from caafe import CAAFEClassifier
from caafe.run_llm_code import run_llm_code

from .base import BaseFeatureTransformer

logger = logging.getLogger(__name__)


class CAAFETransformer(BaseFeatureTransformer):

    identifier = "caafe"

    def __init__(
        self,
        llm_model: str = "gpt-3.5-turbo",
        num_iterations: int = 2,
        optimization_metric: str = "roc",
        eval_model: str = "lightgbm",
        **kwargs,
    ) -> None:
        self.llm_model = llm_model
        self.iterations = num_iterations
        self.optimization_metric = optimization_metric
        self.eval_model = eval_model

        if self.eval_model == "random_forrest":
            from sklearn.ensemble import HistGradientBoostingClassifier

            clf_no_feat_eng = HistGradientBoostingClassifier(max_depth=2)

        elif self.eval_model == "tab_pfn":
            from tabpfn import TabPFNClassifier

            clf_no_feat_eng = TabPFNClassifier(device="cpu", N_ensemble_configurations=16)

        elif self.eval_model == "lightgbm":
            from lightgbm import LGBMClassifier

            clf_no_feat_eng = LGBMClassifier()

        else:
            raise ValueError(f"Unsupported CAAFE eval model: {self.eval_model}")

        self.caafe_clf = CAAFEClassifier(
            base_classifier=clf_no_feat_eng,
            optimization_metric=self.optimization_metric,
            llm_model=self.llm_model,
            iterations=self.iterations,
            display_method="print",
        )

        self.dataset_description = kwargs.get("dataset_description", "a machine learning competition")
        self.target_column_name = kwargs.get("label_column", "")

        self.supported_problem_type = kwargs["problem_type"] in ("binary", "multiclass")

        self.metadata = {"transformer": "CAAFE"}

    def _fit_dataframes(self, train_X: pd.DataFrame, train_y: pd.Series) -> None:

        if not self.supported_problem_type:
            logger.info("Feature transformer CAAFE only supports classification problems.")
            return

        categorical_target = not pd.api.types.is_numeric_dtype(train_y)
        if categorical_target:
            encoded_y, _ = train_y.factorize()

        self.caafe_clf.fit(
            train_X.to_numpy(),
            encoded_y if categorical_target else train_y.to_numpy(),
            self.dataset_description,
            train_X.columns,
            self.target_column_name,
        )
        logger.info("CAAFE generated features:")
        logger.info("{self.caafe_clf.code}")

    def _transform_dataframes(self, train_X: pd.DataFrame, test_X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

        if not self.supported_problem_type:
            return train_X, test_X

        transformed_train_X = run_llm_code(self.caafe_clf.code, train_X)
        transformed_test_X = run_llm_code(self.caafe_clf.code, test_X)

        return transformed_train_X, transformed_test_X

    def get_metadata(self) -> Mapping:
        return self.metadata
