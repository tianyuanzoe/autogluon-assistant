from typing import Any, Dict

from autogluon_assistant.llm import AssistantChatOpenAI, LLMFactory
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from .predictor import AutogluonTabularPredictor
from .task import TabularPredictionTask
from .transformer import (
    EvalMetricInferenceTransformer,
    FilenameInferenceTransformer,
    LabelColumnInferenceTransformer,
    ProblemTypeInferenceTransformer,
    TestIdColumnInferenceTransformer,
)


class TabularPredictionAssistant:
    """A TabularPredictionAssistant performs a supervised tabular learning task"""

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.llm: AssistantChatOpenAI = LLMFactory.get_chat_model(config.llm)
        self.predictor = AutogluonTabularPredictor(config.autogluon)
        self.feature_transformers_config = config.feature_transformers

    def describe(self) -> Dict[str, Any]:
        return {
            "predictor": self.predictor.describe(),
            "config": OmegaConf.to_container(self.config),
            "llm": self.llm.describe(),  # noqa
        }

    def handle_exception(self, stage: str, exception: Exception):
        raise Exception(str(exception), stage)

    def preprocess_task(self, task: TabularPredictionTask) -> TabularPredictionTask:
        # instantiate and run task preprocessors, which infer the problem type, important filenames
        # and columns as well as the feature extractors
        task_preprocessors = (
            [
                FilenameInferenceTransformer(llm=self.llm),
                LabelColumnInferenceTransformer(llm=self.llm),
                TestIdColumnInferenceTransformer(llm=self.llm),
                ProblemTypeInferenceTransformer(),
            ]
            + ([EvalMetricInferenceTransformer(llm=self.llm)] if self.config.infer_eval_metric else [])
            + (
                [
                    instantiate(
                        ft_config,
                        problem_type=task.problem_type,
                        dataset_description=task.data_description,
                        label_column=task.label_column,
                    )
                    for ft_config in self.feature_transformers_config
                ]
                if self.feature_transformers_config
                else []
            )
        )
        for transformer in task_preprocessors:
            try:
                task = transformer.fit_transform(task)
            except Exception as e:
                self.handle_exception(f"Task preprocessing: {transformer.name}", e)

        return task

    def fit_predictor(self, task: TabularPredictionTask):
        try:
            self.predictor.fit(task)
        except Exception as e:
            self.handle_exception("Predictor Fit", e)

    def predict(self, task: TabularPredictionTask) -> Any:
        try:
            return self.predictor.predict(task)
        except Exception as e:
            self.handle_exception("Predictor Predict", e)
