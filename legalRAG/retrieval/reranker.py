"""Local cross-encoder scoring of question/passage pairs."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str | None = None,
        batch_size: int = 4,
        max_length: int = 1024,
    ) -> None:
        if batch_size < 1 or max_length < 1:
            raise ValueError("Batch size and maximum length must be positive.")
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._model: CrossEncoder | None = None

    def score(self, question: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name, device=self.device, max_length=self.max_length
            )
        scores = self._model.predict(
            [(question, passage) for passage in passages],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return [float(score) for score in scores]
