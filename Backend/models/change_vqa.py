from models.shared_vlm import get_shared_vlm

class ChangeVQA:

    def __init__(self):
        self.vlm = get_shared_vlm()

    def predict(
        self,
        image_before: str,
        image_after: str,
        question: str,
    ) -> str:

        return self.vlm.predict_change(
            image_before=image_before,
            image_after=image_after,
            question=question,
        )