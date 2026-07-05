class ModelExecutor:

    """
    Stores the user approved models
    for the training phase.
    """

    def __init__(self):

        self.selected_models = []

    # =====================================================

    def execute(self, models):

        """
        Stores validated model list.
        """

        self.selected_models = models

        return self.selected_models

    # =====================================================

    def get_models(self):

        return self.selected_models