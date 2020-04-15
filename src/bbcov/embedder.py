class Embedder:

    def __init__(self, model):
        self.model = model

    def __call__(self, text):
        return self.model.embed(text)
