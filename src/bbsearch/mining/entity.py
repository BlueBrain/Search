"""Named Entity Recognition."""


def find_entities(text, model):
    """Find entities in a given text.

    Parameters
    ----------
    text : str
        Arbitrarily long text without any preprocessing.

    model : spacy.lang.en.English
        Spacy model. Note that this model defines entity types.

    Returns
    -------
    spacy.tokens.doc.Doc
        Contains the extracted entities among other things.

    References
    ----------
    [1] https://allenai.github.io/scispacy/
    [2] https://spacy.io/api/doc

    """
    return model(text)
