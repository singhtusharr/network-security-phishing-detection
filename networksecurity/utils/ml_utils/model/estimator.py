"""NetworkModel bundles the fitted preprocessor + trained classifier into one
object with a single .predict(). Saving/loading ONE pickled object (instead
of two separately) means inference code can never accidentally apply a model
to un-preprocessed raw input, or mismatch a preprocessor with the wrong model
version.
"""
import sys

from networksecurity.exception.exception import NetworkSecurityException


class NetworkModel:
    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def predict(self, x):
        try:
            x_transformed = self.preprocessor.transform(x)
            return self.model.predict(x_transformed)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
