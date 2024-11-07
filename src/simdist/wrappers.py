from .core import Distribution


# TODO: Implement
class PyMCModel(Distribution):

    def __init__(self, pymc_model):
        self.pymc_model = pymc_model

    def sample(self, context=None):
        # TODO: Context can be used to set sampler stuff.
        ...
