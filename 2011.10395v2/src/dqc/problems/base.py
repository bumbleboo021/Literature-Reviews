"""Problem API: residuals and their partial derivatives w.r.t. function jets."""

import numpy as np


class Problem:
    n_outputs = 1
    order = 1
    domain = (0.0, 0.9)
    initial_values = (1.0,)
    labels = ("u",)

    def residual(self, x, evaluation):
        raise NotImplementedError

    def partials(self, x, evaluation):
        """Tuple d residual / d f^(k), each shape (M, equations, outputs)."""
        raise NotImplementedError

    def residual_jacobian(self, x, evaluation):
        if evaluation.jacobians is None:
            raise ValueError("Parameter Jacobians were not requested from the model.")
        return sum(np.einsum("meo,mop->mep", a, j)
                   for a, j in zip(self.partials(x, evaluation), evaluation.jacobians))

    def reference(self, x):
        """Evaluation-only reference. Not passed to the training objective."""
        return None

