# models/DTLZWrapper.py
import numpy as np
from pymoo.problems import get_problem
from utils import DistanceUtil


def get_ideal_point(problem_name, n_obj):
    return np.zeros(n_obj)



class ModelWrapperDTLZ:
    """
    Wrapper for evaluating DTLZ problems with exact ideal points for d2h calculation.
    """

    def __init__(self, model_config):
        """
        model_config should have:
            - problem_name: str (e.g., "dtlz1")
            - n_var: int (number of decision variables)
            - n_obj: int (number of objectives)
        """
        self.problem_name = model_config.problem.lower()
        self.n_var = model_config.n_vars
        self.n_obj = model_config.n_objs
        self.model_config = model_config
        # Get Pymoo problem
        self.problem = get_problem(
            self.problem_name,
            n_var=self.n_var,
            n_obj=self.n_obj
        )

        # Exact ideal point for d2h
        self.ideal_point = get_ideal_point(self.problem_name, self.n_obj)

    def _vector(self, hyperparams):
        return np.array([hyperparams[p] for p in self.model_config.param_names])
    
    def run_model(self, hyperparams):
        x = self._vector(hyperparams)

        # Ensure 2D input: (1, n_var)
        F = self.problem.evaluate(x[None, :], return_values_of=["F"])
        # F should now be shape (1, n_obj)
        objs = np.asarray(F)[0]          # shape (n_obj,)
        print(F)
        print(objs)
        import sys
        sys.exit(0)
        d2h = DistanceUtil.d2h(self.ideal_point.tolist(), objs.tolist())
        return 1 - d2h


    def evaluate(self, hyperparameters=None):
        x = self._vector(hyperparameters)

        F = self.problem.evaluate(x[None, :], return_values_of=["F"])
        objs = np.asarray(F)[0]

        d2h = DistanceUtil.d2h(self.ideal_point.tolist(), objs.tolist())
        return objs, d2h

    def test(self, hyperparameters=None):
        """
        Alias for evaluate
        """
        return self.evaluate(hyperparameters)
