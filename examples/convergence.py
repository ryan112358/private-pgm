from mbi import Dataset, LinearMeasurement
from mbi import estimation, callbacks
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
from scipy import sparse


def default_params():
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    # Seems to be some instability with RDA and IG even with tuned Lipschitz.
    # They make faster progress than MD at first, but plateau at a suboptimal
    # value or start getting NaNs after many iterations.
    params["estimator"] = "MD"
    params["iters"] = 2000
    # Interior gradient does better with Lipschitz=1, likely some inconistency
    # with normalization somewhere.
    params["lipschitz"] = 0.00005 # needs to be tuned for now
    return params


if __name__ == "__main__":
    description = ""
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument(
        "--estimator",
        choices=["MD", "MD2", "RDA", "LBFGS", "EM", "IG"],
        help="estimator",
    ) 
    parser.add_argument("--iters", type=int, help="number of iterations")
    parser.add_argument("--lipschitz", type=float, help="lipschitz constant")

    parser.set_defaults(**default_params())
    args = parser.parse_args()

    data = Dataset.load("../data/adult.csv", "../data/adult-domain.json")
    projections = [
        ["race", "capital-loss", "income>50K"],
        ["marital-status", "capital-gain", "income>50K"],
        ["race", "native-country", "income>50K"],
        ["workclass", "sex", "hours-per-week"],
        ["fnlwgt", "marital-status", "relationship"],
        ["workclass", "education-num", "occupation"],
        ["age", "relationship", "sex"],
        ["occupation", "sex", "hours-per-week"],
        ["occupation", "relationship", "income>50K"],
    ]

    measurements = []
    for p in projections:
        y = data.project(p).datavector()
        y = y + np.random.normal(loc=0, scale=10, size=y.size)
        measurements.append(LinearMeasurement(y, p))

    callback_fn = callbacks.default(measurements, data)

    if args.estimator == "RDA":
      model = estimation.dual_averaging(data.domain, measurements, lipschitz=args.lipschitz, iters=args.iters, callback_fn=callback_fn)
    if args.estimator == "MD":
      model = estimation.mirror_descent(data.domain, measurements, iters=args.iters, callback_fn=callback_fn)
    if args.estimator == "IG":
      model = estimation.interior_gradient(data.domain, measurements, lipschitz=args.lipschitz, iters=args.iters, callback_fn=callback_fn)
    if args.estimator == "LBFGS":
        model = estimation.lbfgs(data.domain, measurements, iters=args.iters, callback_fn=callback_fn)
