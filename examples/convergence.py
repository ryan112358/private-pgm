from mbi import Dataset, LinearMeasurement
from mbi import estimation, callbacks, marginal_oracles
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
from scipy import sparse
import IPython
import jax

# jax.config.update('jax_debug_nans', True)


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
    params["iters"] = 10000
    params["stddev"] = 10
    # Interior gradient does better with Lipschitz=1, likely some inconistency
    # with normalization somewhere.
    # True Lipschitz constant should be 1 for Q = I
    params["lipschitz"] = 1.0  # 0.00005 # needs to be tuned for now
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
    parser.add_argument("--stddev", type=float, help="noise std deviation")

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

    np.random.seed(0)
    measurements = []
    for p in projections:
        y = data.project(p).datavector()
        y = y + np.random.normal(loc=0, scale=args.stddev, size=y.size)
        measurements.append(LinearMeasurement(y, p))

    callback_fn = callbacks.default(measurements, data)
    marginal_oracle = marginal_oracles.message_passing_new

    if args.estimator == "RDA":
        model = estimation.dual_averaging(
            data.domain,
            measurements,
            lipschitz=args.lipschitz,
            iters=args.iters,
            callback_fn=callback_fn,
            marginal_oracle=marginal_oracle,
        )
    if args.estimator == "MD":
        model = estimation.mirror_descent(
            data.domain,
            measurements,
            iters=args.iters,
            callback_fn=callback_fn,
            marginal_oracle=marginal_oracle,
        )
    if args.estimator == "IG":
        model = estimation.interior_gradient(
            data.domain,
            measurements,
            lipschitz=args.lipschitz,
            iters=args.iters,
            callback_fn=callback_fn,
            marginal_oracle=marginal_oracle,
        )
    if args.estimator == "LBFGS":
        model = estimation.lbfgs(
            data.domain,
            measurements,
            iters=args.iters,
            callback_fn=callback_fn,
            marginal_oracle=marginal_oracle,
        )

    summary = callback_fn.summary
    if args.stddev == 0:
        best = 0  # known minimum
    else:
        best = summary["L2 Loss"].min()  # approximate minimum
        summary = summary.iloc[: summary.shape[0] // 5]

    xs = summary["step"] + 1
    ys = summary["L2 Loss"] - best
    coef = np.polyfit(np.log(xs), np.log(ys), deg=1)
    est = np.exp(coef[1]) * xs ** coef[0]

    plt.plot(xs, ys)
    plt.plot(xs, est, label="$O(1 / t^{%.1f})$" % (-coef[0]))
    plt.xlabel("$t$")
    plt.ylabel("$L_t-L_*$")
    # plt.yscale("log")
    plt.loglog()
    plt.legend()
    plt.savefig(f"{args.estimator}.png")

    IPython.embed()
