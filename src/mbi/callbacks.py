import attr
import jax
from mbi import LinearMeasurement, Dataset, CliqueVector
from mbi import marginal_loss
import pandas as pd

@attr.dataclass
class Callback:
    loss_fns: dict[str, marginal_loss.MarginalLossFn]
    frequency: int = 50
    # Internal state
    _step: int = 0
    _logs: list = attr.field(factory=list)

    def __call__(self, marginals: CliqueVector):
        if self._step == 0:
            print("step", *self.loss_fns.keys(), sep="        ")
        if self._step % self.frequency == 0:
            row = [self.loss_fns[key](marginals) for key in self.loss_fns]
            self._logs.append([self._step] + row)
            print(self._step, *["%.1f" % v for v in row], sep="\t")
        self._step += 1

    @property
    def summary(self):
        return pd.DataFrame(
            columns=["step"] + list(self.loss_fns.keys()), data=self._logs
        ).astype(float)

def default(
    measurements: list[LinearMeasurement],
    data: Dataset | None = None,
    frequency: int = 50,
) -> Callback:
    loss_fns = {}
    # Measures distance between input marginals and noisy marginals.
    loss_fns["L2 Loss"] = marginal_loss.from_linear_measurements(
        measurements, norm="l2"
    )
    loss_fns["L1 Loss"] = marginal_loss.from_linear_measurements(
        measurements, norm="l1"
    )

    if data is not None:
        # Measures distance between input marginals and true marginals.
        ground_truth = [
            LinearMeasurement(
                M.query(data.project(M.clique).datavector()),
                clique=M.clique,
                stddev=1,
                query=M.query,
            )
            for M in measurements
        ]
        loss_fns["L2 Error"] = marginal_loss.from_linear_measurements(
            ground_truth, norm="l2"
        )
        loss_fns["L1 Error"] = marginal_loss.from_linear_measurements(
            ground_truth, norm="l1"
        )


    loss_fns = { key: jax.jit(loss_fns[key].__call__) for key in loss_fns }
    loss_fns['Primal Feasibility'] = jax.jit(marginal_loss.primal_feasibility)

    def grad_norm(mu):
        grad = jax.grad(loss_fns["L2 Loss"])(mu)
        return grad.dot(grad)**0.5

    loss_fns['Gradient Norm'] = grad_norm

    return Callback(loss_fns, frequency)
