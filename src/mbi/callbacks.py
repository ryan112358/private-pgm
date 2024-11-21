import attr
from mbi import LinearMeasurement, Dataset, CliqueVector
from mbi import marginal_loss


@attr.dataclass
class Callback:
    loss_fns: dict[str, marginal_loss.MarginalLossFn]
    frequency: int = 50
    # Internal state
    _step: int = 0
    _logs: list = []

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
        )

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

    return Callback(loss_fns, frequency)
