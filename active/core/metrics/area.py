from ignite.metrics import Metric
from scipy.integrate import trapz

class AreaUnderLearningCurve(Metric):
    """ Compute area under the normalized learning curve.    
        Note that the x-Axis is normalized to range from 0
        to 1. Also the initial starting point (0, 0) is added
        on reset.
    """

    def reset(self) -> None:
        # save points on curve
        self.x = [0]
        self.y = [0]

    def update(self, output):
        # add point to lists
        self.x.append(output[0])
        self.y.append(output[1])

    def compute(self) -> float:
        # normalize x values (i.e. steps)
        # to range from 0 to 1
        x_max = max(self.x)
        x_norm = [x_ / x_max for x_ in self.x]
        # integrate over curve
        return trapz(self.y, x_norm)
