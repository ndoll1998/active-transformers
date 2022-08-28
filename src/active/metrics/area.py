from ignite.metrics import Metric
from scipy.integrate import trapz

class AreaUnderLearningCurve(Metric):
    """ Compute area under the normalized learning curve.
        
        Note that BOTH the x and y values of the curve are
        normalized to be in range [0, 1]. By this normalization
        the shape of the curve is more important than the 
        overall progression during training. (I.e. linear
        behaviour results in better score than quadratic)
        
        Also the normalization guerantees the final score
        to be in the interval [0, 1].
    """

    def reset(self) -> None:
        # save points on curve
        self.x = []
        self.y = []

    def update(self, output):
        # add point to lists
        self.x.append(output[0])
        self.y.append(output[1])

    def compute(self) -> float:
        # normalize x values (i.e. steps)
        # to range from 0 to 1
        x_max, y_max = max(self.x), max(self.y)
        x_norm = [x_ / x_max for x_ in self.x]
        y_norm = [y_ / y_max for y_ in self.y]
        # integrate over curve
        return trapz(y_norm, x_norm)
