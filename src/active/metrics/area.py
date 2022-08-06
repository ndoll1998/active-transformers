from ignite.metrics import Metric
from scipy.integrate import trapz

class AreaUnderLearningCurve(Metric):

    def reset(self) -> None:
        # save points on curve
        self.x = []
        self.y = []

    def update(self, output):
        # add point to lists
        self.x.append(output[0])
        self.y.append(output[1])

    def compute(self) -> float:
        # integrate over curve
        return trapz(self.y, self.x)
