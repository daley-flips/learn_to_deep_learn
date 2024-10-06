class mult(object):
    def __init__(self, mul=100):
        self.mul = mul

    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x * self.mul
        y = y * self.mul
        return x, y
