import lpips

class PerceptualLoss:
    def __init__(self):
        self.loss_fn = lpips.LPIPS(net='vgg').cuda()

    def __call__(self, x, y):
        return self.loss_fn(x, y).mean()
