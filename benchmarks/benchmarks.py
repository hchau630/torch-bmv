import torch

from torch_bmv import bmv


class TimeBmvForward:
    params = (
        [500, 5000],
        [((1,), (10,)), ((10,), (1,)), ((1, 1), (1, 10)), ((1, 1), (2, 5))],
        [False, True],
    )
    param_names = ["n", "bshapes", "naive"]

    def setup(self, n, bshapes, naive):
        self.A = torch.randn(*bshapes[0], n, n)
        self.x = torch.randn(*bshapes[1], n)

    def time_bmv_forward(self, n, bshapes, naive):
        bmv(self.A, self.x, naive=naive)


class MemBmvForward:
    params = (
        [500, 5000],
        [((1,), (10,)), ((10,), (1,)), ((1, 1), (1, 10)), ((1, 1), (2, 5))],
        [False, True],
    )
    param_names = ["n", "bshapes", "naive"]

    def setup(self, n, bshapes, naive):
        self.A = torch.randn(*bshapes[0], n, n)
        self.x = torch.randn(*bshapes[1], n)

    def peakmem_bmv_forward(self, n, bshapes, naive):
        bmv(self.A, self.x, naive=naive)


class TimeBmvBackward:
    params = (
        [500, 5000],
        [((1,), (10,)), ((10,), (1,)), ((1, 1), (1, 10)), ((1, 1), (2, 5))],
        [False, True],
    )
    param_names = ["n", "bshapes", "naive"]

    def setup(self, n, bshapes, naive):
        A = torch.randn(*bshapes[0], n, n, requires_grad=True)
        x = torch.randn(*bshapes[1], n)
        self.out = bmv(A, x, naive=naive).mean()

    def time_bmv_backward(self, n, bshapes, naive):
        self.out.backward()


class MemBmvBackward:
    params = (
        [500, 5000],
        [((1,), (10,)), ((10,), (1,)), ((1, 1), (1, 10)), ((1, 1), (2, 5))],
        [False, True],
    )
    param_names = ["n", "bshapes", "naive"]

    def setup(self, n, bshapes, naive):
        A = torch.randn(*bshapes[0], n, n, requires_grad=True)
        x = torch.randn(*bshapes[1], n)
        self.out = bmv(A, x, naive=naive).mean()

    def peakmem_bmv_backward(self, n, bshapes, naive):
        self.out.backward()
