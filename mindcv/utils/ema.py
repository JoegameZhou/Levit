"""ema define"""

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.common import RowTensor
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P

_ema_op = C.MultitypeFuncGraph("grad_ema_op")
_grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()
_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")

@_ema_op.register("Tensor", "Tensor", "Tensor")
def _ema_weights(factor, ema_weight, weight):
    """Apply grad sum to cumulative gradient."""
    return F.assign(ema_weight, ema_weight * factor + weight * (1 - factor))

@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * F.cast(reciprocal(scale), F.dtype(grad))


@_grad_scale.register("Tensor", "RowTensor")
def tensor_grad_scale_row_tensor(scale, grad):
    return RowTensor(grad.indices,
                     grad.values * F.cast(reciprocal(scale), F.dtype(grad.values)),
                     grad.dense_shape)


class EMACell(nn.Cell):
    """EMACell Define"""
    def __init__(self, weights, ema_decay=0.9999):
        super(EMACell, self).__init__()
        self.ema_weights = weights.clone(prefix="_ema_weights")
        self.ema_decay = Tensor(ema_decay, mstype.float32)
        self.hyper_map = C.HyperMap()

    def construct(self, weights):
        success = self.hyper_map(F.partial(_ema_op, self.ema_decay), self.ema_weights, weights)
        return success


class TrainOneStepWithEMA(nn.TrainOneStepWithLossScaleCell):
    """TrainOneStepWithEMA"""

    def __init__(self, network, optimizer, scale_sense=1.0, with_ema=False, ema_decay=0.9999):
        super(TrainOneStepWithEMA, self).__init__(network, optimizer, scale_sense)
        self.with_ema = with_ema
        if self.with_ema:
            self.ema_model = EMACell(self.weights, ema_decay=ema_decay)

    def construct(self, *inputs):
        """construct"""
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        grads = self.hyper_map(F.partial(_grad_scale, scaling_sens), grads)
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        # get the overflow buffer
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        # if there is no overflow, do optimize
        if not overflow:
            loss = F.depend(loss, self.optimizer(grads))
            if self.with_ema:
                self.ema_model(self.weights)
        return loss
