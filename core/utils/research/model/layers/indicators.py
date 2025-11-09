import typing

import torch
import torch.nn as nn

from core.utils.research.model.layers import Delta, KalmanStaticFilter, MultipleMovingAverages, MovingAverage, \
    MovingStandardDeviation, RelativeStrengthIndex, StochasticOscillator, OverlaysCombiner
from core.utils.research.model.model.savable import SpinozaModule


class Indicators(SpinozaModule):
    def __init__(
            self,
            delta: typing.Union[typing.List[int], typing.Union[bool, int]] = 0,
            ksf: typing.Optional[typing.List[int]] = None,
            mma: typing.Optional[typing.List[int]] = None,
            msa: typing.Optional[typing.List[int]] = None,
            msd: typing.Optional[typing.List[int]] = None,
            rsi: typing.Optional[typing.List[int]] = None,
            so: typing.Optional[typing.List[int]] = None,
            identities: int = 0,
            input_channels: int = 1,
            combiner: SpinozaModule = None
    ):
        super().__init__()
        self.__args = {
            "delta": delta,
            "ksf": ksf,
            "mma": mma,
            "msa": msa,
            "msd": msd,
            "rsi": rsi,
            "so": so,
            "identities": identities,
            "input_channels": input_channels,
            "combiner": combiner
        }
        self.delta = self.__prepare_arg_delta(delta)
        self.ksf = [KalmanStaticFilter(alpha, beta) for alpha, beta in ksf] if ksf else None
        self.mma = [MovingAverage(w) for w in mma] if mma else None
        self.msa = [MovingAverage(size) for size in msa] if msa else None
        self.msd = [MovingStandardDeviation(size) for size in msd] if msd else None
        self.rsi = [RelativeStrengthIndex(size) for size in rsi] if rsi else None
        self.so = [StochasticOscillator(size) for size in so] if so else None
        self.identities = [nn.Identity() for _ in range(identities)]
        self.combiner = OverlaysCombiner() if combiner is None else combiner
        self.input_channels = input_channels

    @staticmethod
    def __prepare_arg_delta(delta: typing.Union[typing.List[int], typing.Union[bool, int]]):
        if isinstance(delta, bool):
            delta = 1 if delta else 0
        if isinstance(delta, int):
            delta = [delta]
        return [Delta(n=n) for n in delta if n > 0]

    @property
    def indicators_len(self):
        count = self.input_channels

        for indicator_set in [self.delta, self.ksf, self.mma, self.msa, self.msd, self.rsi, self.so]:
            if indicator_set is None:
                continue
            count += len(indicator_set)*self.input_channels

        count += len(self.identities)
        return count

    def call(self, inputs: torch.Tensor):

        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)

        outputs = [inputs[:, i, :] for i in range(inputs.shape[1])]

        for indicator_set in [self.delta, self.ksf, self.mma, self.msa, self.msd, self.rsi, self.so]:
            if indicator_set is None:
                continue

            for indicator in indicator_set:
                for i in range(inputs.shape[1]):
                    outputs.append(indicator(inputs[:, i, :]))

        outputs.extend([
            identity(inputs[:, i, :])
            for i in range(inputs.shape[1])
            for identity in self.identities
        ])

        return self.combiner(outputs)

    def export_config(self) -> typing.Dict[str, typing.Any]:
        return self.__args
