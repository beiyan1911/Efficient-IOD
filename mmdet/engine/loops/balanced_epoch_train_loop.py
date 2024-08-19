import copy
from typing import Dict, List, Optional, Sequence, Tuple, Union

from mmengine.registry import LOOPS
from mmengine.runner.loops import EpochBasedTrainLoop
from torch.utils.data import DataLoader
import copy


@LOOPS.register_module()
class BalancedEpochBasedTrainLoop(EpochBasedTrainLoop):
    """Loop for epoch-based training.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        max_epochs (int): Total training epochs.
        val_begin (int): The epoch that begins validating.
            Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1.
        dynamic_intervals (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.
    """

    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            max_epochs: int,
            val_begin: int = 1,
            val_interval: int = 1,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        balannced_dataloader = copy.copy(dataloader.balanced)
        del dataloader.balanced
        super().__init__(runner, dataloader, max_epochs, val_begin, val_interval, dynamic_intervals)
        diff_rank_seed = runner._randomness_cfg.get('diff_rank_seed', False)
        self.balannced_dataloader = runner.build_dataloader(balannced_dataloader, seed=runner.seed,
                                                            diff_rank_seed=diff_rank_seed)
        #

    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()
        print(' ===================> start norm training <================= ')
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        print(' ===================> start balanced training <================= ')
        for idx, data_batch in enumerate(self.balannced_dataloader):
            self.run_iter(idx, data_batch)
        print(' ===================> end norm and balanced training <================= ')

        self.runner.call_hook('after_train_epoch')
        self._epoch += 1

    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_train_iter', batch_idx=idx, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.
        outputs = self.runner.model.train_step(
            data_batch, optim_wrapper=self.runner.optim_wrapper)

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1
