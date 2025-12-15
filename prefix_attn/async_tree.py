from . import PrefixTreeCPP
from . import utils
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List, Union, Optional
import torch
import time

class AsyncTree(PrefixTreeCPP):
    def __init__(self, 
                 block_size: int,
                 seq_lens: Union[List[int], torch.Tensor],
                 table: torch.Tensor,
                 MNWs: Union[None, List[List[int]]],
                 HRatio: int,
                 kvHead: int,
                 device: torch.device,
                 ):
        super().__init__(block_size)
        self._tree: Optional[PrefixTreeCPP] = None
        self._future: Optional[Future] = None
        self._init_time = time.time()
        
        def _build_tree():        
            tree = PrefixTreeCPP(block_size)
            tree.build_radix_tree(seq_lens, table)
            tree.pack_schedule(MNWs=MNWs, HRatio=HRatio, kvHead=kvHead)            
            tree.kernel_info.to_gpu(device)            
            return tree

        self._future = ThreadPoolExecutor(max_workers=1).submit(_build_tree)

    @property
    def tree(self) -> Optional[PrefixTreeCPP]:
        if self._tree is None:
            _start_time = time.time()
            self._tree = self._future.result()
        return self._tree
