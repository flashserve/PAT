import time
import math
import torch
from einops import rearrange, repeat


def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    upcast=True,
    reorder_ops=False,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling q, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if key_padding_mask is not None:
        scores.masked_fill_(
            rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf")
        )
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(
            rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0
        )
    # output = torch.einsum('bhts,bshd->bthd', attention , v)
    output = torch.einsum("bhts,bshd->bthd", attention, v)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


class CUDATimer:
    def __init__(self, disable=False):
        self.disable = disable
        self.elapsed_time = 0.0

    def __enter__(self):
        if not self.disable:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.disable:
            self.end_event.record()
            torch.cuda.synchronize()
            self.elapsed_time = self.start_event.elapsed_time(self.end_event)

    def elapsed_time_ms(self):
        return self.elapsed_time


class CPUTimer:
    def __init__(self, device, cuda_sync=True, disable=False):
        self.device = device
        self.cuda_sync = cuda_sync
        self.disable = disable
        self.elapsed_time = 0.0

    def __enter__(self):
        if not self.disable:
            if self.cuda_sync:
                torch.cuda.synchronize(self.device)
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.disable:
            if self.cuda_sync:
                torch.cuda.synchronize(self.device)
            self.elapsed_time = time.perf_counter() - self.start_time

    def elapsed_time_ms(self):
        return self.elapsed_time * 1000


class GPUTimer:
    def __init__(self, device, n_repeats=20):
        self.elapsed_time = []
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        self.device = device

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def start(self):
        torch.cuda.synchronize(device=self.device)
        self.start_event.record()

    def record(self):
        self.end_event.record()
        torch.cuda.synchronize(device=self.device)
        self.elapsed_time.append(
            self.start_event.elapsed_time(self.end_event)
        )  # ms

    def elapsed_time_ms(self):
        self.elapsed_time = sorted(self.elapsed_time)
        if len(self.elapsed_time) < 15:
            print(
                "[WARNING] n_repeats is smaller than 15, the results may be unstable."
            )
        return sum(self.elapsed_time[:-10]) / (len(self.elapsed_time) - 10)


def Timer(func, iter):
    import numpy as np

    # warm up
    for i in range(10):
        func()

    latencies = []
    for _ in range(iter):
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        func()
        end_event.record()

        torch.cuda.synchronize()

        elapsed_time_ms = start_event.elapsed_time(end_event)
        latencies.append(elapsed_time_ms)

    return np.median(latencies)
