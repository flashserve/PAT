import math
import os
import json
import random

import pandas as pd
from abc import ABC, abstractmethod


def generate_block_token_ids(block_id, block_size, max_token_id=100000):
    # print(list(range(block_id, block_id + block_size)))
    if block_id + block_size > max_token_id:
        return list(range(block_id, block_id - block_size, -1))
    return list(range(block_id, block_id + block_size))


def generate_token_ids(block_ids, block_size, max_token_id=100000, extra_prefix_len=0):
    # Generate token IDs for a list of block IDs, starting from block_id and extending by block_size.
    # This method assumes same block_id will always have the same token IDs.
    token_ids = [] if extra_prefix_len == 0 else list(range(extra_prefix_len))
    for block_id in block_ids:
        token_ids.extend(generate_block_token_ids(block_id, block_size, max_token_id))
    return token_ids


class Loader(ABC):
    """
    load or generate a dataset with list of (input_len, output_len, block_ids)
    """

    @abstractmethod
    def __init__(self):
        ...

    @abstractmethod
    def get_metrics(self) -> dict:
        """get metrics of the dataset, such as average input length, output length, etc."""
        raise NotImplementedError

    @abstractmethod
    def get_dataset(self, count: int) -> list:
        raise NotImplementedError


class MooncakeLoader(Loader):
    """Load the Mooncake dataset."""

    def __init__(self, dataset: str, max_context_len: int = None, filter_zero: bool = False, input_len: int = None, block_size: int = 512):
        assert dataset in ['conversation', 'toolagent']
        file = os.path.join(os.path.dirname(__file__), f'dataset/{dataset}_trace')
        if input_len is not None:
            file += f'_input{input_len}'
        file += ".jsonl"
        with open(file, 'r') as f:
            self.data = [json.loads(line) for line in f.readlines()]
        # filter with max_context_length
        if max_context_len is not None:
            # make input length == block size * len(block ids)
            self.data = [item for item in self.data if item['input_length']//block_size*block_size + item['output_length'] <= max_context_len]
        if filter_zero:
            self.data = [item for item in self.data if item['hash_ids'][0]!=0]

        # for item in self.data:
        #     print(item['hash_ids'])

        self.timestamp = [item['timestamp'] for item in self.data]  # ms
        self.input_lens = [item['input_length'] for item in self.data]
        self.output_lens = [item['output_length'] for item in self.data]
        self.block_ids = [item['hash_ids'] for item in self.data]

        # TODO(zhixin): currently we keep the input block ids and dynamically specify the block size,
        #  which causes the actual input length to be inconsistent with the real input length.
        # remove output blocks
        for i in range(len(self.block_ids)):
            # make input length == block size * len(block ids)
            self.block_ids[i] = self.block_ids[i][:(self.input_lens[i])//block_size]

    def get_metrics(self) -> dict:
        return {
            'count': len(self.data),
            'average_input_length': sum(self.input_lens) / len(self.input_lens),
            'average_output_length': sum(self.output_lens) / len(self.output_lens),
            'max_input_length': max(self.input_lens),
            'max_output_length': max(self.output_lens),
            'max_context_length': max(i + o for i, o in zip(self.input_lens, self.output_lens)),
        }

    def get_dataset(self, count: int) -> list:
        return self.input_lens[:count], self.output_lens[:count], self.block_ids[:count], self.timestamp[:count]

    def generator(
            self,
            count: int,
            batch_size: int | None = None,
            block_size: int = 512,
            max_context_length: int | None = None,
            max_token_id: int = 100000,
    ):
        """
        Yield batches of requests based on the specified batching strategy.
        Used for `benchmark_latency.py` only (static batching benchmark)
        """
        print(f"Generating {count} requests with batch size {batch_size}.")
        generated_count = 0
        data_idx = 0
        datasize = len(self.block_ids)

        while generated_count < count and data_idx < datasize:
            batch_tokens, output_lengths = [], []

            # Determine the size of the current batch
            if batch_size is None:
                # Dynamic batching: group requests with the same timestamp
                current_batch_size = 1
                start_ts = self.timestamp[data_idx]
                while (data_idx + current_batch_size < datasize and
                       self.timestamp[data_idx + current_batch_size] == start_ts):
                    current_batch_size += 1
            else:
                # Fixed batch size
                current_batch_size = batch_size

            # Populate the batch
            items_to_process = min(current_batch_size, count - generated_count)
            while len(batch_tokens) < items_to_process and data_idx < datasize:
                # Check context length limit
                context_len = len(self.block_ids[data_idx]) * block_size + self.output_lens[data_idx]
                if max_context_length is not None and context_len > max_context_length:
                    data_idx += 1
                    if batch_size is None:  # For dynamic batching, skipping an item means ending the current batch
                        break
                    continue

                # Generate and add the request
                tokens = generate_token_ids(self.block_ids[data_idx], block_size, max_token_id)
                batch_tokens.append(tokens)
                output_lengths.append(self.output_lens[data_idx])
                data_idx += 1

            if batch_tokens:
                generated_count += len(batch_tokens)
                yield batch_tokens, output_lengths
            elif batch_size is None and data_idx < datasize:
                # In dynamic batching, if the batch is empty (because all items were skipped),
                # we must advance data_idx to avoid an infinite loop.
                data_idx += 1

        if generated_count < count:
            print(f"Warning: only {generated_count} requests were generated, but {count} were requested.")


class QwenLoader(Loader):
    """Load the Qwen dataset."""
    def __init__(self, dataset: str, max_context_len: int = None, text_only=False, input_len: int = None):
        """
        Load the Qwen dataset.
        Args:
            dataset (str): The dataset to load, either 'traceA' or 'traceB'.
            max_context_len (int): The maximum context length for filtering requests.
            text_only (bool): If True, only load request with "type" == "text". (text, search, file, image)
        """
        assert dataset in ['traceA', 'traceB']
        file = os.path.join(os.path.dirname(__file__), f'dataset/qwen_{dataset}_blksz_16')
        if input_len is not None:
            file += f'_only_text_input{input_len}'
        file += ".jsonl"

        VOCAB_SIZE_THRESHOLD = 120000

        with open(file, 'r') as f:
            self.data = [json.loads(line) for line in f.readlines()]
        if text_only:
            self.data = [item for item in self.data if item['type'] == 'text']
        # filter with max_context_length
        if max_context_len is not None:
            self.data = [item for item in self.data if item['input_length'] + item['output_length'] <= max_context_len]

        self.timestamp = [int(1000*item['timestamp']) for item in self.data]  # ms
        self.input_lens = [item['input_length'] for item in self.data]
        self.output_lens = [item['output_length'] for item in self.data]
        self.block_ids = [
            [
                hash_id % VOCAB_SIZE_THRESHOLD if hash_id >= VOCAB_SIZE_THRESHOLD else hash_id
                for hash_id in item['hash_ids']
            ]
            for item in self.data
        ]

        # TODO(zhixin): currently we keep the input block ids and dynamically specify the block size,
        #  which causes the actual input length to be inconsistent with the real input length.
        # remove output blocks
        for i in range(len(self.block_ids)):
            self.block_ids[i] = self.block_ids[i][:(self.input_lens[i]+15)//16]

    def get_metrics(self) -> dict:
        return {
            'count': len(self.data),
            'average_input_length': sum(self.input_lens) / len(self.input_lens),
            'average_output_length': sum(self.output_lens) / len(self.output_lens),
            'max_input_length': max(self.input_lens),
            'max_output_length': max(self.output_lens),
            'max_context_length': max(i + o for i, o in zip(self.input_lens, self.output_lens)),
        }

    def get_dataset(self, count: int) -> list:
        return self.input_lens[:count], self.output_lens[:count], self.block_ids[:count], self.timestamp[:count]

    def generator(
            self,
            count: int,
            batch_size: int | None = None,
            block_size: int = 512,
            max_context_length: int | None = None,
            max_token_id: int = 100000,
    ):
        raise NotImplementedError


class BurstGPTLoader(Loader):
    """load data from BurstGPT_without_fails_1.csv"""

    def __init__(self,
                 max_context_len: int = None,
                 sys_path: str = None,
                 model_path: str = None):
        file = os.path.join(os.path.dirname(__file__), f"dataset/BurstGPT_without_fails_1.csv")
        if not os.path.exists(file):
            raise FileNotFoundError(f"file not find: {file}")

        VOCAB_SIZE_THRESHOLD = 120000
        df = pd.read_csv(file)

        sys_tokens_list = []

        if sys_path is not None:
            choices = [
                ("the United States", "English"),
                ("the United States", "Spanish"),
                ("Canada", "English"),
                ("China", "Chinese"),
            ]
            syspath = os.path.join(os.path.dirname(__file__), f"dataset/system_prompt.template")
            with open(syspath, "r") as f:
                sys_prompt_templ = f.read()

            sys_prompts = [
                sys_prompt_templ.format(LOCATION=location, LANGUAGE=language)
                for location, language in choices
            ]

            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            sys_tokens_list = [tokenizer.encode(sys) for sys in sys_prompts]

            df['sys_id'] = [random.randint(0, len(sys_prompts) - 1) for _ in range(len(df))]
            df['sys_len'] = df['sys_id'].apply(lambda x: len(sys_tokens_list[x]))
        else:
            df['sys_len'] = 0

        if max_context_len is not None:
            total_length = df['Request tokens'] + df['Response tokens'] + df['sys_len']
            df = df[total_length <= max_context_len].copy()

        df = df.iloc[:10000]

        self.timestamp = df['Timestamp'].tolist()
        self.output_lens = df['Response tokens'].tolist()
        request_lens = df['Request tokens'].tolist()

        self.block_ids = [
            [random.randint(0, VOCAB_SIZE_THRESHOLD - 1) for _ in range(req_len)]
            for req_len in request_lens
        ]

        if sys_path is not None:
            sys_ids = df['sys_id'].tolist()

            self.block_ids = [
                sys_tokens_list[sys_ids[i]] + ids
                for i, ids in enumerate(self.block_ids)
            ]

            def calculate_average_ids(token_list, chunk_size=16):
                average_ids = []
                for i in range(0, len(token_list), chunk_size):
                    chunk = token_list[i:i + chunk_size]
                    if len(chunk) == chunk_size:
                        average = sum(chunk) / len(chunk)
                        average_ids.append(int(round(average)))
                return average_ids

            self.block_ids = [
                calculate_average_ids(ids)
                for ids in self.block_ids
            ]

            self.input_lens = [
                req_len + len(sys_tokens_list[sys_ids[i]])
                for i, req_len in enumerate(request_lens)
            ]
        else:
            self.input_lens = request_lens

        self.count = len(self.input_lens)

    def get_metrics(self) -> dict:
        """
        calc metrics
        """
        if self.count == 0:
            return {
                'count': 0,
                'average_input_length': 0,
                'average_output_length': 0,
                'max_input_length': 0,
                'max_output_length': 0,
                'max_context_length': 0,
            }

        return {
            'count': self.count,
            'average_input_length': sum(self.input_lens) / self.count,
            'average_output_length': sum(self.output_lens) / self.count,
            'max_input_length': max(self.input_lens),
            'max_output_length': max(self.output_lens),
            'max_context_length': max(i + o for i, o in zip(self.input_lens, self.output_lens)),
        }

    def get_dataset(self, count: int) -> tuple:
        """
        Retrieves a subset of data with the specified number of samples.

        Args:
            count (int): The number of data samples to return.

        Returns:
            tuple: A tuple containing input_lens, output_lens, and timestamp.
        """
        if count > self.count:
            count = self.count

        return self.input_lens[:count], self.output_lens[:count], self.block_ids[:count], self.timestamp[:count]

    def generator(
            self,
            count: int,
            batch_size: int | None = None,
            block_size: int = 512,
            max_context_length: int | None = None,
            max_token_id: int = 100000,
    ):
        raise NotImplementedError


def test_load():
    loaders = [
        MooncakeLoader('conversation', max_context_len=None),
        MooncakeLoader('toolagent', max_context_len=None),
        QwenLoader('traceA', max_context_len=None),
        QwenLoader('traceB', max_context_len=None),
    ]
    for loader in loaders:
        print('*'*100)
        print(f'Loader: {loader.__class__.__name__}')
        metrics = loader.get_metrics()
        print('Metrics:', metrics)
        input_lens, output_lens, block_ids = loader.get_dataset(4)
        print('Sample Input Lengths:', input_lens)
        print('Sample Output Lengths:', output_lens)
        print('Sample Block IDs:', block_ids)

    # test generator
    for batch_tokens, output_lengths in loaders[0].generator(20, block_size=4, max_context_length=4096):
        print('BATCH: ', output_lengths)
        print(batch_tokens)
