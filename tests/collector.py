import os
import sys
import json
import unittest

import torch
from torch import tensor
from transformers import AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), '../src/llama-layer-collector'))

from llama_layer_collector import LlamaLayerCollector
from compute import compute_embedding, compute_head, compute_layer
from cache import get_shard_files
from helpers import load_shard_tensor
from load_layer import files_to_load_for_layer

CACHE_FILE_1B: str = 'data/Llama3.2-1b-instruct-cache.json'
MODEL_DIR_1B: str = 'models/Llama3.2-1b-instruct'

CACHE_FILE_8B: str = 'data/Meta-Llama-3-8B-cache.json'
MODEL_DIR_8B: str = 'models/Meta-Llama-3-8B'

NUM_KEYS_1B = 146
NUM_KEYS_8B = 291

PROMPT = "The quick brown fox jumps over the "

class LlamaLayerCollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assertTrue(os.path.exists(MODEL_DIR_1B), "1B Model does not exist for testing, please download Llama3.2-1b-instruct")
        cls.assertTrue(os.path.exists(MODEL_DIR_8B), "8B Model does not exist for testing, please download Llama3-8b")
        if os.path.exists(CACHE_FILE_1B):
            os.remove(CACHE_FILE_1B)
        if os.path.exists(CACHE_FILE_8B):
            os.remove(CACHE_FILE_8B)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(CACHE_FILE_1B):
            os.remove(CACHE_FILE_1B)
        if os.path.exists(CACHE_FILE_8B):
            os.remove(CACHE_FILE_8B)

    def test_cache_1B(self):
        collector = LlamaLayerCollector(MODEL_DIR_1B, CACHE_FILE_1B)
        self.assertEqual(len(collector.layer_files.keys()), NUM_KEYS_1B)
        self.assertTrue(os.path.exists(CACHE_FILE_1B))
        self.assertTrue(os.path.exists(collector.cache_file))
        with open(CACHE_FILE_1B, 'r') as f:
            cache = json.load(f)
            self.assertEqual(len(cache.keys()), NUM_KEYS_1B)

    def test_cache_8B(self):
        collector = LlamaLayerCollector(MODEL_DIR_8B, CACHE_FILE_8B)
        self.assertEqual(len(collector.layer_files.keys()), NUM_KEYS_8B)
        self.assertTrue(os.path.exists(CACHE_FILE_8B))
        self.assertTrue(os.path.exists(collector.cache_file))
        with open(collector.cache_file, 'r') as f:
            cache = json.load(f)
            self.assertEqual(len(cache.keys()), NUM_KEYS_8B)
    
    def test_input_embedding_1B(self):
        collector = LlamaLayerCollector(MODEL_DIR_1B, CACHE_FILE_1B)
        input_embedder = collector.load_input_embedding()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR_1B)
        input_ids = tokenizer(PROMPT, return_tensors='pt')['input_ids']
        state = compute_embedding(input_embedder, input_ids, collector.config)
        self.assertEqual(state.state.shape, (1, 9, 2048))
        self.assertEqual(state.position_ids.shape, (1, 9))
        self.assertEqual(state.position_embeddings[0].shape, (1, 9, 64))
        self.assertEqual(state.position_embeddings[1].shape, (1, 9, 64))
        self.assertEqual(state.causal_mask.shape, (1, 1, 9, 9))

    def test_input_embedding_8B(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR_8B)
        input_ids = tokenizer(PROMPT, return_tensors='pt')['input_ids']
        collector = LlamaLayerCollector(MODEL_DIR_8B, CACHE_FILE_8B)
        input_embedder = collector.load_input_embedding()
        state = compute_embedding(input_embedder, input_ids, collector.config)
        self.assertEqual(state.state.shape, (1, 9, 4096))
        self.assertEqual(state.position_ids.shape, (1, 9))
        self.assertEqual(state.position_embeddings[0].shape, (1, 9, 128))
        self.assertEqual(state.position_embeddings[1].shape, (1, 9, 128))
        self.assertEqual(state.causal_mask.shape, (1, 1, 9, 9))

    def test_norm_1B(self):
        collector = LlamaLayerCollector(MODEL_DIR_1B, CACHE_FILE_1B)
        norm = collector.load_norm()
        self.assertEqual(norm.weight.shape, (2048,))
    
    def test_norm_8B(self):
        collector = LlamaLayerCollector(MODEL_DIR_8B, CACHE_FILE_8B)
        norm = collector.load_norm()
        self.assertEqual(norm.weight.shape, (4096,))
    
    def test_head_1B(self):
        collector = LlamaLayerCollector(MODEL_DIR_1B, CACHE_FILE_1B)
        head = collector.load_head()
        self.assertEqual(head.weight.shape, (128256, 2048))
    
    def test_head_8B(self):
        collector = LlamaLayerCollector(MODEL_DIR_8B, CACHE_FILE_8B)
        head = collector.load_head()
        self.assertEqual(head.weight.shape, (128256, 4096))

    def test_layers_1B(self):
        collector = LlamaLayerCollector(MODEL_DIR_1B, CACHE_FILE_1B)
        layers = collector.load_layer_set(1, 2)
        self.assertEqual(len(layers), 2)

    def test_layers_8B(self):
        collector = LlamaLayerCollector(MODEL_DIR_8B, CACHE_FILE_8B)
        layers = collector.load_layer_set(1, 2)
        self.assertEqual(len(layers), 2)

    def test_stack_1B(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR_1B)
        input_ids = tokenizer(PROMPT, return_tensors='pt')['input_ids']
        original_num_tokens = input_ids.shape[1]
        num_tokens = 4
        current_token = 0
        collector = LlamaLayerCollector(MODEL_DIR_1B, CACHE_FILE_1B)
        input_embedder = collector.load_input_embedding()
        head = collector.load_head()
        norm = collector.load_norm()
        layers = collector.load_layer_set(0, 15)
        while current_token < num_tokens:
            state = compute_embedding(input_embedder, input_ids, collector.config)
            for lyr in layers:
                state.state = compute_layer(lyr, state)
            topk = 1
            result = compute_head(head, norm(state.state), topk)
            self.assertEqual(result.shape, (1, topk))
            token_list = input_ids.tolist()[0]
            token_list.append(result[0][0].item())
            input_ids = tensor([token_list])
            current_token += 1
        print(tokenizer.decode(input_ids[0]))
        self.assertGreater(input_ids.shape[1], original_num_tokens)

    def test_stack_1B_no_cache_file(self):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR_1B)
        input_ids = tokenizer(PROMPT, return_tensors='pt')['input_ids']
        original_num_tokens = input_ids.shape[1]
        num_tokens = 4
        current_token = 0
        os.remove(CACHE_FILE_1B)
        collector = LlamaLayerCollector(MODEL_DIR_1B)
        self.assertFalse(os.path.exists(CACHE_FILE_1B))
        self.assertTrue(collector.layer_files is not None)
        input_embedder = collector.load_input_embedding()
        head = collector.load_head()
        norm = collector.load_norm()
        layers = collector.load_layer_set(0, 15)
        while current_token < num_tokens:
            state = compute_embedding(input_embedder, input_ids, collector.config)
            for lyr in layers:
                state.state = compute_layer(lyr, state)
            topk = 1
            result = compute_head(head, norm(state.state), topk)
            self.assertEqual(result.shape, (1, topk))
            token_list = input_ids.tolist()[0]
            token_list.append(result[0][0].item())
            input_ids = tensor([token_list])
            current_token += 1
        print(tokenizer.decode(input_ids[0]))
        self.assertGreater(input_ids.shape[1], original_num_tokens)

    def test_exceptions(self):
        collector = LlamaLayerCollector(MODEL_DIR_1B, CACHE_FILE_1B)

        try:
            os.mkdir('shard_test')
            get_shard_files(collector.shard_pattern, 'shard_test')
            self.fail("Should have thrown an exception")
        except Exception:
            pass
        os.rmdir('shard_test')
        
        try:
            load_shard_tensor(collector.layer_files, collector.model_dir, 'bad_layer', 'cpu', torch.float16)
            self.fail("Should have thrown an exception")
        except ValueError:
            pass

        try:
            os.mkdir('bad_dir')
            LlamaLayerCollector('bad_dir', CACHE_FILE_1B)
            self.fail("Should have thrown an exception")
        except FileNotFoundError:
            pass

        os.rmdir('bad_dir')

        try:
            os.remove(CACHE_FILE_1B)
            collector._read_cache()
            self.fail("Should have thrown an exception")
        except FileNotFoundError:
            pass
        
        try:
            files_to_load_for_layer('bad_key', [])
            self.fail("Should have thrown an exception")
        except Exception:
            pass

# If you want to run these tests directly from the command line:
if __name__ == '__main__':
    unittest.main()
