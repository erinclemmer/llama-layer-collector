import os
import json
import shutil
import unittest

from layer_data_collector import LayerDataCollector

DATA_DIR_1B: str = 'data/Llama3.2-1b-instruct'
MODEL_DIR_1B: str = 'models/Llama3.2-1b-instruct'

DATA_DIR_8B: str = 'data/Meta-Llama-3-8B'
MODEL_DIR_8B: str = 'models/Meta-Llama-3-8B'

LAYERS_1B = 146
LAYERS_8B = 291

class LayerDataCollecterTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Start up the initial bootstrap node(s) needed for the tests.
        """
        pass
        cls.assertTrue(os.path.exists(MODEL_DIR_1B), "Model does not exist for testing, please download Llama3.2-1b-instruct")

    def setUp(self):
        if os.path.exists(DATA_DIR_1B):
            shutil.rmtree(DATA_DIR_1B)
        if os.path.exists(DATA_DIR_8B):
            shutil.rmtree(DATA_DIR_8B)

    def test_cache_1B(self):
        collector = LayerDataCollector(MODEL_DIR_1B, DATA_DIR_1B)
        self.assertEqual(len(collector.layer_files.keys()), LAYERS_1B)
        self.assertEqual(len(collector.layer_size_cache), collector.num_layers)
        self.assertTrue(os.path.exists(DATA_DIR_1B))
        self.assertTrue(os.path.exists(collector._cache_file()))
        with open(collector._cache_file(), 'r') as f:
            cache = json.load(f)
            self.assertEqual(len(cache['layer_files'].keys()), LAYERS_1B)
            self.assertEqual(len(cache['layer_sizes']), collector.num_layers)

    def test_cache_8B(self):
        collector = LayerDataCollector(MODEL_DIR_8B, DATA_DIR_8B)
        self.assertEqual(len(collector.layer_files.keys()), LAYERS_8B)
        self.assertEqual(len(collector.layer_size_cache), collector.num_layers)
        self.assertTrue(os.path.exists(DATA_DIR_8B))
        self.assertTrue(os.path.exists(collector._cache_file()))
        with open(collector._cache_file(), 'r') as f:
            cache = json.load(f)
            self.assertEqual(len(cache['layer_files'].keys()), LAYERS_8B)
            self.assertEqual(len(cache['layer_sizes']), collector.num_layers)


# If you want to run these tests directly from the command line:
if __name__ == '__main__':
    unittest.main()
