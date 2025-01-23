import unittest
import torch
import json
import shutil
from pathlib import Path
from axbench.models.sae import *
from transformers import AutoModelForCausalLM, AutoTokenizer

class TestGemmaScopeSAE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create cache directory
        cls.cache_dir = Path(__file__).parent / "cache"
        cls.cache_dir.mkdir(exist_ok=True)
        
        # Create dummy metadata
        cls.test_metadata = [
            {"concept_id": 0, "concept": "test concept 1", 
             "ref": "https://www.neuronpedia.org/gemma-2-2b/10-gemmascope-res-16k/0",  # fake link
             "concept_genres_map": {"test concept 1": ["text"]}},
            {"concept_id": 1, "concept": "test concept 2", 
             "ref": "https://www.neuronpedia.org/gemma-2-2b/10-gemmascope-res-16k/1",  # fake link
             "concept_genres_map": {"test concept 2": ["code"]}}
        ]
        
        # Write metadata to file
        cls.metadata_path = cls.cache_dir / "metadata.jsonl"
        with open(cls.metadata_path, 'w') as f:
            for item in cls.test_metadata:
                f.write(json.dumps(item) + '\n')
        
    def setUp(self):
        pass

    def test_load_metadata_flatten(self):
        """Test load_metadata_flatten"""
        metadata_flatten = load_metadata_flatten(
            metadata_path=self.metadata_path)
        self.assertEqual(len(metadata_flatten), len(self.test_metadata))
        self.assertEqual(metadata_flatten[0]["concept"], self.test_metadata[0]["concept"])
        self.assertEqual(metadata_flatten[1]["concept"], self.test_metadata[1]["concept"])
        self.assertEqual(metadata_flatten[0]["ref"], self.test_metadata[0]["ref"])
        self.assertEqual(metadata_flatten[1]["ref"], self.test_metadata[1]["ref"])
        self.assertEqual(metadata_flatten[0]["concept_genres_map"], self.test_metadata[0]["concept_genres_map"])
        self.assertEqual(metadata_flatten[1]["concept_genres_map"], self.test_metadata[1]["concept_genres_map"])
        self.assertEqual(metadata_flatten[0]["concept_id"], self.test_metadata[0]["concept_id"])
        self.assertEqual(metadata_flatten[1]["concept_id"], self.test_metadata[1]["concept_id"])

    def test_save_pruned_sae(self):
        """Test save_pruned_sae"""
        pass

    def tearDown(self):
        """Clean up after each test"""
        torch.cuda.empty_cache()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests in the class"""
        # Remove cache directory and its contents
        shutil.rmtree(cls.cache_dir)
        torch.cuda.empty_cache()