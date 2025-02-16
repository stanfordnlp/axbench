from .model import Model
import torch, datasets
from tqdm.auto import tqdm
import os
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Union, List, Any
from torch.utils.data import DataLoader

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path

import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.WARN)
logger = logging.getLogger(__name__)


class BoW(Model):
    def __str__(self):
        return 'BoW'

    def make_model(self, **kwargs):
        # Initialize vectorizer and classifier
        model_params = kwargs.get("model_params")
        self.vectorizer = CountVectorizer()
        self.classifier = LogisticRegression(
            penalty=model_params.bow_penalty,
            C=model_params.bow_C,
            solver="lbfgs" if model_params.bow_penalty == "l2" else "liblinear",
            max_iter=1000,
            random_state=self.seed
        )
        self.concept_id = kwargs.get("concept_id")
    
    def make_dataloader(self, examples, **kwargs):
        # For BoW we don't need a dataloader, just return the examples
        return examples

    def train(self, examples, **kwargs):
        # Convert categories to binary labels
        labels = examples["labels"]
        # Fit the vectorizer and transform the text
        X = self.vectorizer.fit_transform(examples['input'])

        # Train the classifier
        self.classifier.fit(X, labels)
        
        # Calculate and log training accuracy
        train_acc = self.classifier.score(X, labels)
        logger.warning(f"Training accuracy: {train_acc:.3f}")

    def save(self, dump_dir, **kwargs):
        """Save the trained model and vectorizer for a specific concept"""
        
        # Create concept-specific directory
        dump_dir = Path(f"{dump_dir}/bow/{self.concept_id}")
        dump_dir.mkdir(parents=True, exist_ok=True)
        
        # Save classifier
        classifier_path = dump_dir / "classifier.joblib"
        joblib.dump(self.classifier, classifier_path)
        
        # Save vectorizer
        vectorizer_path = dump_dir / "vectorizer.joblib"
        joblib.dump(self.vectorizer, vectorizer_path)
        
        logger.warning(f"Saved BoW model for concept {self.concept_id} to {dump_dir}")

    def load(self, dump_dir, **kwargs):
        """Load the trained model and vectorizer for a specific concept"""
        
        # Get concept ID from kwargs
        self.concept_id = kwargs.get("concept_id")
        dump_dir = Path(f"{dump_dir}/bow/{self.concept_id}")
        
        # Load classifier
        classifier_path = dump_dir / "classifier.joblib"
        self.classifier = joblib.load(classifier_path)
        
        # Load vectorizer
        vectorizer_path = dump_dir / "vectorizer.joblib"
        self.vectorizer = joblib.load(vectorizer_path)
        
        logger.warning(f"Loaded BoW model for concept {self.concept_id} from {dump_dir}")

    @torch.no_grad()
    def predict_latent(self, examples, **kwargs):
        """Get prediction probabilities and accuracy for examples"""
        X = self.vectorizer.transform(examples['input'])
        print(X.shape)
        probs = self.classifier.predict_proba(X)[:, 1]  # Get positive class probabilities
        # If category is provided, calculate accuracy
        labels = (examples['category'] == "positive").astype(int)
        preds = (probs > 0.5).astype(int)
        accuracy = (preds == labels).mean()
        print(f"Evaluation accuracy: {accuracy}")
        return {
            "max_act": probs.tolist()
        }
    
    @torch.no_grad()
    def predict_latents(self, examples, **kwargs):
        pass