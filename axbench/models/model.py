import torch, einops, os
import pandas as pd
from tqdm.auto import tqdm
from pyreax import (
    gather_residual_activations, 
)
from pyvene import (
    IntervenableModel,
)


class Model(object):

    def __init__(self, model, tokenizer, layer, training_args=None, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        # abstracting layer
        self.layer = layer
        self.training_args = training_args
        self.max_activations = {}
        # Set default device to GPU if available, otherwise CPU
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    def make_model(self, **kwargs):
        pass

    def make_dataloader(self, examples, **kwargs):
        pass
    
    def train(self, examples, **kwargs):
        pass
        
    def save(self, dump_dir, **kwargs):
        model_name = kwargs.get("model_name", self.__str__())
        weight_file = dump_dir / f"{model_name}_weight.pt"
        weight = self.ax.proj.weight.data.cpu()
        if weight_file.exists():
            weight = torch.cat([torch.load(weight_file), weight], dim=0)
        torch.save(weight, weight_file)
        
        bias_file = dump_dir / f"{model_name}_bias.pt"
        bias = self.ax.proj.bias.data.cpu()
        if bias_file.exists():
            bias = torch.cat([torch.load(bias_file), bias], dim=0)
        torch.save(bias, bias_file)

    def load(self, dump_dir=None, **kwargs):
        model_name = kwargs.get("model_name", self.__str__())
        weight = torch.load(
            f"{dump_dir}/{model_name}_weight.pt"
        )
        bias = torch.load(
            f"{dump_dir}/{model_name}_bias.pt"
        )
        self.make_model(low_rank_dimension=weight.shape[1], **kwargs)
        self.ax.proj.weight.data = weight.to(self.device)
        self.ax.proj.bias.data = bias.to(self.device)
    
    @torch.no_grad()
    def predict_latent(self, examples, **kwargs):
        self.ax.eval()
        batch_size = kwargs.get('batch_size', 32)
        
        all_acts = []
        all_max_act = []
        all_max_act_idx = []
        all_max_token = []
        all_tokens = []
        # Process in batches
        for i in range(0, len(examples), batch_size):
            batch = examples.iloc[i:i + batch_size]
            # Batch encode all inputs and send to model's device
            inputs = self.tokenizer(
                batch["input"].tolist(),
                return_tensors="pt",
                padding=True,
                add_special_tokens=True
            ).to(self.device)  # Use model's device
            
            act_in = gather_residual_activations(
                self.model, self.layer, inputs)
            
            ax_acts_batch = self.ax(act_in[:, 1:])  # no bos token
            # Process each sequence in the batch
            seq_lens = inputs["attention_mask"].sum(dim=1) - 1 # no bos token
            for seq_idx, row in enumerate(batch.itertuples()):
                # select acts with attention mask
                acts = ax_acts_batch[
                    seq_idx, :seq_lens[seq_idx], row.concept_id].flatten().cpu().numpy().tolist()
                acts = [round(x, 3) for x in acts]
                max_act = max(acts)
                max_act_indices = [i for i, x in enumerate(acts) if x == max_act]
                max_act_idx = max_act_indices[0]
                # Get tokens for this specific sequence
                tokens = self.tokenizer.tokenize(row.input)
                max_token = tokens[max_act_idx]
                all_acts.append(acts)
                all_max_act.append(max_act)
                all_max_act_idx.append(max_act_idx)
                all_max_token.append(max_token)
                all_tokens.append(tokens)

        return {
            "acts": all_acts,
            "max_act": all_max_act,
            "max_act_idx": all_max_act_idx,
            "max_token": all_max_token,
            "tokens": all_tokens
        }

    @torch.no_grad()
    def predict_steer(self, examples, **kwargs):
        self.ax.eval()
        # set tokenizer padding to left
        self.tokenizer.padding_side = "left"
        # depending on the model, we use different concept id columns
        concept_id_col = "sae_id" if "sae" in self.__str__().lower() else "concept_id"

        # iterate rows in batch
        batch_size = kwargs.get("batch_size", 64)
        eval_output_length = kwargs.get("eval_output_length", 128)
        all_generations = []
        all_perplexities = []
        all_strenghts = []

        progress_bar = tqdm(range(0, len(examples), batch_size))
        for i in range(0, len(examples), batch_size):
            batch_examples = examples.iloc[i:i+batch_size]
            input_strings = batch_examples['input'].tolist()
            mag = torch.tensor(batch_examples['factor'].tolist()).to(self.device)
            idx = torch.tensor(batch_examples[concept_id_col].tolist()).to(self.device)
            max_acts = torch.tensor([
                self.max_activations.get(id, 1.0) 
                for id in batch_examples[concept_id_col].tolist()]).to(self.device)
            # tokenize input_strings
            inputs = self.tokenizer(
                input_strings, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            _, generations = self.ax_model.generate(
                inputs, 
                unit_locations=None, intervene_on_prompt=True, 
                subspaces=[{"idx": idx, "mag": mag, "max_act": max_acts}],
                max_new_tokens=eval_output_length, do_sample=True, 
                # following neuronpedia, we use temperature=0.5 and repetition_penalty=2.0
                temperature=0.5, repetition_penalty=2.0
            )

            # Decode and print only the generated text without prompt tokens
            input_lengths = [len(input_ids) for input_ids in inputs.input_ids]
            generated_texts = [
                self.tokenizer.decode(generation[input_length:], skip_special_tokens=True)
                for generation, input_length in zip(generations, input_lengths)
            ]
            all_generations += generated_texts

            # Calculate perplexity for each sequence
            batch_input_ids = self.tokenizer(
                generated_texts, return_tensors="pt", padding=True, truncation=True).input_ids.to(self.device)
            batch_attention_mask = (batch_input_ids != self.tokenizer.pad_token_id).float()
            
            # Forward pass without labels to get logits
            outputs = self.model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            
            logits = outputs.logits[:, :-1, :].contiguous()  # Remove last token prediction
            target_ids = batch_input_ids[:, 1:].contiguous()  # Shift right by 1
            
            # Calculate loss for each token
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            token_losses = loss_fct(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            
            # Reshape losses and mask
            token_losses = token_losses.view(batch_input_ids.size(0), -1)
            mask = batch_attention_mask[:, 1:].contiguous()
            
            # Calculate perplexity for each sequence
            seq_lengths = mask.sum(dim=1)
            seq_losses = (token_losses * mask).sum(dim=1) / seq_lengths
            seq_perplexities = torch.exp(seq_losses).tolist()
            all_perplexities.extend(seq_perplexities)
            all_strenghts.extend((mag*max_acts).tolist())
            progress_bar.update(1)

        return {
            "steered_generation": all_generations,
            "perplexity": all_perplexities,
            "strength": all_strenghts,
        }

    def get_logits(self, concept_id, k=10):
        top_logits, neg_logits = [None], [None]
        if concept_id is not None:
            W_U = self.model.lm_head.weight.T
            W_U = W_U * (self.model.model.norm.weight +
                        torch.ones_like(self.model.model.norm.weight, dtype=torch.float32))[:, None]
            W_U -= einops.reduce(
                W_U, "d_model d_vocab -> 1 d_vocab", "mean"
            )

            vocab_logits = self.ax.proj.weight.data[concept_id] @ W_U
            top_values, top_indices = vocab_logits.topk(k=k, sorted=True)
            top_tokens = self.tokenizer.batch_decode(top_indices.unsqueeze(dim=-1))
            top_logits = [list(zip(top_tokens, top_values.tolist()))]
            
            neg_values, neg_indices = vocab_logits.topk(k=k, largest=False, sorted=True)
            neg_tokens = self.tokenizer.batch_decode(neg_indices.unsqueeze(dim=-1))
            neg_logits = [list(zip(neg_tokens, neg_values.tolist()))]
        return top_logits, neg_logits
    
    def pre_compute_mean_activations(self, dump_dir, **kwargs):
        # For ReAX, we need to look into the concept in the same group, since they are used in training.
        max_activations = {} # sae_id to max_activation
        # Loop over saved latent files in dump_dir.
        for file in os.listdir(dump_dir):
            if file.startswith("latent_") and file.endswith(".parquet"):
                latent_path = os.path.join(dump_dir, file)
                latent = pd.read_parquet(latent_path)
                # loop through unique sorted concept_id
                for concept_id in sorted(latent["concept_id"].unique()):
                    concept_latent = latent[latent["concept_id"] == concept_id]
                    max_act = concept_latent["ReAX_max_act"].max()
                    max_activations[concept_id] = max_act if max_act > 0 else 50
        self.max_activations = max_activations
        return max_activations  

    def to(self, device):
        """Move model to specified device"""
        self.device = device
        if hasattr(self, 'ax'):
            self.ax = self.ax.to(device)
            if hasattr(self, 'ax_model'):
                if isinstance(self.ax_model, IntervenableModel):
                    self.ax_model.set_device(device)
                else:
                    self.ax_model = self.ax_model.to(device)
        return self
