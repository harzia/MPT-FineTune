import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import umap
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

class EmbeddingEvaluator:
    def __init__(self, model, device='cuda', input_adapter=None):
        self.model = model
        self.device = device
        self.input_adapter = input_adapter
        self.model.to(device)
        self.model.eval()

    def extract_embeddings(self, dataloader):
        embeddings_list = []
        labels_list = []
        
        if hasattr(self.model, 'mod'):
            inner_model = self.model.mod
        else:
            inner_model = self.model
            
        has_fc = getattr(inner_model, 'fc', None) is not None
        
        activation = {}
        hook_handle = None

        if has_fc:
            def get_activation(name):
                def hook(model, input, output):
                    activation[name] = input[0].detach()
                return hook
            hook_handle = inner_model.fc.register_forward_hook(get_activation('fc_in'))

        print(f"Extracting features")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Inference"):
                inputs_dict = batch[0] 
                y = batch[1].to(self.device)
               
                model_args = self.input_adapter(inputs_dict, self.device)
                
                output = self.model(*model_args)
                
                if has_fc:
                    features = activation['fc_in']
                else:
                    features = output

                embeddings_list.append(features.cpu())
                labels_list.append(y.cpu())

        if hook_handle:
            hook_handle.remove()
        
        embeddings = torch.cat(embeddings_list, dim=0).numpy()
        labels = torch.cat(labels_list, dim=0).numpy()
        
        return embeddings, labels

    def evaluate_all(self, train_loader, val_loader, n_neighbors=20):
        train_emb, train_lbl = self.extract_embeddings(train_loader)
        val_emb, val_lbl = self.extract_embeddings(val_loader)
        
        results = {}

        rankme = self.compute_rankme(val_emb)
        results['rankme'] = rankme
        print(f"[Metric] RankMe Score: {rankme:.4f} (Max: {min(val_emb.shape)})")

        if train_lbl.ndim > 1: train_lbl = np.argmax(train_lbl, axis=1)
        if val_lbl.ndim > 1: val_lbl = np.argmax(val_lbl, axis=1)

        knn_acc = self.compute_knn(train_emb, train_lbl, val_emb, val_lbl, k=n_neighbors)
        results['knn_acc'] = knn_acc
        print(f"[Metric] kNN Accuracy (k={n_neighbors}): {knn_acc:.4f}")

        probe_acc = self.compute_linear_probe(train_emb, train_lbl, val_emb, val_lbl)
        results['linear_probe_acc'] = probe_acc
        print(f"[Metric] Linear Probe Accuracy: {probe_acc:.4f}")

        self.plot_embeddings(val_emb, val_lbl)
        
        return results

    def compute_rankme(self, embeddings, epsilon=1e-7):
        Z = torch.tensor(embeddings, device=self.device)
        try:
            _, S, _ = torch.linalg.svd(Z, full_matrices=False)
        except:
            _, S, _ = torch.linalg.svd(Z.cpu(), full_matrices=False)
            S = S.to(self.device)
        p = S / (torch.sum(S) + epsilon)
        return torch.exp(-torch.sum(p * torch.log(p + epsilon))).item()

    def compute_knn(self, tr_x, tr_y, te_x, te_y, k=20):
        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
        knn.fit(tr_x, tr_y)
        return knn.score(te_x, te_y)

    def compute_linear_probe(self, tr_x, tr_y, te_x, te_y):
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
        pipe.fit(tr_x, tr_y)
        return pipe.score(te_x, te_y)

    def plot_embeddings(self, embeddings, labels, save_path=None):
        plt.figure(figsize=(10, 8))
        if len(embeddings) > 5000:
            idx = np.random.choice(len(embeddings), 5000, replace=False)
            X, y = embeddings[idx], labels[idx]
        else:
            X, y = embeddings, labels

        reducer = umap.UMAP(n_components=2)
            
        proj = reducer.fit_transform(X)
        plt.scatter(proj[:, 0], proj[:, 1], c=y, cmap='tab10', s=10, alpha=0.6)
        plt.colorbar(label="Class")
        plt.title(f"Embedding Space (UMAP)")
        
        if save_path: plt.savefig(save_path); plt.close()
        else: plt.show()