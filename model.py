"""
model.py - STEP 3.2/3.3 Hybrid Neural Collaborative Filtering with LSTM sequence encoder (PyTorch)

Architecture:

   user_id --> Embedding(d) ----+                      item_id --> Embedding(d) ----+
   user features --> Linear+ReLU --> up                item features --> Linear+ReLU --> ip
                                 |                                                 |
   GMF branch :  (ue + up) * (ie + ip)                                           |
   LSTM branch:  concat[ue, ie, up, ip] -> sequence encoder -> final hidden state  |
   MLP branch :  concat[ue, ie, up, ip] -> FFN (128 -> 64, ReLU, Dropout)          |
   fusion     :  concat[GMF, MLP, LSTM] -> Linear(1) + biases + global_mean
                                                          ==> predicted rating (1..5)

The LSTM adds temporal-style sequence reasoning across the user/item feature interactions while
keeping the original recommender API compatible with the rest of the project.
"""
import torch
import torch.nn as nn


class HybridNCF(nn.Module):
    def __init__(self, user_features, item_features, emb_dim=32, hidden=(128, 64),
                 dropout=0.3, global_mean=3.5):
        super().__init__()
        n_users, uf_dim = user_features.shape
        n_items, if_dim = item_features.shape
        # Feature tables are stored as buffers: they travel with state_dict (reproducibility)
        self.register_buffer("user_features", torch.as_tensor(user_features, dtype=torch.float32))
        self.register_buffer("item_features", torch.as_tensor(item_features, dtype=torch.float32))
        self.register_buffer("global_mean", torch.tensor(float(global_mean)))

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.user_proj = nn.Sequential(nn.Linear(uf_dim, emb_dim), nn.ReLU())
        self.item_proj = nn.Sequential(nn.Linear(if_dim, emb_dim), nn.ReLU())

        layers, in_dim = [], emb_dim * 4
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.mlp = nn.Sequential(*layers)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=emb_dim, batch_first=True,
                            num_layers=2, dropout=dropout)
        self.out = nn.Linear(emb_dim + in_dim + emb_dim, 1)

        for emb in (self.user_emb, self.item_emb):
            nn.init.normal_(emb.weight, std=0.05)
        for b in (self.user_bias, self.item_bias):
            nn.init.zeros_(b.weight)

    def forward(self, u, i):
        """u, i: LongTensor (batch,) -> predicted ratings FloatTensor (batch,)"""
        ue, ie = self.user_emb(u), self.item_emb(i)
        up = self.user_proj(self.user_features[u])
        ip = self.item_proj(self.item_features[i])
        gmf = (ue + up) * (ie + ip)

        interaction_seq = torch.stack([ue, ie, up, ip], dim=1)
        lstm_out, _ = self.lstm(interaction_seq)
        lstm_repr = lstm_out[:, -1, :]

        mlp = self.mlp(torch.cat([ue, ie, up, ip], dim=1))
        y = self.out(torch.cat([gmf, mlp, lstm_repr], dim=1)).squeeze(-1)
        return y + self.user_bias(u).squeeze(-1) + self.item_bias(i).squeeze(-1) + self.global_mean
