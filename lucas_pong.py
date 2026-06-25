"""bigpong.py -- agent de Lucas, version "CNN pousse" : un reseau convolutif PLUS GROS
entraine en PPO (par train_bigpong.py), pour jouer fort sur LES DEUX arenes
(arena.py standard ET arena_chaos.py). UN seul .pt pour les deux tournois.

Le tournoi importe juste la classe Agent :
    python arena.py        bigpong.py:bigpong.pt  bf
    python arena_chaos.py  bigpong.py:bigpong.pt  bf

Pixels only. L'agent voit la vraie frame 80x80, la reduit a 40x40 (max-pool 2x2,
l'env n'est PAS modifie -- l'agent "plisse les yeux"), et donne au CNN une image
2 canaux [position, mouvement] (mouvement = frame courante - frame precedente, en
40x40 -> lit la VITESSE de la balle directement dans les pixels). Deux actions
seulement (UP/DOWN), joue a chaque frame, argmax a l'evaluation.

Reseau (notre version "poussee", ~2,5 M params, plus gros que pong.py) :
    conv 2->32 (4x4,s2) -> conv 32->64 (4x4,s2) -> conv 64->64 (3x3,s1)
    -> FC 6400->384 -> tete politique (2 logits) + tete valeur (1).
Ce fichier est AUTONOME (numpy + torch) : aucune dependance a arena/entrainement.
"""
from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn

UP, DOWN = 2, 3
ACTIONS = [UP, DOWN]          # index politique 0 -> UP, 1 -> DOWN
SIZE = 80
DS = 40                       # cote apres max-pool 2x2 de la frame 80x80
HIDDEN = 384


def downsample(frame):
    """80x80 -> 40x40 par max-pool 2x2. Frames binaires -> le max garde la balle
    d'1 pixel et les colonnes de raquette. Pretraitement cote agent uniquement."""
    f = np.asarray(frame, dtype=np.float32).reshape(SIZE, SIZE)
    return f.reshape(DS, 2, DS, 2).max(axis=(1, 3))


def features(cur_ds, prev_ds):
    """Image (2,40,40) : [position, mouvement]. Mouvement = diff des 2 dernieres
    frames downsamplees ; zero a la toute premiere frame d'une partie."""
    motion = (cur_ds - prev_ds) if prev_ds is not None else np.zeros_like(cur_ds)
    return np.stack([cur_ds, motion]).astype(np.float32)


class Net(nn.Module):
    """CNN pousse : 3 convs -> tronc partage -> politique (2 logits) + valeur."""
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 4, stride=2, padding=1), nn.ReLU(),   # (2,40,40) -> (32,20,20)
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),  # -> (64,10,10)
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(),  # -> (64,10,10) = 6400
        )
        self.fc = nn.Linear(64 * 10 * 10, hidden)
        self.policy_head = nn.Linear(hidden, 2)    # logits [UP, DOWN]
        self.value_head = nn.Linear(hidden, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.conv:
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, np.sqrt(2.0)); nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.fc.weight, np.sqrt(2.0)); nn.init.zeros_(self.fc.bias)
        nn.init.orthogonal_(self.policy_head.weight, 0.01); nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, 1.0);  nn.init.zeros_(self.value_head.bias)

    def forward(self, x):                          # x: (B, 2, 40, 40)
        h = self.conv(x).flatten(1)
        h = torch.relu(self.fc(h))
        return self.policy_head(h), self.value_head(h).squeeze(-1)


class Agent:
    """Contrat tournoi : reset() au debut de chaque partie, act(frame 80x80, raquette
    a DROITE) -> 2 (UP) ou 3 (DOWN)."""
    def __init__(self, weights_path=None):
        self.net = Net()
        if weights_path and os.path.exists(weights_path):
            ck = torch.load(weights_path, map_location="cpu", weights_only=False)
            state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
            self.net.load_state_dict(state)
        self.net.eval()
        self.prev_ds = None

    def reset(self):
        self.prev_ds = None

    @torch.no_grad()
    def act(self, frame):
        cur_ds = downsample(frame)
        x = features(cur_ds, self.prev_ds)
        self.prev_ds = cur_ds
        logits, _ = self.net(torch.from_numpy(x).unsqueeze(0))
        return ACTIONS[int(torch.argmax(logits, dim=-1).item())]
