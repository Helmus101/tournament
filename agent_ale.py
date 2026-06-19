"""
================================================================================
 AGENT ALE — Coeur partage (reseau, pretraitement, ball-follower, contrat d'agent)
================================================================================
Source de verite UNIQUE pour pong_v3 (vrai Atari via PettingZoo). Importe par
train_ale.py (entrainement) et arena_ale.py (defi 1v1). Tout est verifie
empiriquement sur pong_v3 :
  - frame 210x160x3 uint8, fond = couleur 144, zone de jeu = lignes 34..194
  - actions : 2 = HAUT (y diminue), 3 = BAS (y augmente), 0 = immobile
  - raquette gauche (second_0) en x=16..19 ; raquette droite (first_0) en x=140..143
  - balle dans la bande centrale x=30..130

Reseau = Karpathy "from pixels" : 1 couche cachee de 200, tete politique P(HAUT)
(+ tete de valeur = baseline qui reduit la variance et accelere l'apprentissage).
================================================================================
"""
import os
import numpy as np
import torch
import torch.nn as nn

# ── Constantes du jeu (verifiees sur pong_v3) ──────────────────────────────────
UP, DOWN, NOOP = 2, 3, 0
D = 80 * 80
BG = 144
TOP, BOT = 34, 194
LEFT_BAND  = (0, 30)        # raquette gauche  (second_0)
RIGHT_BAND = (130, 160)     # raquette droite  (first_0)
BALL_BAND  = (30, 130)      # zone centrale = balle

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Pretraitement facon Karpathy : frame -> vecteur binaire 6400 ───────────────
def preprocess(frame: np.ndarray) -> np.ndarray:
    f = frame[35:195][::2, ::2, 0].copy()   # zone de jeu, sous-echantillonne x2, canal 0
    f[f == BG] = 0                          # efface le fond
    f[f != 0] = 1                           # binarise : objets -> 1
    return f.astype(np.float32).ravel()


# ── Reseau ─────────────────────────────────────────────────────────────────────
class PolicyNet(nn.Module):
    def __init__(self, hidden: int = 200, d_in: int = D):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden)
        self.policy_head = nn.Linear(hidden, 1)   # -> P(HAUT)
        self.value_head  = nn.Linear(hidden, 1)   # -> V(s) baseline
        nn.init.normal_(self.fc1.weight, 0, 1 / np.sqrt(d_in)); nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.policy_head.weight, 0, 1 / np.sqrt(hidden)); nn.init.zeros_(self.policy_head.bias)
        nn.init.zeros_(self.value_head.weight); nn.init.zeros_(self.value_head.bias)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        prob  = torch.sigmoid(self.policy_head(h)).squeeze(-1)   # [B]
        value = self.value_head(h).squeeze(-1)                   # [B]
        return prob, value


# ── Detection pixels (pour l'adversaire scripte) ───────────────────────────────
def _foreground(frame: np.ndarray) -> np.ndarray:
    g = frame[:, :, 0]
    m = (g != BG).copy()
    m[:TOP] = False; m[BOT:] = False
    return m

def _band_y(mask, band):
    ys, _ = np.nonzero(mask[:, band[0]:band[1]])
    return float(ys.mean()) if len(ys) else None

def ball_follower_action(frame: np.ndarray, side: str = "left", last_ball_y=None):
    """Adversaire scripte qui SUIT la balle. Renvoie (action, ball_y_memorise).
    side = 'left' (second_0) ou 'right' (first_0). Sparring-partner fort mais battable."""
    m = _foreground(frame)
    ball_y = _band_y(m, BALL_BAND)
    if ball_y is None:
        ball_y = last_ball_y                     # balle non visible -> derniere position connue
    pad_y = _band_y(m, LEFT_BAND if side == "left" else RIGHT_BAND)
    if ball_y is None or pad_y is None:
        return NOOP, ball_y
    if ball_y < pad_y - 3:   return UP, ball_y
    if ball_y > pad_y + 3:   return DOWN, ball_y
    return NOOP, ball_y


# ── Contrat d'agent attendu par l'arene ────────────────────────────────────────
class Agent:
    """Robot pour le defi. Interface : reset() + act(frame 210x160x3) -> 2 (HAUT) | 3 (BAS)."""
    def __init__(self, weights_path: str = None):
        self.model = PolicyNet().to(DEVICE)
        if weights_path and os.path.exists(weights_path):
            ck = torch.load(weights_path, map_location=DEVICE, weights_only=False)
            state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
            self.model.load_state_dict(state)
        self.model.eval()
        self.prev = None

    def reset(self):
        self.prev = None

    @torch.no_grad()
    def act(self, frame: np.ndarray) -> int:
        cur = preprocess(frame)
        diff = cur - self.prev if self.prev is not None else np.zeros(D, np.float32)
        self.prev = cur
        prob, _ = self.model(torch.from_numpy(diff).unsqueeze(0).to(DEVICE))
        return UP if prob.item() > 0.5 else DOWN
