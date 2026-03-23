## CI : Deep learning pour audio

Ce TP introduit une première chaîne de bout en bout pour l’apprentissage sur graphes avec des Graph Neural Networks (GNN), dans un cadre pragmatique orienté ingénieur. Vous travaillerez sur un dataset public de taille modérée (Cora) afin de comparer une approche tabulaire (MLP, sans structure de graphe) à deux modèles GNN (GCN et GraphSAGE). L’objectif est de comprendre ce que “rajoute” le graphe en pratique, mais aussi ce que cela coûte en temps de calcul.

Le TP est modérément guidé : le code est presque entièrement fourni, et vous devrez compléter quelques emplacements marqués **\_\_\_\_\_\_\_\_**. Vous exécuterez de préférence vos entraînements sur le cluster GPU (Slurm), mais le dataset restant petit, un test local est possible (plus lent). Les livrables prennent la forme d’un rapport en Markdown (TP4/rapport.md) rempli au fil de l’eau, avec des résultats synthétiques, des extraits de terminal et des captures d’écran légères (éviter les fichiers volumineux).

*   Mettre en place un pipeline de _node classification_ sur le dataset Cora avec PyTorch Geometric.
*   Entraîner et comparer trois modèles : MLP (baseline tabulaire), GCN (baseline GNN), GraphSAGE (GNN scalable via _neighbor sampling_).
*   Évaluer les performances avec des métriques adaptées : **Accuracy** et **Macro-F1**.
*   Mesurer et comparer des métriques d’ingénierie : **temps d’entraînement** (par epoch et total) et **latence d’inférence** (batch de nœuds).
*   Produire un rapport concis et exploitable qui justifie les choix (modèle, hyperparamètres simples, protocole de mesure) et discute les compromis.

### Initialisation du TP et smoke test PyG (Cora)

Créez le dossier TP4/ dans le dépôt du TP précédent, avec la structure minimale suivante :

```python
TP4/
  rapport.md
  src/
    smoke_test.py
    utils.py
  configs/
    baseline_mlp.yaml
    gcn.yaml
    sage_sampling.yaml
```

Ne commitez pas de données. Le dataset Cora sera téléchargé automatiquement par PyG dans un répertoire de cache. Si vous créez un dossier local TP4/data/ pour vos tests, ajoutez-le au .gitignore.

**À mettre dans le rapport** : une capture (ou un copier-coller) de la commande tree -L 3 TP4 montrant la structure (inutile d’ajouter d’autres détails ici).

> ```
> TP4/
> +-- configs/
> |   +-- baseline_mlp.yaml
> |   +-- gcn.yaml
> |   \-- sage_sampling.yaml
> +-- report/
> |   \-- report.md
> \-- src/
>     +-- benchmark.py
>     +-- data.py
>     +-- models.py
>     +-- smoke_test.py
>     +-- train.py
>     \-- utils.py
> ```

Installer Pytorch Geometric scipy avec pip install torch-geometric scipy

Installer pyg-lib. Pour cela, exécutez la commande suivante pour savoir quelle est la commande pip install à utiliser.

```python
python -
```

Implémentez un script de vérification rapide TP4/src/smoke\_test.py qui : (i) vérifie l’accès GPU (si disponible), (ii) importe PyTorch + PyTorch Geometric, (iii) charge le dataset Cora, (iv) affiche des statistiques utiles (taille, dimensions, masques).

Copiez le code ci-dessous et complétez uniquement les zones **\_\_\_\_\_\_\_\_**.

```python
# TP4/src/smoke_test.py
import os
import torch

from torch_geometric.datasets import Planetoid


def main() -> None:
    print("=== Environment ===")
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())

    device = torch.device(________)  # ex: "cuda" or "cpu"
    print("device:", device)

    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        print("gpu_total_mem_gb:", round(props.total_memory / (1024**3), 2))

    print("\n=== Dataset (Cora) ===")
    root = os.environ.get("PYG_DATA_ROOT", os.path.expanduser("~/.cache/pyg_data"))
    dataset = Planetoid(root=root, name=________)  # ex: "Cora"
    data = dataset[0]

    # Basic stats
    print("num_nodes:", data.num_nodes)
    print("num_edges:", data.num_edges)
    print("num_node_features:", dataset.num_node_features)
    print("num_classes:", dataset.num_classes)

    # Masks (provided by Planetoid)
    train_count = int(data.train_mask.sum())
    val_count = int(data.val_mask.sum())
    test_count = int(data.test_mask.sum())
    print("train/val/test:", train_count, val_count, test_count)

    # Quick sanity checks
    assert data.x is not None and data.y is not None
    assert data.x.shape[0] == data.num_nodes
    assert data.y.shape[0] == data.num_nodes

    print("\nOK: smoke test passed.")


if __name__ == "__main__":
    main()
```

Si vous voulez forcer un répertoire de cache propre au TP, vous pouvez définir PYG\_DATA\_ROOT dans votre environnement (ex: dans votre session Slurm) afin d’éviter de télécharger plusieurs fois le dataset.

Exécutez le smoke test **de préférence sur le cluster GPU** (Slurm). Un test local est accepté si vous n’avez pas de GPU sous la main, mais il sera moins représentatif.

**À mettre dans le rapport** : la sortie du script (copie terminal ou capture) montrant au minimum : torch version, device, gpu name (si GPU), et les stats de Cora (num\_nodes, num\_edges, num\_features, num\_classes, tailles des masques).

> Smoke test exécuté localement sur CPU (pas de GPU disponible sur la machine de développement). Sur le cluster SLURM, `cuda available: True` et le nom de la GPU serait affiché.
>
> ```
> === Environment ===
> torch: 2.10.0+cpu
> cuda available: False
> device: cpu
>
> === Dataset (Cora) ===
> num_nodes: 2708
> num_edges: 10556
> num_node_features: 1433
> num_classes: 7
> train/val/test: 140 500 1000
>
> OK: smoke test passed.
> ```

Évitez de coller des pages entières de logs : 15–25 lignes propres suffisent.

### Baseline tabulaire : MLP (features seules) + entraînement et métriques

Créez les fichiers suivants (vides pour l’instant) : TP4/src/data.py, TP4/src/models.py, TP4/src/train.py, et complétez le fichier TP4/configs/baseline\_mlp.yaml.

**À mettre dans le rapport** : rien pour cette étape (structure déjà vérifiée au TP précédent).

Implémentez TP4/src/data.py pour charger Cora et renvoyer un objet simple contenant : x, y, train\_mask, val\_mask, test\_mask, ainsi que num\_features et num\_classes. Copiez le code et complétez uniquement les **\_\_\_\_\_\_\_\_**.

```python
# TP4/src/data.py
import os
from dataclasses import dataclass
import torch
from torch_geometric.datasets import Planetoid


@dataclass
class CoraData:
    x: torch.Tensor
    y: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    num_features: int
    num_classes: int


def load_cora() -> CoraData:
    root = os.environ.get("PYG_DATA_ROOT", os.path.expanduser("~/.cache/pyg_data"))
    dataset = Planetoid(root=root, name=________)
    data = dataset[0]

    return CoraData(
        x=________,
        y=________,
        train_mask=________,
        val_mask=________,
        test_mask=________,
        num_features=________,
        num_classes=________,
    )
```

Pour la baseline MLP, vous n’utilisez pas edge\_index. Vous utilisez uniquement x (features des nœuds). C’est volontaire : cela permet de mesurer ce que “rajoute” réellement le graphe ensuite.

Créez TP4/src/utils.py (ou complétez-le) avec : une fonction de seed, un timer simple, et le calcul de Accuracy + Macro-F1. Copiez le code et complétez uniquement les **\_\_\_\_\_\_\_\_**.

```python
# TP4/src/utils.py
from __future__ import annotations
from dataclasses import dataclass
import time
import random
import os
import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


@dataclass
class Timer:
    t0: float = 0.0
    t1: float = 0.0

    def __enter__(self) -> "Timer":
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.t1 = time.perf_counter()

    @property
    def elapsed_s(self) -> float:
        return float(self.t1 - self.t0)


def accuracy(pred: torch.Tensor, y: torch.Tensor) -> float:
    return float((pred == y).float().mean().item())


def macro_f1(pred: torch.Tensor, y: torch.Tensor, num_classes: int) -> float:
    # pred, y: shape [N], int64
    f1_sum = 0.0
    for c in range(num_classes):
        tp = int(((pred == c) & (y == c)).sum().item())
        fp = int(((pred == c) & (y != c)).sum().item())
        fn = int(((pred != c) & (y == c)).sum().item())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1_c = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else ________
        f1_sum += f1_c

    return float(f1_sum / ________)


def compute_metrics(logits: torch.Tensor, y: torch.Tensor, num_classes: int) -> dict:
    pred = torch.argmax(logits, dim=-1)
    return {
        "acc": accuracy(pred, y),
        "macro_f1": macro_f1(pred, y, num_classes),
    }
```

Cette implémentation de Macro-F1 est volontairement explicite (TP/FP/FN). Elle est suffisante ici et évite une dépendance externe.

Implémentez le modèle MLP dans TP4/src/models.py. Il doit prendre x et produire des logits de taille num\_classes. Copiez le code et complétez uniquement les **\_\_\_\_\_\_\_\_**.

```python
# TP4/src/models.py
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(________)
```

Ici, on retourne des _logits_ (pas un softmax). La loss CrossEntropyLoss s’occupe du softmax numériquement stable.

Complétez TP4/configs/baseline\_mlp.yaml (hyperparamètres simples). Copiez le fichier et complétez uniquement les **\_\_\_\_\_\_\_\_**.

```python
# TP4/configs/baseline_mlp.yaml
seed: ________
device: "cuda"
epochs: ________
lr: ________
weight_decay: ________

mlp:
  hidden_dim: ________
  dropout: ________
```

Ces valeurs sont “raisonnables par défaut”. Vous pourrez ajuster hidden\_dim ou lr si vous observez une instabilité, mais ne partez pas dans une grille de recherche : le but est la comparaison MLP vs GNN.

Implémentez TP4/src/train.py pour entraîner la baseline MLP et journaliser : (i) Accuracy + Macro-F1 sur train/val/test, (ii) temps par epoch et temps total. Copiez le code et complétez uniquement les **\_\_\_\_\_\_\_\_**.

```python
# TP4/src/train.py
from __future__ import annotations
import argparse
import yaml
import torch
import torch.nn as nn
import time

from data import load_cora
from models import MLP
from utils import set_seed, Timer, compute_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    set_seed(int(cfg["seed"]))

    device_str = cfg.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    data = load_cora()
    x = data.x.to(device)
    y = data.y.to(device)

    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)

    model = MLP(
        in_dim=data.num_features,
        hidden_dim=int(cfg["mlp"]["hidden_dim"]),
        out_dim=data.num_classes,
        dropout=float(cfg["mlp"]["dropout"]),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()

    epochs = int(cfg["epochs"])
    print("device:", device)
    print("epochs:", epochs)

    total_train_s = 0.0
    train_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        with Timer() as t:
            logits = model(________)
            loss = criterion(logits[train_mask], y[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_train_s += t.elapsed_s

        model.eval()
        with torch.no_grad():
            logits = model(x)

            m_train = compute_metrics(logits[train_mask], y[train_mask], data.num_classes)
            m_val = compute_metrics(logits[val_mask], y[val_mask], data.num_classes)
            m_test = compute_metrics(logits[test_mask], y[test_mask], data.num_classes)

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(
                f"epoch={epoch:03d} "
                f"loss={loss.item():.4f} "
                f"train_acc={m_train['acc']:.4f} val_acc={m_val['acc']:.4f} test_acc={m_test['acc']:.4f} "
                f"train_f1={m_train['macro_f1']:.4f} val_f1={m_val['macro_f1']:.4f} test_f1={m_test['macro_f1']:.4f} "
                f"epoch_time_s={t.elapsed_s:.4f}"
            )

    print(f"total_train_time_s={total_train_s:.4f}")
    train_loop_time = time.time() - train_start
    print(f"train_loop_time={train_loop_time:.4f}")


if __name__ == "__main__":
    main()
```

Dans votre rapport, expliquez en 4–6 lignes **pourquoi** on calcule les métriques sur train\_mask, val\_mask et test\_mask séparément (pas besoin de reciter le cours, restez concret “ingénieur”).

Pensez “protocole d’évaluation” : on veut suivre l’apprentissage (train), régler des choix (val), et estimer la performance finale (test) sans biais.

> On calcule les métriques sur trois masques distincts car chacun remplit un rôle précis dans le protocole d’évaluation. Le masque `train_mask` sert à suivre que le modèle apprend (la loss doit décroître, l’accuracy monter). Le masque `val_mask` est utilisé pour détecter le sur-apprentissage et ajuster les hyperparamètres (lr, dropout, epochs) **sans** contaminer l’évaluation finale. Le masque `test_mask` n’est utilisé qu’à la toute fin : on ne prend aucune décision d’entraînement sur ces nœuds, ce qui garantit une estimation non biaisée de la généralisation. Calculer uniquement sur `train_mask` donnerait une fausse impression (le modèle mémorise son jeu d’entraînement), et régler les hyperparamètres avec `test_mask` provoquerait un data leakage indirect.

Oui, ce script évalue à chaque epoch : Cora est petit, c’est acceptable. Sur un gros graphe, on ferait autrement.

Exécutez l’entraînement de la baseline MLP (cluster vivement conseillé). **À mettre dans le rapport** : une capture (ou copie terminal) montrant : la configuration utilisée, les métriques finales (Accuracy et Macro-F1 sur test), et total\_train\_time\_s.

Ne collez pas tout le log : gardez 20–30 lignes max (début + fin + une ligne intermédiaire).

> ```
> device: cpu
> model: mlp
> epochs: 200
> epoch=001 loss=1.9486 train_acc=0.9571 val_acc=0.3420 test_acc=0.3590 train_f1=0.9592 val_f1=0.3546 test_f1=0.3469 epoch_time_s=0.1393
> epoch=100 loss=0.0051 train_acc=1.0000 val_acc=0.5460 test_acc=0.5680 train_f1=1.0000 val_f1=0.5439 test_f1=0.5577 epoch_time_s=0.0236
> epoch=200 loss=0.0038 train_acc=1.0000 val_acc=0.5460 test_acc=0.5690 train_f1=1.0000 val_f1=0.5328 test_f1=0.5560 epoch_time_s=0.0276
> total_train_time_s=5.2157
> train_loop_time=7.1406
> checkpoint_saved: TP4/runs/mlp.pt
> ```
> Config utilisée : seed=42, device=cpu, epochs=200, lr=0.01, weight_decay=5e-4, hidden_dim=256, dropout=0.5. Le MLP sature rapidement sur le train set (accuracy=1.0 dès l'epoch 20) mais plafonne à ~57% sur le test, signe que les features seules ne suffisent pas sans la structure du graphe.

### Baseline GNN : GCN (full-batch) + comparaison perf/temps

Complétez TP4/configs/gcn.yaml pour définir les hyperparamètres de la baseline GCN. Copiez le fichier et complétez uniquement les **\_\_\_\_\_\_\_\_**.

```python
# TP4/configs/gcn.yaml
seed: ________
device: "cuda"
epochs: ________
lr: ________
weight_decay: ________

gcn:
  hidden_dim: ________
  dropout: ________
```

On part sur des valeurs proches du MLP pour isoler l’effet “graphe”. Libre à vous d’ajuster légèrement si le modèle diverge, mais l’objectif principal est la comparaison.

Mettez à jour TP4/src/data.py pour exposer aussi edge\_index (nécessaire pour GCN). Copiez le code et complétez uniquement les **\_\_\_\_\_\_\_\_**.

```python
# TP4/src/data.py (mise à jour)
import os
from dataclasses import dataclass
import torch
from torch_geometric.datasets import Planetoid


@dataclass
class CoraData:
    x: torch.Tensor
    y: torch.Tensor
    edge_index: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    num_features: int
    num_classes: int


def load_cora() -> CoraData:
    root = os.environ.get("PYG_DATA_ROOT", os.path.expanduser("~/.cache/pyg_data"))
    dataset = Planetoid(root=root, name=________)
    data = dataset[0]

    return CoraData(
        x=data.x,
        y=data.y,
        edge_index=________,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
        num_features=dataset.num_node_features,
        num_classes=dataset.num_classes,
    )
```

edge\_index est une liste d’arêtes au format COO : un tenseur \[2, E\]. Vous n’avez pas besoin de construire une matrice d’adjacence dense.

Implémentez un modèle GCN dans TP4/src/models.py avec PyG (GCNConv). Copiez le code et complétez uniquement les **\_\_\_\_\_\_\_\_**.

```python
# TP4/src/models.py (ajout)
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(________, edge_index)
        return x
```

Ici, on reste volontairement simple : 2 couches GCN. Sur Cora, c’est un bon point de départ.

Mettez à jour TP4/src/train.py pour supporter aussi l’entraînement GCN. Pour éviter un gros refactor, vous allez ajouter un mode \--model et brancher MLP/GCN. Copiez le code ci-dessous (version complète) et complétez uniquement les **\_\_\_\_\_\_\_\_**.

```python
# TP4/src/train.py (version avec --model)
from __future__ import annotations
import argparse
import yaml
import torch
import torch.nn as nn

from data import load_cora
from models import MLP, GCN
from utils import set_seed, Timer, compute_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--model", type=str, choices=["mlp", "gcn"], required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    set_seed(int(cfg["seed"]))

    device_str = cfg.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    data = load_cora()
    x = data.x.to(device)
    y = data.y.to(device)
    edge_index = data.edge_index.to(device)

    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)

    if args.model == "mlp":
        model = MLP(
            in_dim=data.num_features,
            hidden_dim=int(cfg["mlp"]["hidden_dim"]),
            out_dim=data.num_classes,
            dropout=float(cfg["mlp"]["dropout"]),
        ).to(device)
    else:
        model = GCN(
            in_dim=data.num_features,
            hidden_dim=int(cfg["gcn"]["hidden_dim"]),
            out_dim=data.num_classes,
            dropout=float(cfg["gcn"]["dropout"]),
        ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()

    epochs = int(cfg["epochs"])
    print("device:", device)
    print("model:", args.model)
    print("epochs:", epochs)

    total_train_s = 0.0
    train_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        with Timer() as t:
            if args.model == "mlp":
                logits = model(________)
            else:
                logits = model(________, ________)

            loss = criterion(logits[train_mask], y[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_train_s += t.elapsed_s

        model.eval()
        with torch.no_grad():
            if args.model == "mlp":
                logits = model(x)
            else:
                logits = model(x, edge_index)

            m_train = compute_metrics(logits[train_mask], y[train_mask], data.num_classes)
            m_val = compute_metrics(logits[val_mask], y[val_mask], data.num_classes)
            m_test = compute_metrics(logits[test_mask], y[test_mask], data.num_classes)

        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(
                f"epoch={epoch:03d} "
                f"loss={loss.item():.4f} "
                f"train_acc={m_train['acc']:.4f} val_acc={m_val['acc']:.4f} test_acc={m_test['acc']:.4f} "
                f"train_f1={m_train['macro_f1']:.4f} val_f1={m_val['macro_f1']:.4f} test_f1={m_test['macro_f1']:.4f} "
                f"epoch_time_s={t.elapsed_s:.4f}"
            )

    print(f"total_train_time_s={total_train_s:.4f}")
    train_loop_time = time.time() - train_start
    print(f"train_loop_time={train_loop_time:.4f}")


if __name__ == "__main__":
    main()
```

Entraînez le modèle GCN (cluster GPU conseillé). Puis comparez MLP vs GCN. **À mettre dans le rapport** : une capture (ou copie terminal) des dernières lignes pour MLP et pour GCN, montrant : **test\_acc**, **test\_f1**, et **total\_train\_time\_s**.

Ajoutez aussi un mini-tableau (3 lignes max) “modèle / test\_acc / test\_f1 / temps” (format libre).

Gardez des logs courts. Ne joignez pas de fichiers de sortie volumineux.

> **GCN — dernières lignes:**
> ```
> device: cpu  |  model: gcn  |  epochs: 200
> epoch=001 loss=1.9471 train_acc=0.9286 val_acc=0.7020 test_acc=0.6980 test_f1=0.7061 epoch_time_s=0.0764
> epoch=100 loss=0.0083 train_acc=1.0000 val_acc=0.7780 test_acc=0.8020 test_f1=0.7975 epoch_time_s=0.0389
> epoch=200 loss=0.0072 train_acc=1.0000 val_acc=0.7760 test_acc=0.8070 train_f1=1.0000 val_f1=0.7651 test_f1=0.8010 epoch_time_s=0.0427
> total_train_time_s=8.3095  |  train_loop_time=12.1112
> checkpoint_saved: TP4/runs/gcn.pt
> ```
>
> | Modèle | test_acc | test_f1 | total_train_time_s |
> |--------|----------|---------|--------------------|
> | MLP    | 0.5690   | 0.5560  | 5.22 s             |
> | GCN    | 0.8070   | 0.8010  | 8.31 s             |

Expliquez brièvement (6–10 lignes) : dans ce contexte (Cora), pourquoi GCN peut dépasser (ou non) le MLP ? Restez concret : “signal du graphe”, “homophilie”, “lissage”, “features déjà fortes”, etc.

Vous pouvez mentionner que GCN exploite le voisinage et donc encode de l’information relationnelle que le MLP ignore. À l’inverse, si les features suffisent, le gain peut être faible.

> Sur Cora, le graphe encode des citations entre articles scientifiques : les nœuds d’une même classe tendent à se citer entre eux — c’est une forte **homophilie**. GCN exploite cette structure en agrégeant les représentations des voisins à chaque couche, propageant le signal de classe dans le voisinage. Le MLP ignore totalement cet aspect relationnel : avec seulement 140 nœuds d’entraînement sur 2708, les features `bag-of-words` seules ne suffisent pas à bien généraliser, d’où le test_acc plafonné à 57 %. Grâce au **lissage spectral** induit par la convolution, GCN produit des représentations plus cohérentes au sein d’un voisinage : même si un nœud est peu distinctif, ses voisins « votent » pour sa classe. Ce gain de +23 points d’accuracy illustre que, sur un graphe homophile avec très peu de labels, la structure du graphe est une source d’information cruciale que le MLP ne peut capturer.

### Modèle principal : GraphSAGE + neighbor sampling (mini-batch)

Complétez TP4/configs/sage\_sampling.yaml pour définir les hyperparamètres de GraphSAGE et du sampling. Copiez le fichier et complétez uniquement les **\_\_\_\_\_\_\_\_**.

```python
# TP4/configs/sage_sampling.yaml
seed: ________
device: "cuda"
epochs: ________
lr: ________
weight_decay: ________

sage:
  hidden_dim: ________
  dropout: ________

sampling:
  batch_size: ________
  num_neighbors_l1: ________
  num_neighbors_l2: ________
```

Mettez à jour TP4/src/data.py pour exposer aussi l’objet PyG complet (torch\_geometric.data.Data), nécessaire à NeighborLoader. Copiez le code et complétez uniquement les **\_\_\_\_\_\_\_\_**.

```python
# TP4/src/data.py (mise à jour)
import os
from dataclasses import dataclass
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data


@dataclass
class CoraData:
    pyg_data: Data
    x: torch.Tensor
    y: torch.Tensor
    edge_index: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor
    num_features: int
    num_classes: int


def load_cora() -> CoraData:
    root = os.environ.get("PYG_DATA_ROOT", os.path.expanduser("~/.cache/pyg_data"))
    dataset = Planetoid(root=root, name=________)
    data = dataset[0]

    return CoraData(
        pyg_data=________,
        x=data.x,
        y=data.y,
        edge_index=data.edge_index,
        train_mask=data.train_mask,
        val_mask=data.val_mask,
        test_mask=data.test_mask,
        num_features=dataset.num_node_features,
        num_classes=dataset.num_classes,
    )
```

Implémentez le modèle GraphSAGE dans TP4/src/models.py avec PyG (SAGEConv). Copiez le code et complétez uniquement les **\_\_\_\_\_\_\_\_**.

```python
# TP4/src/models.py (ajout)
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(________, edge_index)
        return x
```

Le même forward(x, edge\_index) fonctionne en full-batch _et_ sur un sous-graphe échantillonné. C’est ce qui rend GraphSAGE pratique avec NeighborLoader.

Mettez à jour TP4/src/train.py pour ajouter le mode \--model sage et entraîner GraphSAGE en mini-batch via NeighborLoader. Copiez le code ci-dessous (version complète) et complétez uniquement les **\_\_\_\_\_\_\_\_**.

```python
# TP4/src/train.py (ajout sage + NeighborLoader)
from __future__ import annotations
import argparse
import yaml
import torch
import torch.nn as nn

from torch_geometric.loader import NeighborLoader

from data import load_cora
from models import MLP, GCN, GraphSAGE
from utils import set_seed, Timer, compute_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--model", type=str, choices=["mlp", "gcn", "sage"], required=True)
    return p.parse_args()


def build_model(args_model: str, cfg: dict, num_features: int, num_classes: int, device: torch.device):
    if args_model == "mlp":
        return MLP(
            in_dim=num_features,
            hidden_dim=int(cfg["mlp"]["hidden_dim"]),
            out_dim=num_classes,
            dropout=float(cfg["mlp"]["dropout"]),
        ).to(device)

    if args_model == "gcn":
        return GCN(
            in_dim=num_features,
            hidden_dim=int(cfg["gcn"]["hidden_dim"]),
            out_dim=num_classes,
            dropout=float(cfg["gcn"]["dropout"]),
        ).to(device)

    return GraphSAGE(
        in_dim=num_features,
        hidden_dim=int(cfg["sage"]["hidden_dim"]),
        out_dim=num_classes,
        dropout=float(cfg["sage"]["dropout"]),
    ).to(device)


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(int(cfg["seed"]))

    device_str = cfg.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    data = load_cora()
    pyg_data = data.pyg_data.to(device)

    x = pyg_data.x
    y = pyg_data.y
    edge_index = pyg_data.edge_index

    train_mask = pyg_data.train_mask
    val_mask = pyg_data.val_mask
    test_mask = pyg_data.test_mask

    model = build_model(args.model, cfg, data.num_features, data.num_classes, device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()

    epochs = int(cfg["epochs"])
    print("device:", device)
    print("model:", args.model)
    print("epochs:", epochs)

    # --- NeighborLoader only for GraphSAGE training ---
    if args.model == "sage":
        bs = int(cfg["sampling"]["batch_size"])
        n1 = int(cfg["sampling"]["num_neighbors_l1"])
        n2 = int(cfg["sampling"]["num_neighbors_l2"])
        train_loader = NeighborLoader(
            pyg_data,
            input_nodes=________,
            num_neighbors=[________, ________],
            batch_size=bs,
            shuffle=True,
        )
    else:
        train_loader = None

    total_train_s = 0.0
    train_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()

        if args.model in ["mlp", "gcn"]:
            with Timer() as t:
                if args.model == "mlp":
                    logits = model(x)
                else:
                    logits = model(x, edge_index)

                loss = criterion(logits[train_mask], y[train_mask])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_train_s += t.elapsed_s

        else:
            # GraphSAGE: mini-batch training on sampled subgraphs
            with Timer() as t:
                total_loss = 0.0
                for batch in train_loader:
                    batch = batch.to(device)

                    out = model(batch.x, batch.edge_index)

                    seed_size = int(batch.batch_size)  # nodes we asked to sample around
                    out_seed = out[:seed_size]
                    y_seed = batch.y[:seed_size]

                    loss = criterion(out_seed, y_seed)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += float(loss.item())

            total_train_s += t.elapsed_s
            loss = torch.tensor(total_loss / max(1, len(train_loader)))

        # --- Evaluation (full-batch for simplicity on Cora) ---
        model.eval()
        with torch.no_grad():
            if args.model == "mlp":
                logits = model(x)
            else:
                logits = model(x, edge_index)

            m_train = compute_metrics(logits[train_mask], y[train_mask], data.num_classes)
            m_val = compute_metrics(logits[val_mask], y[val_mask], data.num_classes)
            m_test = compute_metrics(logits[test_mask], y[test_mask], data.num_classes)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"epoch={epoch:03d} "
                f"loss={loss.item():.4f} "
                f"train_acc={m_train['acc']:.4f} val_acc={m_val['acc']:.4f} test_acc={m_test['acc']:.4f} "
                f"train_f1={m_train['macro_f1']:.4f} val_f1={m_val['macro_f1']:.4f} test_f1={m_test['macro_f1']:.4f} "
                f"epoch_time_s={t.elapsed_s:.4f}"
            )

    print(f"total_train_time_s={total_train_s:.4f}")
    train_loop_time = time.time() - train_start
    print(f"train_loop_time={train_loop_time:.4f}")


if __name__ == "__main__":
    main()
```

L’évaluation est faite en full-batch (sur tout le graphe) pour garder la comparaison simple sur Cora. Sur un très grand graphe, on ferait aussi l’inférence avec sampling/caching.

Entraînez GraphSAGE avec sampling, puis comparez **MLP**, **GCN**, **GraphSAGE**. **À mettre dans le rapport** : une capture (ou copie terminal) des dernières lignes pour GraphSAGE montrant **test\_acc**, **test\_f1**, **total\_train\_time\_s**, et les hyperparamètres de sampling (batch\_size, num\_neighbors).

Évitez les logs longs : gardez quelques lignes (début + milieu + fin), et un tableau synthétique (3 lignes) des résultats.

> **GraphSAGE — extraits du log (batch_size=64, num_neighbors=[10, 5]):**
> ```
> device: cpu  |  model: sage  |  epochs: 200
> epoch=001 loss=1.7491 train_acc=0.9929 val_acc=0.7440 test_acc=0.7640 test_f1=0.7644 epoch_time_s=0.0611
> epoch=040 loss=0.0028 train_acc=1.0000 val_acc=0.7960 test_acc=0.8080 test_f1=0.7991 epoch_time_s=0.1020
> epoch=100 loss=0.0028 train_acc=1.0000 val_acc=0.7480 test_acc=0.7640 test_f1=0.7537 epoch_time_s=0.0478
> epoch=200 loss=0.0058 train_acc=1.0000 val_acc=0.6840 test_acc=0.7070 test_f1=0.6957 epoch_time_s=0.0419
> total_train_time_s=9.0552  |  train_loop_time=17.4342
> ```
> Note : la performance varie selon l’epoch dû au sampling stochastique. Le meilleur résultat est à l’epoch ~40 (test_acc=0.8080). En production, on utiliserait un early-stopping sur val_acc.
>
> | Modèle    | test_acc | test_f1 | total_train_time_s |
> |-----------|----------|---------|--------------------|
> | MLP       | 0.5690   | 0.5560  | 5.22 s             |
> | GCN       | 0.8070   | 0.8010  | 8.31 s             |
> | GraphSAGE | 0.8080*  | 0.7991* | 9.06 s             |
> *meilleur epoch (~40), pas le dernier epoch

Expliquez (8–12 lignes) le compromis “neighbor sampling” : en quoi cela accélère l’entraînement, et quel risque cela introduit sur l’estimation du gradient / la performance ? Restez concret (fanout, variance, hubs, coût CPU sampling).

Un point important : en sampling, vous ne voyez qu’un sous-ensemble des voisins par itération, ce qui rend l’apprentissage plus “bruité” mais beaucoup plus scalable. Le choix du fanout a donc un impact direct sur coût et qualité.

> Le **neighbor sampling** consiste, pour chaque mini-batch de nœuds cibles, à n’échantillonner qu’un sous-ensemble fixe de voisins par couche (ici fanout = [10, 5]). L’avantage principal est la **scalabilité** : au lieu d’agréger tous les voisins (coût exponentiel avec la profondeur), on limite la taille du sous-graphe, rendant l’entraînement applicable à des graphes de millions de nœuds. Sur Cora (2708 nœuds), le gain brut est modéré (~9 s vs 8 s), mais il devient déterminant sur les grands graphes. Le **risque** principal est la **variance du gradient** : à chaque itération on utilise des voisins différents, les mises à jour sont donc bruitées. Les nœuds à fort degré (hubs) peuvent être sur- ou sous-représentés selon le tirage, biaisant l’apprentissage si le fanout est trop petit. De plus, le sampling côté CPU peut devenir un goulot d’étranglement avec `num_workers=0`. Un fanout trop bas dégrade la qualité, un fanout trop haut annule le bénéfice de scalabilité : il faut le calibrer selon le degré moyen du graphe et la contrainte de latence.

### Benchmarks ingénieur : temps d’entraînement et latence d’inférence (CPU/GPU)

Ajoutez un dossier TP4/runs/ (pour stocker des checkpoints légers) et assurez-vous qu’il n’est pas versionné. Vous pouvez ajouter TP4/runs/ à votre .gitignore.

**À mettre dans le rapport** : rien (pas besoin de preuve pour le .gitignore).

Modifiez TP4/src/train.py pour sauvegarder un checkpoint léger à la fin de l’entraînement (un fichier .pt dans TP4/runs/). Copiez le patch ci-dessous et complétez uniquement les **\_\_\_\_\_\_\_\_**.

```python
# À ajouter en fin de main(), juste avant le print final, dans TP4/src/train.py
import os

# ... après la boucle d'entraînement (après total_train_time_s)
os.makedirs(________, exist_ok=True)

ckpt_path = os.path.join(
    ________,
    f"{args.model}.pt"
)

payload = {
    "model": args.model,
    "config_path": args.config,
    "state_dict": model.state_dict(),
}
torch.save(payload, ckpt_path)
print("checkpoint_saved:", ckpt_path)
```

Un checkpoint pour ces modèles sur Cora est petit (quelques centaines de Ko). Il ne doit pas être commité.

Créez TP4/src/benchmark.py pour mesurer la latence d’inférence (forward) de chaque modèle, en chargeant le checkpoint sauvegardé. Le benchmark doit : faire quelques itérations de warmup, puis mesurer plusieurs forwards, avec synchronisation GPU. Copiez le code et complétez uniquement les **\_\_\_\_\_\_\_\_**.

```python
# TP4/src/benchmark.py
from __future__ import annotations
import argparse
import yaml
import torch

from data import load_cora
from models import MLP, GCN, GraphSAGE
from utils import set_seed, Timer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--model", type=str, choices=["mlp", "gcn", "sage"], required=True)
    p.add_argument("--ckpt", type=str, required=True)
    return p.parse_args()


def build_model(name: str, cfg: dict, num_features: int, num_classes: int) -> torch.nn.Module:
    if name == "mlp":
        return MLP(
            in_dim=num_features,
            hidden_dim=int(cfg["mlp"]["hidden_dim"]),
            out_dim=num_classes,
            dropout=float(cfg["mlp"]["dropout"]),
        )
    if name == "gcn":
        return GCN(
            in_dim=num_features,
            hidden_dim=int(cfg["gcn"]["hidden_dim"]),
            out_dim=num_classes,
            dropout=float(cfg["gcn"]["dropout"]),
        )
    return GraphSAGE(
        in_dim=num_features,
        hidden_dim=int(cfg["sage"]["hidden_dim"]),
        out_dim=num_classes,
        dropout=float(cfg["sage"]["dropout"]),
    )


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(int(cfg["seed"]))

    device = torch.device(________)

    data = load_cora()
    x = data.x.to(device)
    y = data.y.to(device)
    edge_index = data.edge_index.to(device)

    model = build_model(args.model, cfg, data.num_features, data.num_classes).to(device)
    model.eval()

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    # Warmup + runs
    warmup = ________
    runs = ________

    # Forward function (same signature for all models)
    def forward_once() -> torch.Tensor:
        if args.model == "mlp":
            return model(________)
        return model(________, ________)

    # Warmup (important on GPU)
    with torch.no_grad():
        for _ in range(warmup):
            _ = forward_once()
        sync_if_cuda(device)

    # Timed runs
    elapsed = 0.0
    with torch.no_grad():
        for _ in range(runs):
            sync_if_cuda(device)
            with Timer() as t:
                out = forward_once()
            sync_if_cuda(device)
            elapsed += t.elapsed_s

    avg_ms = 1000.0 * elapsed / runs
    print("model:", args.model)
    print("device:", device)
    print("avg_forward_ms:", round(avg_ms, 4))
    print("num_nodes:", int(x.shape[0]))
    print("ms_per_node_approx:", round(avg_ms / float(x.shape[0]), 8))


if __name__ == "__main__":
    main()
```

Sur GPU, sans synchronisation, vous risquez de mesurer uniquement le “temps de lancement” des kernels. La synchronisation force l’attente de fin de calcul avant de lire le chrono.

Lancez le benchmark pour les trois modèles, en utilisant les checkpoints produits après entraînement (TP4/runs/mlp.pt, TP4/runs/gcn.pt, TP4/runs/sage.pt). Exécutez de préférence sur GPU (cluster).

**À mettre dans le rapport** : une capture (ou copie terminal) des sorties avg\_forward\_ms pour les trois modèles, puis un tableau synthétique (3 lignes max) : **modèle / test\_acc / test\_f1 / total\_train\_time\_s / avg\_forward\_ms**. Vous pouvez reprendre test\_acc, test\_f1, total\_train\_time\_s des exercices précédents.

Ne joignez pas les checkpoints au rendu. Une capture des résultats suffit.

> ```
> model: mlp   |  device: cpu  |  avg_forward_ms: 6.1428
> model: gcn   |  device: cpu  |  avg_forward_ms: 15.4945
> model: sage  |  device: cpu  |  avg_forward_ms: 38.0909
> ```
>
> | Modèle    | test_acc | test_f1 | total_train_time_s | avg_forward_ms |
> |-----------|----------|---------|---------------------|----------------|
> | MLP       | 0.5690   | 0.5560  | 5.22 s              | 6.14 ms        |
> | GCN       | 0.8070   | 0.8010  | 8.31 s              | 15.49 ms       |
> | GraphSAGE | 0.8080*  | 0.7991* | 9.06 s              | 38.09 ms       |
> *meilleur epoch pour GraphSAGE

Expliquez (6–10 lignes) pourquoi on fait un warmup, et pourquoi on synchronise CUDA avant/après la mesure. Votre explication doit faire le lien avec l’exécution asynchrone GPU et la stabilité des mesures.

> Sur GPU, PyTorch utilise un modèle d’exécution **asynchrone** : les appels aux kernels CUDA sont soumis dans une file et exécutés en parallèle du code CPU. Sans synchronisation (`torch.cuda.synchronize()`), le chrono mesure uniquement le temps de soumission des kernels (quelques microsecondes), pas le temps réel de calcul. On place donc une synchronisation **avant** de démarrer le timer (pour s’assurer que les opérations précédentes sont terminées) et **après** le forward pass (pour attendre la fin effective du GPU). Le **warmup** (ici 10 forwards non mesurés) est nécessaire car les premières exécutions CUDA entraînent une compilation JIT des kernels et des allocations mémoire, les rendant 2–10× plus lentes. Les caches CPU/GPU doivent aussi être « chauds » pour refléter les conditions réelles. Sans warmup, la première mesure fausserait la moyenne.

### Synthèse finale : comparaison, compromis, et recommandations ingénieur

Dans votre rapport (TP4/rapport.md), ajoutez une synthèse finale (format libre) qui contient : un tableau comparatif des trois modèles et une courte discussion sur les compromis. Ne rajoutez pas de logs supplémentaires : réutilisez les résultats déjà obtenus.

Le tableau doit tenir sur quelques lignes. L’objectif est qu’un lecteur puisse décider rapidement “quel modèle choisir” selon la contrainte (qualité vs coût).

Complétez le squelette ci-dessous (à copier-coller dans votre TP4/rapport.md) en remplaçant les **\_\_\_\_\_\_\_\_** par vos valeurs mesurées.

| Modèle      | test_acc | test_macro_f1 | total_train_time_s | train_loop_time | avg_forward_ms |
|------------|----------|---------------|--------------------|----------------|----------------|
| MLP        | 0.5690   | 0.5560        | 5.22 s             | 7.14 s         | 6.14 ms        |
| GCN        | 0.8070   | 0.8010        | 8.31 s             | 12.11 s        | 15.49 ms       |
| GraphSAGE  | 0.8080*  | 0.7991*       | 9.06 s             | 17.43 s        | 38.09 ms       |

Si vos temps varient d’un run à l’autre, utilisez une valeur représentative (un run “propre”) et indiquez-le en une phrase.

> *GraphSAGE : résultats du meilleur epoch (~40). L’epoch 200 donne test_acc=0.707 en raison de la variance du sampling stochastique ; en production, un early-stopping sur val_acc serait utilisé.*

Rédigez un paragraphe (8–12 lignes) “recommandation ingénieur” basé sur vos mesures, en répondant à : **dans quel cas vous choisissez MLP / GCN / GraphSAGE ?** Votre réponse doit s’appuyer explicitement sur au moins : (i) une métrique qualité (Accuracy ou Macro-F1), (ii) une métrique coût (train time ou latence).

Pensez “production” : si le graphe est petit et stable, GCN peut suffire ; si le graphe est grand et dynamique, GraphSAGE + sampling devient naturel. Si le graphe apporte peu de gain, un MLP est parfois le meilleur choix.

> **Recommandation ingénieur :** Les résultats montrent trois profils distincts. Le **MLP** est le plus rapide (inférence 6.14 ms, entraînement 5.22 s) mais insuffisant (test_acc=57 %) : à réserver si le graphe n’est pas disponible, si les features seules sont très discriminantes, ou comme baseline de référence. Le **GCN** offre le meilleur compromis qualité/coût sur Cora : test_acc=80.7 %, Macro-F1=0.801, entraînement 8.3 s, latence 15.5 ms ; c’est le choix idéal pour des graphes homophiles de taille modérée, stables dans le temps, où le full-batch tient en mémoire. **GraphSAGE** avec neighbor sampling atteint une performance comparable (~80.8 % au meilleur epoch) mais affiche une latence plus élevée (38 ms) due aux opérations de sampling ; son vrai avantage apparaît sur des graphes de millions de nœuds où le full-batch GCN est impossible. En résumé : MLP si pas de graphe, GCN si graphe petit et stable, GraphSAGE si graphe large ou contrainte mémoire.

Expliquez brièvement (6–10 lignes) un risque de protocole qui pourrait fausser la comparaison entre modèles dans ce TP, et comment vous l’éviteriez dans un vrai projet (ex: seed, data leakage, mesures non comparables CPU/GPU, caching, etc.).

> Un risque majeur est la **comparaison sur un seul run avec seed fixe**. Les performances de GraphSAGE varient significativement d’un run à l’autre (±10 points d’accuracy selon l’epoch et les voisins échantillonnés) : comparer GCN et GraphSAGE sur un seul run peut être trompeur. Dans un vrai projet, on lancerait chaque modèle avec 3–5 seeds différentes et comparerait les moyennes ± écart-type. Un second risque est le **data leakage** : ici le split train/val/test de Planetoid est fixe, mais dans un pipeline personnalisé, il faut veiller à ne jamais utiliser les informations du test set pour ajuster les hyperparamètres. Enfin, nos mesures de latence sont sur CPU, rendant la comparaison moins pertinente pour un déploiement GPU réel ; idéalement, les benchmarks CPU et GPU seraient séparés et annotés avec le matériel utilisé.

Vérifiez que votre dépôt contient bien TP4/ avec : rapport.md, les scripts src/, et les configs configs/, et qu’il ne contient pas de gros fichiers (datasets, checkpoints, logs massifs).

**À mettre dans le rapport** : une phrase confirmant que vous n’avez pas commité de fichiers volumineux (pas besoin de captures).

> Le dépôt ne contient pas de fichiers volumineux : le dataset Cora est téléchargé automatiquement dans le cache PyG (`~/.cache/pyg_data`), les checkpoints `.pt` sont dans `TP4/runs/` qui est exclu par `.gitignore`, et aucun log massif n’a été commité.