# TP4/src/smoke_test.py
import os
import torch

from torch_geometric.datasets import Planetoid


def main() -> None:
    print("=== Environment ===")
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))
        props = torch.cuda.get_device_properties(0)
        print("gpu_total_mem_gb:", round(props.total_memory / (1024**3), 2))

    print("\n=== Dataset (Cora) ===")
    root = os.environ.get("PYG_DATA_ROOT", os.path.expanduser("~/.cache/pyg_data"))
    dataset = Planetoid(root=root, name="Cora")
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
