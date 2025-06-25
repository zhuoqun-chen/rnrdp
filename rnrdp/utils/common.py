import torch


def print_keys_recursively(d, indent=0):
    for k, v in d.items():
        print("\t" * indent, end="")
        if hasattr(v, "items"):
            print(k)
            print_keys_recursively(v, indent + 1)
        else:
            print(k)


# extracted from single file env for reuse
def to_tensor(x, device):
    if isinstance(x, dict):
        return {k: to_tensor(v, device) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)
