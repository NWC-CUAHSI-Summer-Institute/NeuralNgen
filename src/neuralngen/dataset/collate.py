# ./src/neuralngen/dataset/collate.py

from torch.utils.data.dataloader import default_collate

def custom_collate(batch):
    """
    Custom collate_fn to preserve x_info as list of dicts.
    """
    # Separate all keys
    batch_x_info = [sample["x_info"] for sample in batch]

    # Use default_collate for the tensors
    batch_collated = default_collate([
        {k: v for k, v in sample.items() if k != "x_info"}
        for sample in batch
    ])

    batch_collated["x_info"] = batch_x_info
    return batch_collated
