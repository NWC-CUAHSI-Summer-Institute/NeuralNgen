import numpy as np
import torch
from collections import defaultdict, deque, Counter
import random
from typing import List, Dict, Union, Tuple, Any, Optional
import matplotlib.pyplot as plt

# collate_fn adapted for the (B, T, S, F) output structure
def collate_fn(
    samples_for_batch: List[Dict[str, Any]], # Each item here is a prepared spatial slot dict (with lists of S-length tensors)
    temporal_batch_length: int,             # self.temporal_length_batcher from the batcher
    sequence_length_input: int,             # self.sequence_length_input from the batcher (e.g., 24h, this is D)
    xs_feature_length: int                  # New: A, the feature length for x_s when it's not D-length
) -> Dict[str, Union[torch.Tensor, np.ndarray, List[str], List[float]]]:
    """
    Collates a list of prepared spatial slot dictionaries into a single batch dictionary.
    Temporal data (x_d, y, date) is stacked to (B, T, S, F) or (B, T, S).
    Semi-temporal data (x_s) is stacked to (B, T, A).
    Static data is stacked to (B, F_static).

    Args:
        samples_for_batch (List[Dict]): List of dictionaries, each representing a single
                                         spatial slot. Temporal features (x_d, y, date) as lists of
                                         (S, F) or (S,) tensors/arrays, and x_s as a list of (A,) tensors.
                                         Static features as single values.
        temporal_batch_length (int): The 'T' dimension for the output batch.
        sequence_length_input (int): The 'S' dimension for the output batch (D).
        xs_feature_length (int): The 'A' dimension for the output x_s feature.

    Returns:
        Dict: A single dictionary representing the collated batch.
    """
    batch: Dict[str, Union[torch.Tensor, np.ndarray, List[str], List[float]]] = {}
    if not samples_for_batch:
        return batch

    # Get the first sample to determine the ACTUAL key names for this batch
    first_spatial_slot_data = samples_for_batch[0]
    
    # --- Collating Static/Metadata Features ---
    # These are directly extracted as lists and will remain lists in the batch dict
    actual_catchment_id_key = next((k for k in first_spatial_slot_data if k.startswith('gauge') and isinstance(first_spatial_slot_data[k], str)), None)
    # MODIFIED: Allow 'Cluster' to be float, int, or str
    actual_category_key = next((k for k in first_spatial_slot_data if k.startswith('Cluster') and isinstance(first_spatial_slot_data[k], (float, int, str))), None)
    actual_lat_key = next((k for k in first_spatial_slot_data if k.startswith('lat') and isinstance(first_spatial_slot_data[k], (float, int))), None)
    actual_lon_key = next((k for k in first_spatial_slot_data if k.startswith('lon') and isinstance(first_spatial_slot_data[k], (float, int))), None)

    # Ensure all required static keys were found
    if actual_catchment_id_key is None:
        raise ValueError(f"collate_fn: Could not find key starting with 'gauge' for catchment ID in first sample: {first_spatial_slot_data.keys()}")
    if actual_category_key is None:
        raise ValueError(f"collate_fn: Could not find key starting with 'Cluster' for category in first sample: {first_spatial_slot_data.keys()}")
    if actual_lat_key is None:
        raise ValueError(f"collate_fn: Could not find key starting with 'lat' for latitude in first sample: {first_spatial_slot_data.keys()}")
    if actual_lon_key is None:
        raise ValueError(f"collate_fn: Could not find key starting with 'lon' for longitude in first sample: {first_spatial_slot_data.keys()}")
    
    batch[actual_catchment_id_key] = [s[actual_catchment_id_key] for s in samples_for_batch]
    batch[actual_category_key] = [s[actual_category_key] for s in samples_for_batch]
    batch[actual_lat_key] = [s[actual_lat_key] for s in samples_for_batch]
    batch[actual_lon_key] = [s[actual_lon_key] for s in samples_for_batch]
    
    # --- Collating Temporal Features ---
    actual_date_key = next((k for k in first_spatial_slot_data if k.startswith('date') and isinstance(first_spatial_slot_data[k], list) and (not first_spatial_slot_data[k] or isinstance(first_spatial_slot_data[k][0], np.ndarray))), None)
    actual_xd_key = next((k for k in first_spatial_slot_data if k.startswith('x_d') and isinstance(first_spatial_slot_data[k], list) and (not first_spatial_slot_data[k] or isinstance(first_spatial_slot_data[k][0], torch.Tensor))), None)
    actual_y_key = next((k for k in first_spatial_slot_data if k.startswith('y') and isinstance(first_spatial_slot_data[k], list) and (not first_spatial_slot_data[k] or isinstance(first_spatial_slot_data[k][0], torch.Tensor))), None)
    actual_xs_temporal_key = next((k for k in first_spatial_slot_data if k.startswith('x_s') and isinstance(first_spatial_slot_data[k], list) and (not first_spatial_slot_data[k] or isinstance(first_spatial_slot_data[k][0], torch.Tensor))), None) # x_s is here!

    if actual_date_key is None:
        raise ValueError(f"collate_fn: Could not find key starting with 'date' for temporal dates in first sample: {first_spatial_slot_data.keys()}")
    if actual_xd_key is None:
        raise ValueError(f"collate_fn: Could not find key starting with 'x_d' for dynamic features in first sample: {first_spatial_slot_data.keys()}")
    if actual_y_key is None:
        raise ValueError(f"collate_fn: Could not find key starting with 'y' for target in first sample: {first_spatial_slot_data.keys()}")
    if actual_xs_temporal_key is None:
        raise ValueError(f"collate_fn: Could not find key starting with 'x_s' for (semi-temporal) features in first sample: {first_spatial_slot_data.keys()}")

    temporal_features_to_process = {
        'date': actual_date_key,
        'x_d': actual_xd_key,
        'y': actual_y_key,
        'x_s': actual_xs_temporal_key # x_s is back in temporal processing
    }

    # Determine the maximum actual temporal length (T) that occurred in any spatial slot
    max_actual_T_len = 0
    for s_data in samples_for_batch:
        if actual_xd_key in s_data and s_data[actual_xd_key]:
            max_actual_T_len = max(max_actual_T_len, len(s_data[actual_xd_key]))
    
    # If max_actual_T_len is 0, it means all temporal lists were empty.
    if max_actual_T_len == 0:
        # Define default padded snippets for each type based on known lengths (S or A)
        padded_date_snippet = np.full((sequence_length_input,), np.datetime64('NaT'), dtype='datetime64[ns]')
        
        # Get example tensors for dtype and shape inference, robustly handling empty lists
        example_xd_tensor = None
        example_y_tensor = None
        example_xs_tensor = None # for x_s

        for s_data in samples_for_batch:
            if not example_xd_tensor and actual_xd_key in s_data and s_data[actual_xd_key]:
                example_xd_tensor = s_data[actual_xd_key][0]
            if not example_y_tensor and actual_y_key in s_data and s_data[actual_y_key]:
                example_y_tensor = s_data[actual_y_key][0]
            if not example_xs_tensor and actual_xs_temporal_key in s_data and s_data[actual_xs_temporal_key]:
                example_xs_tensor = s_data[actual_xs_temporal_key][0]
            if example_xd_tensor is not None and example_y_tensor is not None and example_xs_tensor is not None:
                break
        
        # If no example tensor found, use default shapes/dtypes, otherwise infer from examples
        # Note: x_d and y use sequence_length_input (D). x_s uses xs_feature_length (A).
        xd_shape = (sequence_length_input, example_xd_tensor.shape[1]) if example_xd_tensor is not None and example_xd_tensor.ndim > 1 else (sequence_length_input, 1)
        y_shape = (sequence_length_input,)
        xs_shape = (xs_feature_length,) # x_s has shape (A,)
        
        padded_xd_snippet = torch.full(xd_shape, float('nan'), dtype=example_xd_tensor.dtype if example_xd_tensor is not None else torch.float32)
        padded_y_snippet = torch.full(y_shape, float('nan'), dtype=example_y_tensor.dtype if example_y_tensor is not None else torch.float32)
        padded_xs_snippet = torch.full(xs_shape, float('nan'), dtype=example_xs_tensor.dtype if example_xs_tensor is not None else torch.float32)

        for prefix, actual_feature_key in temporal_features_to_process.items():
            if prefix == 'date':
                batch[actual_feature_key] = np.full((len(samples_for_batch), temporal_batch_length, sequence_length_input), np.datetime64('NaT'), dtype='datetime64[ns]')
            elif prefix == 'x_d':
                batch[actual_feature_key] = torch.full((len(samples_for_batch), temporal_batch_length, *padded_xd_snippet.shape), float('nan'), dtype=padded_xd_snippet.dtype)
            elif prefix == 'y':
                batch[actual_feature_key] = torch.full((len(samples_for_batch), temporal_batch_length, *padded_y_snippet.shape), float('nan'), dtype=padded_y_snippet.dtype)
            elif prefix == 'x_s': # This is the unique shape (A,) for x_s
                batch[actual_feature_key] = torch.full((len(samples_for_batch), temporal_batch_length, *padded_xs_snippet.shape), float('nan'), dtype=padded_xs_snippet.dtype)
        return batch

    for prefix, actual_feature_key in temporal_features_to_process.items():
        list_of_temporal_sequences_for_feature = [] # This will hold (T_actual, S, F) or (T_actual, A) for each spatial slot
        
        # Determine the shape for padding elements based on feature type
        example_temporal_snippet = None
        for s_data in samples_for_batch:
            if s_data[actual_feature_key]:
                example_temporal_snippet = s_data[actual_feature_key][0]
                break
        
        # Define the padding snippet based on the example or sensible defaults for each feature
        if example_temporal_snippet is None: 
            if prefix == 'date':
                padded_snippet = np.full((sequence_length_input,), np.datetime64('NaT'), dtype='datetime64[ns]')
            elif prefix == 'x_d':
                padded_snippet = torch.full((sequence_length_input, 1), float('nan'))
            elif prefix == 'y':
                padded_snippet = torch.full((sequence_length_input,), float('nan'))
            elif prefix == 'x_s': # Special handling for x_s shape
                padded_snippet = torch.full((xs_feature_length,), float('nan'))
        else:
            if prefix == 'date':
                padded_snippet = np.full_like(example_temporal_snippet, np.datetime64('NaT'), dtype='datetime64[ns]')
            else:
                padded_snippet = torch.full_like(example_temporal_snippet, float('nan'))

        for spatial_slot_data in samples_for_batch:
            current_spatial_slot_temporal_data = spatial_slot_data[actual_feature_key]
            current_T_len = len(current_spatial_slot_temporal_data)

            if current_T_len < max_actual_T_len:
                num_to_pad = max_actual_T_len - current_T_len
                padding_snippets_for_slot = [padded_snippet] * num_to_pad
                current_spatial_slot_temporal_data.extend(padding_snippets_for_slot)
            
            if prefix == 'date':
                list_of_temporal_sequences_for_feature.append(np.stack(current_spatial_slot_temporal_data, axis=0))
            else:
                list_of_temporal_sequences_for_feature.append(torch.stack(current_spatial_slot_temporal_data, dim=0))
        
        if prefix == 'date':
            batch[actual_feature_key] = np.stack(list_of_temporal_sequences_for_feature, axis=0)
        else:
            batch[actual_feature_key] = torch.stack(list_of_temporal_sequences_for_feature, dim=0)

    return batch


class SpatialTemporalBatcher:
    """
    A custom batcher for LSTM input data that handles spatial and temporal dimensions.

    User requirements interpreted (UPDATED FOR X_S HYBRID CASE):
    1.  Input: A flat list of (temporal_dict, static_dict) tuples/lists.
        Each (temporal_dict, static_dict) represents a single pre-sliced
        temporal window (e.g., 24 hours) for a specific catchment.
        'x_d', 'y', 'date' in temporal_dict are D-length sequences.
        'x_s' in temporal_dict is a single value/vector (shape [A]) per 24h slice.
    2.  Batching:
        a.  **Spatial Batch:** Select `spatial_length` *unique catchment IDs* proportionally.
        b.  **Temporal Batch:** For each selected spatial slot, gather `temporal_length_batcher`
            consecutive pre-sliced (24h) samples. These are then aggregated (as a list
            of (24h) samples for that slot) before being passed to `collate_fn`.
        c.  **Temporal Filling:** If a catchment runs out of its own 24h samples before filling
            its `temporal_length_batcher` quota for a spatial slot, it will draw *additional
            24h samples from other un-exhausted catchments* to complete the `temporal_length_batcher`
            for that specific spatial slot. The `catchment_id` in the output batch will reflect
            the *primary* catchment for that slot (or "PADDED" if completely empty).
        d.  **Padding:** The very last batch, or any slot that cannot be fully filled
            by primary or filler data, will be padded with NaNs/NaTs.
    3.  **Output Structure:** A single dictionary, where:
        - Temporal features (x_d, y, date) have shape (Batch_Size, Temporal_Batch_Length, Sequence_Length_Input, Feature_Dim) or (B, T, D).
        - Semi-temporal feature (x_s) has shape (Batch_Size, Temporal_Batch_Length, X_S_Feature_Dim).
        - Static features (gauge_id, category, lat, lon) have shape (Batch_Size,).
    """

    def __init__(self,
                 all_pre_sliced_samples: List[Tuple[Dict[str, Any], Dict[str, Any]]],
                 spatial_length: int,
                 temporal_length_batcher: int,
                 seed: int = 42):
        """
        Initializes the SpatialTemporalBatcher.

        Args:
            all_pre_sliced_samples (List[Tuple[Dict, Dict]]): A flat list of pre-sliced samples.
                                Each element is a tuple: (temporal_features_dict, static_metadata_dict).
                                temporal_features_dict: {'dates': np.ndarray(D,), 'x_d': torch.Tensor(D,X), 'y': torch.Tensor(D,), 'x_s': torch.Tensor(A,)}
                                static_metadata_dict: {'catchment_id': str, 'category': str, 'lat': float, 'lon': float}
                                Note: 'x_s' is now back in temporal_dict but with shape (A,).
            spatial_length (int): The desired number of spatial slots in each batch.
            temporal_length_batcher (int): The desired number of 24h samples to concatenate
                                           temporally for each spatial slot in a batch.
            seed (int): Random seed for reproducibility.
        """
        if not all_pre_sliced_samples:
            raise ValueError("SpatialTemporalBatcher: 'all_pre_sliced_samples' cannot be empty.")
        if spatial_length <= 0:
            raise ValueError("SpatialTemporalBatcher: 'spatial_length' must be positive.")
        if temporal_length_batcher <= 0:
            raise ValueError("SpatialTemporalBatcher: 'temporal_length_batcher' must be positive.")

        self.spatial_length = spatial_length
        self.temporal_length_batcher = temporal_length_batcher
        self.rng = random.Random(seed)

        self.example_pre_sliced_sample = all_pre_sliced_samples[0]
        example_temporal_dict = self.example_pre_sliced_sample[0]
        example_static_dict = self.example_pre_sliced_sample[1]

        # Dynamically find the actual keys from the example sample
        # Temporal keys (including x_s)
        self.actual_date_key = next((k for k in example_temporal_dict if k.startswith('date') and isinstance(example_temporal_dict[k], np.ndarray)), None)
        self.actual_xd_key = next((k for k in example_temporal_dict if k.startswith('x_d') and isinstance(example_temporal_dict[k], torch.Tensor)), None)
        self.actual_y_key = next((k for k in example_temporal_dict if k.startswith('y') and isinstance(example_temporal_dict[k], torch.Tensor)), None)
        self.actual_xs_key = next((k for k in example_temporal_dict if k.startswith('x_s') and isinstance(example_temporal_dict[k], torch.Tensor)), None) # x_s is here now!

        # Static keys
        self.actual_catchment_id_key = next((k for k in example_static_dict if k.startswith('gauge') and isinstance(example_static_dict[k], str)), None)
        # MODIFIED: Allow 'Cluster' to be float, int, or str
        self.actual_category_key = next((k for k in example_static_dict if k.startswith('Cluster') and isinstance(example_static_dict[k], (float, int, str))), None)
        self.actual_lat_key = next((k for k in example_static_dict if k.startswith('lat') and isinstance(example_static_dict[k], (float, int))), None)
        self.actual_lon_key = next((k for k in example_static_dict if k.startswith('lon') and isinstance(example_static_dict[k], (float, int))), None)

        # Validate that all expected keys were found
        if self.actual_date_key is None: raise ValueError(f"SpatialTemporalBatcher: Could not find key starting with 'date' in example temporal_dict: {example_temporal_dict.keys()}")
        if self.actual_xd_key is None: raise ValueError(f"SpatialTemporalBatcher: Could not find key starting with 'x_d' in example temporal_dict: {example_temporal_dict.keys()}")
        if self.actual_y_key is None: raise ValueError(f"SpatialTemporalBatcher: Could not find key starting with 'y' in example temporal_dict: {example_temporal_dict.keys()}")
        if self.actual_xs_key is None: raise ValueError(f"SpatialTemporalBatcher: Could not find key starting with 'x_s' in example temporal_dict: {example_temporal_dict.keys()}") # x_s is in temporal_dict now
        
        if self.actual_catchment_id_key is None: raise ValueError(f"SpatialTemporalBatcher: Could not find key starting with 'gauge' in example static_dict: {example_static_dict.keys()}")
        if self.actual_category_key is None: raise ValueError(f"SpatialTemporalBatcher: Could not find key starting with 'Cluster' in example static_dict: {example_static_dict.keys()}")
        if self.actual_lat_key is None: raise ValueError(f"SpatialTemporalBatcher: Could not find key starting with 'lat' in example static_dict: {example_static_dict.keys()}")
        if self.actual_lon_key is None: raise ValueError(f"SpatialTemporalBatcher: Could not find key starting with 'lon' in example static_dict: {example_static_dict.keys()}")


        # Determine the expected sequence length (D) from standard temporal features
        self.sequence_length_input = example_temporal_dict[self.actual_date_key].shape[0] 
        # Determine the expected feature length (A) for x_s
        self.xs_feature_length = example_temporal_dict[self.actual_xs_key].shape[0]


        # --- Input Data Consistency Validation ---
        self.catchment_samples_map: Dict[str, deque] = defaultdict(deque)
        self.all_unique_catchment_ids: List[str] = []

        for i, (temp_dict, static_dict) in enumerate(all_pre_sliced_samples):
            current_catchment_id_key = next((k for k in static_dict if k.startswith('gauge') and isinstance(static_dict[k], str)), None)
            if current_catchment_id_key is None:
                raise ValueError(f"SpatialTemporalBatcher: Sample {i}: Could not find a key starting with 'gauge' in static_dict: {static_dict}")
            catchment_id_val = static_dict[current_catchment_id_key]

            # Validate TEMPORAL features (D-length)
            if temp_dict[self.actual_date_key].shape[0] != self.sequence_length_input:
                raise ValueError(f"SpatialTemporalBatcher: Sample {i} ({catchment_id_val}): 'date' sequence length mismatch. Expected {self.sequence_length_input}, got {temp_dict[self.actual_date_key].shape[0]}.")
            if temp_dict[self.actual_xd_key].shape[0] != self.sequence_length_input:
                raise ValueError(f"SpatialTemporalBatcher: Sample {i} ({catchment_id_val}): '{self.actual_xd_key}' sequence length mismatch. Expected {self.sequence_length_input}, got {temp_dict[self.actual_xd_key].shape[0]}.")
            if temp_dict[self.actual_y_key].shape[0] != self.sequence_length_input:
                raise ValueError(f"SpatialTemporalBatcher: Sample {i} ({catchment_id_val}): '{self.actual_y_key}' sequence length mismatch. Expected {self.sequence_length_input}, got {temp_dict[self.actual_y_key].shape[0]}.")
            
            # Validate SEMI-TEMPORAL feature (x_s, A-length, 1D)
            if temp_dict[self.actual_xs_key].ndim != 1 or temp_dict[self.actual_xs_key].shape[0] != self.xs_feature_length:
                raise ValueError(f"SpatialTemporalBatcher: Sample {i} ({catchment_id_val}): '{self.actual_xs_key}' feature length mismatch. Expected 1D tensor of shape ({self.xs_feature_length},), got {temp_dict[self.actual_xs_key].shape}.")

            self.catchment_samples_map[catchment_id_val].append((temp_dict, static_dict))
            if catchment_id_val not in self.all_unique_catchment_ids:
                self.all_unique_catchment_ids.append(catchment_id_val)
        
        self.category_counts = defaultdict(int)
        for cid in self.all_unique_catchment_ids:
            if self.catchment_samples_map[cid]:
                sample_static_dict_for_cat = self.catchment_samples_map[cid][0][1]
                # MODIFIED: Get the category value directly using the identified key
                current_category_val = sample_static_dict_for_cat[self.actual_category_key] 

                # NEW CRITICAL FIX: Convert np.nan from original data to a specific string to avoid hashing/comparison issues
                if isinstance(current_category_val, float) and np.isnan(current_category_val):
                    current_category_val = "NaN_Original_Category" # Standardize this specific string
                
                self.category_counts[current_category_val] += 1 
        
        total_unique_catchments = len(self.all_unique_catchment_ids)
        if total_unique_catchments == 0:
            raise ValueError("SpatialTemporalBatcher: No unique catchments found in the provided samples.")

        self.category_proportions = {cat: count / total_unique_catchments
                                     for cat, count in self.category_counts.items()}

        self._reset_state()

    def _reset_state(self):
        """Resets the internal state for a new iteration (e.g., a new epoch)."""
        self.active_catchment_sample_queues = {
            cid: deque(self.rng.sample(list(self.catchment_samples_map[cid]), len(self.catchment_samples_map[cid])))
            for cid in self.all_unique_catchment_ids
        }
        self.exhausted_catchment_ids_for_epoch = set() 
        self.current_spatial_batch_active_cids = deque() 
        
        self.global_unselected_cids_pool = deque(self.rng.sample(self.all_unique_catchment_ids, len(self.all_unique_catchment_ids)))
        
        self.available_cids_by_category: Dict[Any, List[str]] = defaultdict(list)
        for cid in self.global_unselected_cids_pool:
            sample_static_dict_for_cat = self.catchment_samples_map[cid][0][1]
            category = sample_static_dict_for_cat[self.actual_category_key]
            # NEW: Handle np.nan categories during selection too, ensuring consistency
            if isinstance(category, float) and np.isnan(category):
                category = "NaN_Original_Category"
            self.available_cids_by_category[category].append(cid)
        # MODIFIED: Use key=str for sorting to handle mixed types (float, int, str)
        for cat_list in self.available_cids_by_category.values():
            self.rng.shuffle(cat_list)


    def _select_next_spatial_batch_proportional(self):
        """
        Selects `spatial_length` unique catchment IDs proportionally from the available global pool.
        """
        self.current_spatial_batch_active_cids.clear() 
        
        selected_cids = []
        current_total_available = sum(len(cids) for cids in self.available_cids_by_category.values())

        if current_total_available == 0:
            return 

        temp_category_counts = {cat: len(cids) for cat, cids in self.available_cids_by_category.items()}
        temp_category_proportions = {cat: count / current_total_available
                                     for cat, count in temp_category_counts.items()}
        
        # MODIFIED: Use key=str for sorting to handle mixed types (float, int, str)
        all_categories_sorted = sorted(list(self.category_proportions.keys()), key=str) 

        for cat in all_categories_sorted:
            if cat not in self.available_cids_by_category:
                continue

            ideal_count_for_cat = int(round(temp_category_proportions.get(cat, 0) * self.spatial_length))
            
            num_to_select = min(ideal_count_for_cat, len(self.available_cids_by_category[cat]))
            
            selected_from_cat = [self.available_cids_by_category[cat].pop(0) for _ in range(num_to_select)]
            selected_cids.extend(selected_from_cat)

        if len(selected_cids) < self.spatial_length:
            remaining_needed = self.spatial_length - len(selected_cids)
            all_remaining_cids = []
            for cat in all_categories_sorted: 
                all_remaining_cids.extend(self.available_cids_by_category[cat])
                self.available_cids_by_category[cat].clear() 

            self.rng.shuffle(all_remaining_cids) 
            selected_cids.extend(all_remaining_cids[:remaining_needed])
        
        elif len(selected_cids) > self.spatial_length:
            excess_count = len(selected_cids) - self.spatial_length
            cids_to_return = self.rng.sample(selected_cids, excess_count)
            selected_cids = [cid for cid in selected_cids if cid not in cids_to_return]
            
            for cid_return in cids_to_return:
                sample_static_dict = self.catchment_samples_map[cid_return][0][1]
                category = sample_static_dict[self.actual_category_key]
                # NEW: Handle np.nan categories when returning excess
                if isinstance(category, float) and np.isnan(category):
                    category = "NaN_Original_Category"
                self.available_cids_by_category[category].append(cid_return) 
                self.rng.shuffle(self.available_cids_by_category[category])

        self.current_spatial_batch_active_cids.extend(selected_cids)


    def _create_padded_spatial_slot_data(self, num_temporal_samples_to_pad: int) -> Dict[str, Any]:
        """
        Creates a dictionary representing a padded spatial slot.
        Temporal features are lists of (S,F) or (A,) padded tensors/arrays.
        Static features are padded to match their expected types/shapes.
        """
        example_temporal_dict = self.example_pre_sliced_sample[0]
        # example_static_dict = self.example_pre_sliced_sample[1] # Not directly used for padding snippets now

        # Create single (S,F) padded blocks for standard temporal features
        padded_date_block = np.full((self.sequence_length_input,), np.datetime64('NaT'), dtype='datetime64[ns]')
        
        xd_feat_dim = example_temporal_dict[self.actual_xd_key].shape[1:]
        y_feat_dim = example_temporal_dict[self.actual_y_key].shape[1:] if example_temporal_dict[self.actual_y_key].ndim > 1 else () 

        padded_xd_block = torch.full((self.sequence_length_input,) + xd_feat_dim, float('nan'), dtype=example_temporal_dict[self.actual_xd_key].dtype)
        padded_y_block = torch.full((self.sequence_length_input,) + y_feat_dim, float('nan'), dtype=example_temporal_dict[self.actual_y_key].dtype)

        # NEW: Padded x_s block (shape (A,))
        padded_xs_block = torch.full((self.xs_feature_length,), float('nan'), dtype=example_temporal_dict[self.actual_xs_key].dtype)


        # Create lists of these padded blocks for the desired temporal length
        padded_dates = [padded_date_block] * num_temporal_samples_to_pad
        padded_xds = [padded_xd_block] * num_temporal_samples_to_pad
        padded_ys = [padded_y_block] * num_temporal_samples_to_pad
        padded_xss = [padded_xs_block] * num_temporal_samples_to_pad # x_s is still collected per temporal slice

        # Use actual keys for the padded slot data (stored from __init__)
        padded_slot_data: Dict[str, Any] = {
            self.actual_catchment_id_key: "PADDED",
            self.actual_category_key: "N/A", # Explicitly a string for padded slots
            self.actual_lat_key: float('nan'),
            self.actual_lon_key: float('nan'),
            # Temporal features (even if just for padding)
            self.actual_date_key: padded_dates,
            self.actual_xd_key: padded_xds,
            self.actual_y_key: padded_ys,
            self.actual_xs_key: padded_xss # x_s is here now
        }
        return padded_slot_data


    def _generate_single_batch(self) -> Optional[Dict[str, Any]]: 
        """
        Generates a single batch by preparing data for `spatial_length` slots.
        Returns None if no more batches can be formed.
        """
        batch_items_for_collation: List[Dict[str, Any]] = [] 
        batch_categories: List[Union[str, float, int]] = [] 

        # Store keys as local variables for slight lookup efficiency within the loop
        actual_date_key = self.actual_date_key
        actual_xd_key = self.actual_xd_key
        actual_y_key = self.actual_y_key
        actual_xs_key = self.actual_xs_key # x_s is here now, in temporal dict
        actual_catchment_id_key = self.actual_catchment_id_key
        actual_category_key = self.actual_category_key
        actual_lat_key = self.actual_lat_key
        actual_lon_key = self.actual_lon_key

        while len(batch_items_for_collation) < self.spatial_length:
            if not self.current_spatial_batch_active_cids:
                self._select_next_spatial_batch_proportional()
                if not self.current_spatial_batch_active_cids:
                    if batch_items_for_collation:
                        # Pass xs_feature_length to collate_fn
                        return collate_fn(batch_items_for_collation, self.temporal_length_batcher, self.sequence_length_input, self.xs_feature_length)
                    else:
                        return None

            primary_cid = self.current_spatial_batch_active_cids.popleft()

            current_spatial_slot_dates: List[np.ndarray] = []
            current_spatial_slot_xd: List[torch.Tensor] = []
            current_spatial_slot_y: List[torch.Tensor] = []
            current_spatial_slot_xs: List[torch.Tensor] = [] # x_s is back in temporal collection

            spatial_slot_static_data: Dict[str, Union[str, float, int]] = {} # x_s is NOT here

            num_samples_collected_for_slot = 0
            
            while num_samples_collected_for_slot < self.temporal_length_batcher:
                sample_to_add: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None

                if self.active_catchment_sample_queues[primary_cid]:
                    sample_to_add = self.active_catchment_sample_queues[primary_cid].popleft()
                else:
                    self.exhausted_catchment_ids_for_epoch.add(primary_cid)
                    
                    available_filler_cids = [
                        cid for cid in self.all_unique_catchment_ids
                        if cid not in self.exhausted_catchment_ids_for_epoch
                        and self.active_catchment_sample_queues[cid]
                    ]
                    self.rng.shuffle(available_filler_cids) 

                    if available_filler_cids:
                        filler_cid = available_filler_cids[0]
                        sample_to_add = self.active_catchment_sample_queues[filler_cid].popleft()
                        if not self.active_catchment_sample_queues[filler_cid]:
                            self.exhausted_catchment_ids_for_epoch.add(filler_cid)
                
                if sample_to_add is None:
                    break 
                
                temp_dict, static_dict = sample_to_add

                # Collect static data only from the *first* actual sample used for this slot
                if num_samples_collected_for_slot == 0:
                    spatial_slot_static_data = {
                        actual_catchment_id_key: static_dict[actual_catchment_id_key],
                        actual_category_key: static_dict[actual_category_key],
                        actual_lat_key: static_dict[actual_lat_key],
                        actual_lon_key: static_dict[actual_lon_key],
                    }
                    # NEW: Convert category to a specific string for 'nan' values when storing
                    if isinstance(spatial_slot_static_data[actual_category_key], float) and np.isnan(spatial_slot_static_data[actual_category_key]):
                        spatial_slot_static_data[actual_category_key] = "NaN_Original_Category"


                current_spatial_slot_dates.append(temp_dict[actual_date_key])
                current_spatial_slot_xd.append(temp_dict[actual_xd_key])
                current_spatial_slot_y.append(temp_dict[actual_y_key])
                current_spatial_slot_xs.append(temp_dict[actual_xs_key]) # Collect x_s here

                num_samples_collected_for_slot += 1
                
            prepared_spatial_slot_data: Dict[str, Any] = {}
            if num_samples_collected_for_slot == 0:
                prepared_spatial_slot_data = self._create_padded_spatial_slot_data(self.temporal_length_batcher)
                batch_categories.append("N/A") 
            else:
                prepared_spatial_slot_data[actual_date_key] = current_spatial_slot_dates
                prepared_spatial_slot_data[actual_xd_key] = current_spatial_slot_xd
                prepared_spatial_slot_data[actual_y_key] = current_spatial_slot_y
                prepared_spatial_slot_data[actual_xs_key] = current_spatial_slot_xs # x_s is here
                
                needed_padding_samples = self.temporal_length_batcher - num_samples_collected_for_slot
                if needed_padding_samples > 0:
                    padded_snippet_data = self._create_padded_spatial_slot_data(needed_padding_samples)
                    
                    prepared_spatial_slot_data[actual_date_key].extend(padded_snippet_data[actual_date_key])
                    prepared_spatial_slot_data[actual_xd_key].extend(padded_snippet_data[actual_xd_key])
                    prepared_spatial_slot_data[actual_y_key].extend(padded_snippet_data[actual_y_key])
                    prepared_spatial_slot_data[actual_xs_key].extend(padded_snippet_data[actual_xs_key]) # x_s padding
                    
                prepared_spatial_slot_data.update(spatial_slot_static_data)
                
                batch_categories.append(spatial_slot_static_data[actual_category_key])

            batch_items_for_collation.append(prepared_spatial_slot_data)

            if primary_cid not in self.exhausted_catchment_ids_for_epoch:
                self.current_spatial_batch_active_cids.append(primary_cid)


        # Pass xs_feature_length to collate_fn
        return collate_fn(batch_items_for_collation, self.temporal_length_batcher, self.sequence_length_input, self.xs_feature_length)

    def _print_batch_distribution(self, batch_categories: List[Union[str, float, int]]):
        """
        Calculates and prints the category distribution for the current batch.
        """
        batch_category_counts = defaultdict(int)
        for cat in batch_categories:
            batch_category_counts[cat] += 1
        
        print("\nBatch Category Distribution:")
        if len(batch_categories) > 0:
            # MODIFIED: Use key=str for sorting to handle mixed types (float, int, str)
            all_known_categories = sorted(list(self.category_proportions.keys()) + ['N/A'], key=str) 
            
            for cat in all_known_categories:
                batch_count = batch_category_counts.get(cat, 0)
                batch_prop = batch_count / len(batch_categories)
                print(f"  {cat}: Count = {batch_count}, Proportion = {batch_prop:.2f}")
        else:
            print("  (Empty batch)")

    def get_all_batches(self) -> List[Dict[str, Any]]:
        """
        Generates all possible batches from the dataset and returns them as a list.
        """
        self._reset_state()
        all_collated_batches = []
        while True:
            batch = self._generate_single_batch()
            if batch is None:
                break
            all_collated_batches.append(batch)
        return all_collated_batches