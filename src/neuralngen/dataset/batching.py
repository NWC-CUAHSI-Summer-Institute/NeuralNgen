# src/neuralngen/dataset/batching.py

from typing import List, Dict, Union
import numpy as np
import torch

def collate_fn(
    samples: List[Dict[str, Union[torch.Tensor, np.ndarray, Dict[str, torch.Tensor], str, float]]]
) -> Dict[str, Union[torch.Tensor, np.ndarray, Dict[str, torch.Tensor], List[str], List[float]]]:
    """
    Collates a list of individual catchment samples into a single batch.
    This function is designed to be compatible with PyTorch DataLoader.

    Args:
        samples (List[Dict]): A list of dictionaries, where each dictionary represents
                               data for one catchment over a specific temporal window.

    Returns:
        Dict: A single dictionary representing the collated batch, with features stacked.
    """
    batch = {}
    if not samples:
        return batch

    # Collect all unique features present in the samples to handle potential variations
    all_features = set()
    for sample in samples:
        all_features.update(sample.keys())

    # Example sample to infer shapes and types for padding/stacking
    example_sample_template = samples[0]

    for feature in all_features:
        # Special handling for 'dates' (numpy array) - now checks `startswith`
        if feature.startswith('date'):
            # Dates are stored as a numpy array of datetime64. Stack them.
            dates_list = [s.get(feature, np.full((samples[0][feature].shape[0],), np.datetime64('NaT'), dtype='datetime64[ns]')) for s in samples]
            # Ensure consistent length for stacking if padding was applied to individual samples
            max_len = max(len(d) for d in dates_list) # This will typically be self.temporal_length
            stacked_dates = np.stack([
                np.pad(d, (0, max_len - len(d)), mode='constant', constant_values=np.datetime64('NaT'))
                for d in dates_list
            ], axis=0)
            batch[feature] = stacked_dates
        
        # Special handling for 'x_d' (dictionary of tensors) - already uses `startswith`
        elif feature.startswith('x_d'): # Handles 'x_d' itself and any other features starting with 'x_d'
            # Assuming 'x_d' is a dictionary, and its sub-features are tensors
            if isinstance(example_sample_template.get(feature), dict):
                sub_features = list(example_sample_template.get(feature, {}).keys())
                batch[feature] = {}
                for k in sub_features:
                    max_len = max(s.get(feature, {}).get(k, torch.empty(0)).shape[0] for s in samples)
                    tensors_to_stack = []
                    for s in samples:
                        if k in s.get(feature, {}):
                            current_tensor = s[feature][k]
                            if current_tensor.shape[0] < max_len:
                                pad_shape = (max_len - current_tensor.shape[0],) + current_tensor.shape[1:]
                                current_tensor = torch.cat((current_tensor, torch.full(pad_shape, float('nan'), dtype=current_tensor.dtype)), dim=0)
                            tensors_to_stack.append(current_tensor)
                        else:
                            # If a sub-feature is missing, create a NaN-filled tensor of appropriate size
                            shape = (max_len,) + example_sample_template[feature][k].shape[1:]
                            tensors_to_stack.append(torch.full(shape, float('nan'), dtype=example_sample_template[feature][k].dtype))
                    batch[feature][k] = torch.stack(tensors_to_stack, dim=0)
            else: # If 'x_d' or similar prefix is a direct tensor, treat it like other tensors below
                # This case should ideally not happen if 'x_d' is always a dict of tensors
                pass # Fall through to the general tensor handling if necessary, but unlikely for 'x_d'
        
        # Metadata handling (string or float lists)
        elif feature in ['catchment_id', 'category']:
            batch[feature] = [s.get(feature, 'N/A') for s in samples]
        elif feature in ['lat', 'lon']:
            batch[feature] = [s.get(feature, float('nan')) for s in samples]
        
        # Generic handling for all other torch.Tensor features (y, x_s, or any new dynamic/static tensors)
        else:
            # Check if this feature is a torch.Tensor in the first sample
            if feature in samples[0] and isinstance(samples[0][feature], torch.Tensor):
                # Determine if it's a temporal tensor based on its first dimension
                # Compare to the temporal length of 'dates' from the first sample,
                # as 'dates' is already guaranteed to be padded to `temporal_length` (or shorter for last batch).
                # To be robust, find the actual 'dates' feature key in samples[0]
                dates_key = next((k for k in samples[0] if k.startswith('date')), None)
                
                is_temporal_tensor = False
                if dates_key and samples[0][feature].ndim > 0:
                    # Compare to the actual length of the dates array for that sample
                    is_temporal_tensor = (samples[0][feature].shape[0] == samples[0][dates_key].shape[0])

                if is_temporal_tensor:
                    # It's a temporal tensor, so calculate max_len and pad if necessary
                    max_len = max(s.get(feature, torch.empty(0)).shape[0] for s in samples)
                    tensors_to_stack = []
                    for s in samples:
                        current_tensor = s.get(feature, torch.empty(0))
                        if current_tensor.shape[0] < max_len:
                            pad_shape = (max_len - current_tensor.shape[0],) + current_tensor.shape[1:]
                            current_tensor = torch.cat((current_tensor, torch.full(pad_shape, float('nan'), dtype=current_tensor.dtype)), dim=0)
                        tensors_to_stack.append(current_tensor)
                    batch[feature] = torch.stack(tensors_to_stack, dim=0)
                else: # It's a static tensor
                    # Assumes static tensors have consistent shapes across samples, or are handled by _create_padded_sample
                    batch[feature] = torch.stack([s.get(feature, torch.full_like(example_sample_template[feature], float('nan'))) for s in samples], dim=0)
            else:
                # For any other non-tensor, non-date, non-x_d, non-metadata features, just copy them as-is
                # These are typically scalars or other simple types not meant for stacking as tensors
                batch[feature] = [s.get(feature, None) for s in samples] # Use None if missing or default appropriate value


    return batch


class SpatialTemporalBatcher:
    """
    A custom batcher for LSTM input data that handles spatial and temporal dimensions
    according to specific user requirements for CAMELS/AORC data.

    User requirements interpreted:
    1.  Spatial batches (of `spatial_length` catchments) are formed by selecting catchments
        proportionally to their overall category distribution.
    2.  Temporal sub-batches (of `temporal_length` timesteps) are extracted for these catchments.
    3.  ASSUMPTION: All catchments have the same total number of days.
        If a catchment's data for a temporal window is shorter than `temporal_length`,
        the remaining duration for *that specific series within the batch* is filled by
        concatenating data from the beginning of an *un-exhausted, globally available catchment*.
        This means a single 'sample' in the batch might contain data from two different catchments
        concatenated temporally.
    4.  The very last batch might have fewer than `spatial_length` catchments or fewer than
        `temporal_length` timesteps if the global data pool is exhausted. This is the ONLY
        scenario where `NaT`/`NaN` padding should appear.
    """

    def __init__(self,
                 all_catchment_data: List[Dict],
                 spatial_length: int,
                 temporal_length: int,
                 seed: int = 42):
        """
        Initializes the SpatialTemporalBatcher.

        Args:
            all_catchment_data (List[Dict]): A list where each dictionary represents
                                             all data for a single catchment.
                                             Each dict must contain:
                                             - 'catchment_id' (str)
                                             - 'category' (str)
                                             - 'dates' (np.ndarray of datetime64)
                                             - 'x_d' (Dict[str, torch.Tensor]) - dynamic features
                                             - 'x_s' (torch.Tensor) - static features (optional)
                                             - 'y' (torch.Tensor) - target variable (optional)
                                             - 'lat' (float), 'lon' (float) (optional, if to be used as x_s)
            spatial_length (int): The desired number of catchments in each spatial batch.
            temporal_length (int): The desired number of timesteps in each temporal sub-batch.
            seed (int): Random seed for reproducibility.
        """
        if not all_catchment_data:
            raise ValueError("all_catchment_data cannot be empty.")
        if spatial_length <= 0 or temporal_length <= 0:
            raise ValueError("spatial_length and temporal_length must be positive.")

        self.all_catchment_data_map = {d['catchment_id']: d for d in all_catchment_data}
        self.all_catchment_ids = list(self.all_catchment_data_map.keys())
        self.spatial_length = spatial_length
        self.temporal_length = temporal_length
        self.rng = random.Random(seed)

        # Store an example sample to create padded samples with correct shapes/types
        self.example_sample_template = list(self.all_catchment_data_map.values())[0]
        # Store the actual key for 'dates' based on what's in the template
        self.dates_key_in_template = next((k for k in self.example_sample_template if k.startswith('date')), None)
        if not self.dates_key_in_template:
            raise ValueError("Input data must contain a feature starting with 'date' for temporal alignment.")


        # Calculate overall category distribution
        self.category_counts = defaultdict(int)
        for data in all_catchment_data:
            self.category_counts[data['category']] += 1
        total_catchments = len(all_catchment_data)
        self.category_proportions = {cat: count / total_catchments
                                     for cat, count in self.category_counts.items()}

        # Initialize state for iteration (this state will be used by _generate_single_batch)
        self._reset_state()

    def _reset_state(self):
        """Resets the internal state for a new iteration (e.g., a new epoch)."""
        self.current_temporal_offsets = {cid: 0 for cid in self.all_catchment_ids}
        self.exhausted_catchments = set()
        # This deque holds the CIDs that are currently part of the active spatial batch
        # for which we are drawing temporal slices. It gets refilled by _select_next_spatial_batch_proportional.
        self.current_spatial_batch_active_cids = deque()
        # This deque holds all available CIDs globally, from which new spatial batches are drawn
        # and fillers are taken. CIDs are removed from here once used.
        self.global_unselected_cids = deque(self.rng.sample(self.all_catchment_ids, len(self.all_catchment_ids)))

    def _select_next_spatial_batch_proportional(self):
        """
        Selects `spatial_length` catchments proportionally from the available global pool.
        These become the *initial* set of active catchments for a new spatial batch chunk.
        """
        self.current_spatial_batch_active_cids.clear() # Clear for a fresh start

        # Identify all truly available catchments for selection from the global pool (not exhausted)
        available_for_selection = [cid for cid in self.global_unselected_cids if cid not in self.exhausted_catchments]

        if not available_for_selection:
            return # No more catchments to select from the global pool

        # Calculate category distribution of *only* the currently available catchments
        temp_category_counts = defaultdict(int)
        for cid in available_for_selection:
            temp_category_counts[self.all_catchment_data_map[cid]['category']] += 1
        
        total_available = len(available_for_selection)
        if total_available == 0: # Should be caught by the outer check
            return 

        temp_category_proportions = {cat: count / total_available
                                     for cat, count in temp_category_counts.items()}
        
        selected_cids = []
        # Distribute the `spatial_length` slots among categories proportionally
        # Iterate over all possible categories to ensure all are considered for proportional selection
        all_categories = sorted(list(self.category_proportions.keys())) # Use original categories for completeness
        
        for cat in all_categories:
            if cat not in temp_category_proportions: # If no available candidates in this category
                continue

            # Calculate ideal count based on the proportion of this category *within the currently available pool*
            # and the total desired spatial_length.
            ideal_count_for_cat = int(round(temp_category_proportions.get(cat, 0) * self.spatial_length))
            
            # Get candidates for this category from the `available_for_selection`
            cat_candidates = [cid for cid in available_for_selection 
                              if self.all_catchment_data_map[cid]['category'] == cat and cid not in selected_cids]
            self.rng.shuffle(cat_candidates) # Shuffle to get random selection within category
            
            selected_from_cat = cat_candidates[:ideal_count_for_cat]
            selected_cids.extend(selected_from_cat)

            # Remove selected from available_for_selection temporarily to avoid re-selection in same batch
            available_for_selection = [cid for cid in available_for_selection if cid not in selected_from_cat]

        # Adjust selected_cids to exactly `spatial_length` if needed (due to rounding or small pool)
        if len(selected_cids) < self.spatial_length:
            # If not enough were selected by proportionality, fill with remaining random ones
            remaining_needed = self.spatial_length - len(selected_cids)
            self.rng.shuffle(available_for_selection) # Shuffle remaining candidates
            selected_cids.extend(available_for_selection[:remaining_needed])
        elif len(selected_cids) > self.spatial_length:
            # If too many due to rounding, randomly trim down to spatial_length
            selected_cids = self.rng.sample(selected_cids, self.spatial_length)

        # Populate the active queue with the selected CIDs
        self.current_spatial_batch_active_cids.extend(selected_cids)

        # Corrected line: Remove selected CIDs from the global_unselected_cids pool
        self.global_unselected_cids = deque([cid_global for cid_global in self.global_unselected_cids if cid_global not in selected_cids])


    def _create_padded_sample(self, length: int) -> Dict:
        """
        Creates a padded sample (dictionary) with NaNs/zeros for numerical features
        and 'NaT' for dates, based on an example sample's structure.
        """
        padded_sample = {
            'catchment_id': "PADDED", # Indicate it's a padded sample
            'category': "N/A",
            'lat': float('nan'),
            'lon': float('nan'),
        }

        # Dates: create a numpy array of 'Not a Time'
        padded_sample[self.dates_key_in_template] = np.full((length,), np.datetime64('NaT'), dtype='datetime64[ns]')

        # Iterate through all other features in the example template to create padded versions
        for key, value_template in self.example_sample_template.items():
            # Skip features already explicitly handled above or known to be metadata/strings
            if key in ['catchment_id', 'category', 'lat', 'lon', self.dates_key_in_template]:
                continue
            
            if key.startswith('x_d') and isinstance(value_template, dict):
                padded_sample[key] = {}
                for sub_k, sub_v_template in value_template.items():
                    if isinstance(sub_v_template, torch.Tensor):
                        shape = (length,) + sub_v_template.shape[1:]
                        padded_sample[key][sub_k] = torch.full(shape, float('nan'), dtype=sub_v_template.dtype)
                continue

            if isinstance(value_template, torch.Tensor):
                # If it's a temporal tensor in the template (its first dim matches the template's dates length)
                # We use the original full length of dates from the template for this check.
                if value_template.ndim >= 1 and value_template.shape[0] == self.example_sample_template[self.dates_key_in_template].shape[0]:
                    shape = (length,) + value_template.shape[1:]
                    padded_sample[key] = torch.full(shape, float('nan'), dtype=value_template.dtype)
                else: # Assume it's a static tensor
                    padded_sample[key] = torch.full_like(value_template, float('nan'))
            # For non-tensor features not explicitly handled, they are typically copied as-is,
            # so no special padding logic is needed here for them.

        return padded_sample

    def _generate_single_batch(self) -> Union[Dict, None]:
        """
        Generates a single collated batch. This is an internal helper method.
        Returns None if no more batches can be formed.
        """
        batch_samples = []
        batch_categories = [] # To store categories for the current batch

        # Keep looping until we form a full spatial_length batch or run out of global data
        while len(batch_samples) < self.spatial_length:
            # If the active spatial batch queue is empty (meaning the previous set of catchments
            # has been fully processed for initial assignment), refill it with new catchments.
            if not self.current_spatial_batch_active_cids:
                self._select_next_spatial_batch_proportional()
                if not self.current_spatial_batch_active_cids:
                    # No more catchments globally available to select from for *initial* spatial slots.
                    # If we've collected any partial batch, return it. Otherwise, signal end of iteration.
                    if not batch_samples:
                        return None # No more batches can be formed
                    else:
                        # Before returning the last partial batch, calculate and print its distribution
                        self._print_batch_distribution(batch_categories)
                        return collate_fn(batch_samples) # Return the last partial collated batch

            # Get the next catchment ID from the active spatial batch queue.
            # This is the primary catchment for the current sample slot in the batch.
            primary_cid = self.current_spatial_batch_active_cids.popleft()

            # If this primary catchment is already exhausted from previous temporal slices,
            # we need to find a new, un-exhausted catchment to fill this spatial slot entirely.
            if primary_cid in self.exhausted_catchments:
                replacement_cid = None
                # Search for a fresh, un-exhausted catchment from the global pool.
                # Prioritize those not already added to the current `batch_samples` to avoid duplicates
                # within the current output batch.
                available_global_cids_for_replacement = [
                    cid for cid in self.global_unselected_cids
                    if cid not in self.exhausted_catchments and cid not in [s['catchment_id'] for s in batch_samples]
                ]
                self.rng.shuffle(available_global_cids_for_replacement)

                if available_global_cids_for_replacement:
                    replacement_cid = available_global_cids_for_replacement[0]
                    # Remove the replacement from the global_unselected_cids since it's now used.
                    self.global_unselected_cids = deque([c for c in self.global_unselected_cids if c != replacement_cid])
                
                if replacement_cid is None:
                    # No more real catchments to substitute for an exhausted one, so add a padded sample.
                    padded_sample = self._create_padded_sample(self.temporal_length)
                    batch_samples.append(padded_sample)
                    batch_categories.append(padded_sample['category']) # Add category for padded sample
                    continue # Move to the next spatial slot
                else:
                    primary_cid = replacement_cid # Use the replacement for this slot

            # Now, process the (potentially new/replacement) `primary_cid` for its temporal slice.
            catchment_data_full = self.all_catchment_data_map[primary_cid]
            current_temporal_idx = self.current_temporal_offsets[primary_cid]
            total_temporal_len_cid = catchment_data_full[self.dates_key_in_template].shape[0]

            # Initialize the sample dictionary for this batch slot.
            sliced_sample = {
                'catchment_id': primary_cid,
                'category': catchment_data_full['category'],
                'lat': catchment_data_full.get('lat', float('nan')),
                'lon': catchment_data_full.get('lon', float('nan')),
            }

            # Prepare to collect all features that need temporal slicing/padding or direct copying
            # This will store initial slices of temporal features that will be further filled/padded
            temporal_feature_slices = {}
            
            # --- Iterate through all items in the full catchment data to handle features flexibly ---
            for key, value in catchment_data_full.items():
                # Skip metadata already handled
                if key in ['catchment_id', 'category', 'lat', 'lon']:
                    continue
                
                # Handle 'dates' explicitly as a numpy array
                if key.startswith('date'):
                    temporal_feature_slices[key] = value[current_temporal_idx : current_temporal_idx + self.temporal_length]
                    continue

                # Handle 'x_d' (dictionary of tensors)
                if key.startswith('x_d') and isinstance(value, dict):
                    temporal_feature_slices[key] = {sub_k: sub_v[current_temporal_idx : current_temporal_idx + self.temporal_length]
                                                  for sub_k, sub_v in value.items()}
                    continue
                
                # Handle any other torch.Tensor feature (e.g., 'y', 'additional_temporal_feature', 'x_s')
                if isinstance(value, torch.Tensor):
                    # If its first dimension matches the total temporal length, it's a temporal feature
                    if value.ndim >= 1 and value.shape[0] == total_temporal_len_cid:
                        temporal_feature_slices[key] = value[current_temporal_idx : current_temporal_idx + self.temporal_length]
                    else: # Otherwise, assume it's a static tensor (e.g., x_s)
                        sliced_sample[key] = value # Directly copy static tensors
                else:
                    # For any other non-tensor, non-date, non-x_d, non-metadata, just copy it as-is
                    sliced_sample[key] = value

            # Add the category to the batch_categories list
            batch_categories.append(sliced_sample['category'])

            # Determine how much data we can take from the current `primary_cid` for this temporal window.
            available_from_current = max(0, total_temporal_len_cid - current_temporal_idx)
            take_from_current = min(self.temporal_length, available_from_current)

            # Update the temporal offset for this primary catchment.
            self.current_temporal_offsets[primary_cid] += take_from_current

            # Mark primary_cid as fully exhausted if all its data has been consumed.
            if self.current_temporal_offsets[primary_cid] >= total_temporal_len_cid:
                self.exhausted_catchments.add(primary_cid)


            # --- Temporal Filling Logic (if take_from_current is less than temporal_length) ---
            needed_to_fill = self.temporal_length - take_from_current
            if needed_to_fill > 0:
                # Store the original needed_to_fill to correctly track when padding is necessary
                original_needed_to_fill = needed_to_fill

                # List of filler CIDs to try. Randomize to ensure fair sampling.
                # Crucially, this now considers ALL non-exhausted catchments, not just those
                # that haven't been 'primary' yet.
                candidate_filler_cids = [
                    cid for cid in self.all_catchment_ids # Use all catchments as potential fillers
                    if cid not in self.exhausted_catchments and cid != primary_cid
                    # Removed: and cid not in [s['catchment_id'] for s in batch_samples]
                ]
                self.rng.shuffle(candidate_filler_cids)

                filler_found = False
                for filler_cid in candidate_filler_cids:
                    filler_data_full = self.all_catchment_data_map[filler_cid]
                    filler_current_temporal_offset = self.current_temporal_offsets[filler_cid]
                    filler_total_temporal_len_cid = filler_data_full[self.dates_key_in_template].shape[0]
                    
                    available_from_filler = max(0, filler_total_temporal_len_cid - filler_current_temporal_offset)

                    if available_from_filler > 0:
                        fill_length = min(needed_to_fill, available_from_filler)

                        # Concatenate dates
                        filler_dates_slice = filler_data_full[self.dates_key_in_template][filler_current_temporal_offset : filler_current_temporal_offset + fill_length]
                        temporal_feature_slices[self.dates_key_in_template] = np.concatenate((temporal_feature_slices[self.dates_key_in_template], filler_dates_slice))

                        # Concatenate other temporal features (including x_d sub-features)
                        for key, sliced_val in list(temporal_feature_slices.items()): # Use list to modify dict during iteration
                            if key.startswith('date'): # Already handled dates
                                continue
                            
                            if key.startswith('x_d') and isinstance(sliced_val, dict): # Handle x_d dictionary
                                for sub_k, sub_v in list(sliced_val.items()): # Iterate x_d sub-features
                                    if sub_k in filler_data_full[key] and isinstance(filler_data_full[key][sub_k], torch.Tensor):
                                        filler_sub_tensor_slice = filler_data_full[key][sub_k][filler_current_temporal_offset : filler_current_temporal_offset + fill_length]
                                        temporal_feature_slices[key][sub_k] = torch.cat((sub_v, filler_sub_tensor_slice))
                            elif isinstance(sliced_val, torch.Tensor): # General temporal tensor (like y, additional_temporal_feature)
                                if key in filler_data_full and isinstance(filler_data_full[key], torch.Tensor):
                                    filler_tensor_slice = filler_data_full[key][filler_current_temporal_offset : filler_current_temporal_offset + fill_length]
                                    temporal_feature_slices[key] = torch.cat((sliced_val, filler_tensor_slice))


                        # Update filler_cid's temporal offset
                        self.current_temporal_offsets[filler_cid] += fill_length
                        if self.current_temporal_offsets[filler_cid] >= filler_total_temporal_len_cid:
                            self.exhausted_catchments.add(filler_cid)
                            # If a filler is fully exhausted, remove it from the global_unselected_cids as it cannot be used again
                            # This needs careful thought if filler_cid was from global_unselected_cids.
                            # Best practice: if filler_cid is in global_unselected_cids, remove it.
                            if filler_cid in self.global_unselected_cids:
                                self.global_unselected_cids = deque([c for c in self.global_unselected_cids if c != filler_cid])


                        needed_to_fill -= fill_length # Update remaining needed
                        if needed_to_fill == 0: # If filled completely by this filler, stop
                            filler_found = True
                            break
                
                # If still needed_to_fill after trying all fillers, or no filler found, pad with NaNs.
                if needed_to_fill > 0:
                    # Pad dates with NaT
                    pad_dates = np.full((needed_to_fill,), np.datetime64('NaT'), dtype='datetime64[ns]')
                    temporal_feature_slices[self.dates_key_in_template] = np.concatenate((temporal_feature_slices[self.dates_key_in_template], pad_dates))

                    # Pad all temporal tensors
                    for key, sliced_val in list(temporal_feature_slices.items()):
                        if key.startswith('date'): # Dates already handled for padding
                            continue

                        if key.startswith('x_d') and isinstance(sliced_val, dict):
                            for sub_k in sliced_val: # Iterate through sub-features already in sliced_feature_values
                                current_sub_tensor = sliced_val[sub_k]
                                pad_shape = (needed_to_fill,) + current_sub_tensor.shape[1:]
                                pad_tensor = torch.full(pad_shape, float('nan'), dtype=current_sub_tensor.dtype)
                                temporal_feature_slices[key][sub_k] = torch.cat((current_sub_tensor, pad_tensor))
                        elif isinstance(sliced_val, torch.Tensor):
                            current_tensor = sliced_val
                            pad_shape = (needed_to_fill,) + current_tensor.shape[1:]
                            pad_tensor = torch.full(pad_shape, float('nan'), dtype=current_tensor.dtype)
                            temporal_feature_slices[key] = torch.cat((current_tensor, pad_tensor))
            
            # Transfer the processed temporal features from `temporal_feature_slices` to `sliced_sample`
            for key, value in temporal_feature_slices.items():
                sliced_sample[key] = value


            batch_samples.append(sliced_sample)
            # Put the primary_cid back to the active queue for potential future temporal slices
            # (unless it was just exhausted).
            if primary_cid not in self.exhausted_catchments:
                self.current_spatial_batch_active_cids.append(primary_cid)

        # After forming a full spatial_length batch, calculate and print its distribution
        self._print_batch_distribution(batch_categories)
        # Call collate_fn on the collected samples and return the final collated batch
        return collate_fn(batch_samples)

    def _print_batch_distribution(self, batch_categories: List[str]):
        """
        Calculates and prints the category distribution for the current batch.
        """
        batch_category_counts = defaultdict(int)
        for cat in batch_categories:
            batch_category_counts[cat] += 1
        
        print("\nBatch Category Distribution:")
        if len(batch_categories) > 0:
            for cat in sorted(self.category_proportions.keys()): # Iterate over all known categories
                batch_count = batch_category_counts.get(cat, 0)
                batch_prop = batch_count / len(batch_categories)
                print(f"  {cat}: Count = {batch_count}, Proportion = {batch_prop:.2f}")
        else:
            print("  (Empty batch)")

    def get_all_batches(self) -> List[Dict]:
        """
        Generates all possible batches from the dataset and returns them as a list.
        """
        self._reset_state() # Ensure state is reset for a fresh generation
        all_collated_batches = []
        while True:
            batch = self._generate_single_batch()
            if batch is None:
                break
            all_collated_batches.append(batch)
        return all_collated_batches

