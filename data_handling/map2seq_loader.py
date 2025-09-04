import torch
import json
import os
from typing import Dict, Any, Optional, List
try:
    from .base_loader import BaseVLNDataset
    # from .utils import tokenize_instructions # Tokenization is handled by base class
except ImportError:
    print("Warning: Map2SeqDatasetLoader using placeholder for BaseVLNDataset due to import error.")
    class BaseVLNDataset(torch.utils.data.Dataset):   
        def __init__(self, config, split, tokenizer=None, max_instruction_length=128, name="Base"):
            self.config = config.get(name.lower(), config)
            self.dataset_root_path = self.config.get('dataset_path', './datasets')
            self.specific_dataset_path = self.config.get('specific_path') # Path to Map2Seq VLN splits
            self.split = split
            self.tokenizer = tokenizer
            self.max_instruction_length = max_instruction_length
            self.data: List[Dict[str, Any]] = []
            self._load_data()
        def _load_data(self): raise NotImplementedError
        def __len__(self): return len(self.data)
        def __getitem__(self, index): return self._preprocess_instance(self.data[index])
        def _preprocess_instance(self, instance_data):
            if self.tokenizer and 'instruction_text' in instance_data:
                mock_tokens = torch.randint(0, 100, (self.max_instruction_length,))
                mock_attn_mask = torch.ones(self.max_instruction_length, dtype=torch.long)
                instance_data['instruction_tokens_ids'] = mock_tokens
                instance_data['instruction_attention_mask'] = mock_attn_mask
            return instance_data
    # PreTrainedTokenizerBase placeholder if not imported by base_loader
    try:
        from transformers import PreTrainedTokenizerBase
    except ImportError:
        PreTrainedTokenizerBase = Any   


class Map2SeqDatasetLoader(BaseVLNDataset):
    """
    Dataset loader for the Map2Seq Vision-and-Language Navigation dataset.
    Loads data from the VLN splits of Map2Seq.
    """
    def __init__(self,
                 config: Dict[str, Any],
                 split: str,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,   
                 max_instruction_length: int = 128,
                 name: str = "Map2SeqDatasetLoader"):
        """
        Initializes the Map2Seq dataset loader.

        Args:
            config (Dict[str, Any]): Configuration dictionary. Expected to contain:
                - 'specific_path': Path to the root of the Map2Seq VLN splits directory.
                                   E.g., /path/to/map2seq_vln_splits/
                - 'landmarks_file_path' (Optional): Path to a JSON file containing
                                                     pre-extracted landmarks for episodes.
                                                     (Relative to dataset_root_path or specific_path)
            split (str): The dataset split to load (e.g., "train", "dev_seen", "dev_unseen",
                         "test_seen", "test_unseen"). Map2Seq often has seen/unseen dev/test.
            tokenizer (Optional[PreTrainedTokenizerBase]): Tokenizer for instructions.
            max_instruction_length (int): Max length for tokenized instructions.
            name (str): Name for this loader instance.
        """
        self.landmarks_data: Optional[Dict[str, Any]] = None
        # Config for this specific loader instance
        loader_config = config.get('map2seq_loader', {})
        self.landmarks_file_path = loader_config.get('landmarks_file_path')

        super().__init__(config, split, tokenizer, max_instruction_length, name)
        # Note: self.specific_dataset_path should point to the root of the Map2Seq VLN splits directory.

    def _load_data(self) -> None:
        """
        Loads Map2Seq data from JSON line files (VLN splits).
        Populates `self.data` with a list of episode dictionaries.
        """
        if not self.specific_dataset_path:
            raise ValueError(f"Configuration for '{self.name}' must specify 'specific_path' to the Map2Seq VLN splits directory.")

        data_file_path = os.path.join(self.specific_dataset_path, f"{self.split}.json")
        
        if not os.path.exists(data_file_path):
            # Some older structures might have a 'data/' subdirectory, though less common for Map2Seq splits
            data_file_path_alt = os.path.join(self.specific_dataset_path, "data", f"{self.split}.json")
            if not os.path.exists(data_file_path_alt):
                raise FileNotFoundError(f"Map2Seq data file not found at '{data_file_path}' or '{data_file_path_alt}' for split '{self.split}'.")
            data_file_path = data_file_path_alt
            
        print(f"Loading Map2Seq data from: {data_file_path}")

        with open(data_file_path, 'r') as f:
            for line in f:
                try:
                    instance = json.loads(line.strip())
                    self.data.append(instance)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping line due to JSON decode error: {e} in file {data_file_path}")
        
        if not self.data:
            print(f"Warning: No data loaded for Map2Seq split '{self.split}' from '{data_file_path}'.")
            return

        # Load landmarks if a landmarks file path is provided
        if self.landmarks_file_path:
            lm_path = os.path.join(self.dataset_root_path, self.landmarks_file_path) # Relative to dataset_root_path
            if not os.path.exists(lm_path):
                lm_path_alt = os.path.join(self.specific_dataset_path, self.landmarks_file_path) # Relative to specific_path
                if not os.path.exists(lm_path_alt):
                     print(f"Warning: Landmarks file '{self.landmarks_file_path}' not found at '{lm_path}' or '{lm_path_alt}'. Proceeding without pre-extracted landmarks.")
                else:
                    lm_path = lm_path_alt
            
            if os.path.exists(lm_path):
                print(f"Loading landmarks from: {lm_path}")
                try:
                    with open(lm_path, 'r') as f_lm:
                        # VELMA landmarks file is structured as {'instances': {map_id_instr_id: {'unfiltered': [...]}}}
                        # Map2Seq 'id' field is often like "map_id_instruction_idx"
                        loaded_landmarks_data = json.load(f_lm)
                        if 'instances' in loaded_landmarks_data:
                            self.landmarks_data = loaded_landmarks_data['instances']
                        else:
                            self.landmarks_data = loaded_landmarks_data 
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse landmarks JSON file '{lm_path}': {e}")
                except Exception as e:
                    print(f"Warning: Error loading landmarks file '{lm_path}': {e}")

        print(f"Map2SeqDatasetLoader: Loaded {len(self.data)} instances for split '{self.split}'.")


    def _preprocess_instance(self, instance_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocesses a single Map2Seq data instance.
        """
        # Call base class preprocessing (handles instruction tokenization via 'instruction_text')
        # Ensure 'instruction_text' field exists for base class
        if 'navigation_text' in instance_data and 'instruction_text' not in instance_data:
            instance_data['instruction_text'] = instance_data['navigation_text']
        elif 'instruction_text' not in instance_data:
            instance_data['instruction_text'] = "" # Default to empty if completely missing

        processed_instance = super()._preprocess_instance(instance_data)

        # Ensure essential Map2Seq fields
        if 'id' not in processed_instance:
            # Map2Seq instances usually have an 'id' like 'mapid_instructionidx'
            # or 'mapid' and 'instructions_id' / 'instruction_idx' separately
            # For now, if 'id' is missing, we'll try to construct one or warn.
            map_id = processed_instance.get('map_id', processed_instance.get('map'))
            instr_idx = processed_instance.get('instruction_idx', processed_instance.get('instructions_id'))
            if map_id is not None and instr_idx is not None:
                processed_instance['id'] = f"{map_id}_{instr_idx}"
            else:
                # print(f"Warning: 'id' field missing and could not be constructed for instance: {instance_data.get('instruction_text','N/A')[:50]}")
                pass # Allow to pass, subsequent code should handle missing 'id' if critical.

        # Target panoid
        if 'route_panoids' in processed_instance and processed_instance['route_panoids']:
            processed_instance['target_panoid'] = processed_instance['route_panoids'][-1]
        else:
            processed_instance['route_panoids'] = []
            processed_instance['target_panoid'] = None
            # print(f"Warning: 'route_panoids' missing or empty for instance_id {processed_instance.get('id')}")

        # Start heading (ensure it's a float)
        if 'start_heading' in processed_instance:
            try:
                processed_instance['start_heading'] = float(processed_instance['start_heading'])
            except ValueError:
                print(f"Warning: Could not convert start_heading '{processed_instance['start_heading']}' to float for instance {processed_instance.get('id')}. Setting to 0.0.")
                processed_instance['start_heading'] = 0.0
        else:
            processed_instance['start_heading'] = 0.0

        # Add landmarks if available
        instance_id_str = str(processed_instance.get('id', '')) # Map2Seq ID is often string like "mapID_instrNo"
        if self.landmarks_data and instance_id_str in self.landmarks_data:
            landmark_info = self.landmarks_data[instance_id_str]
            if isinstance(landmark_info, dict) and 'unfiltered' in landmark_info:
                 processed_instance['landmarks_unfiltered'] = landmark_info['unfiltered']
            else:
                processed_instance['landmarks_raw'] = landmark_info
        else:
            processed_instance['landmarks_unfiltered'] = []

        # Add 'is_map2seq' flag, sometimes used by shared agent codebases
        processed_instance['is_map2seq'] = True
        
        return processed_instance

if __name__ == '__main__':
    print("--- Testing Map2SeqDatasetLoader ---")

    dummy_dataset_root_m2s = "./tmp_datasets_m2s"
    # Map2Seq specific path usually points to the VLN splits directory
    dummy_map2seq_specific_path = os.path.join(dummy_dataset_root_m2s, "map2seq_vln_splits")
    os.makedirs(dummy_map2seq_specific_path, exist_ok=True)

    mock_config_m2s = {
        'dataset_path': dummy_dataset_root_m2s,
        'Map2SeqDatasetLoader': {
            'specific_path': dummy_map2seq_specific_path,
            'landmarks_file_path': 'landmarks/map2seq_landmarks_mock.json' # Relative to dataset_root_path
        }
    }

    # Create dummy dev_seen.json for Map2Seq
    dummy_dev_seen_data = [
        {"id": "map1_instr0", "navigation_text": "Go straight and turn right at the cafe.", "route_panoids": ["p1", "p2", "p3"], "start_heading": 0.0, "map_id": "map1", "instruction_idx": 0},
        {"id": "map1_instr1", "navigation_text": "Walk towards the park.", "route_panoids": ["pA", "pB", "pC"], "start_heading": "180"},
        {"id": "map2_instr0", "navigation_text": "Stop by the fountain.", "route_panoids": ["pX"], "start_heading": 270.0, "map_id": "map2"}
    ]
    with open(os.path.join(dummy_map2seq_specific_path, "dev_seen.json"), "w") as f:
        for item in dummy_dev_seen_data:
            f.write(json.dumps(item) + "\n")

    # Create dummy landmarks file
    dummy_landmarks_dir_m2s = os.path.join(dummy_dataset_root_m2s, "landmarks")
    os.makedirs(dummy_landmarks_dir_m2s, exist_ok=True)
    dummy_landmarks_data_m2s = {
        "instances": {
            "map1_instr0": {"unfiltered": ["cafe"]},
            "map1_instr1": {"unfiltered": ["park"]}
        }
    }
    with open(os.path.join(dummy_landmarks_dir_m2s, "map2seq_landmarks_mock.json"), "w") as f:
        json.dump(dummy_landmarks_data_m2s, f)

    # Mock Tokenizer (can reuse the one from Touchdown test for this example)
    class MockTokenizerForM2S:
        def __init__(self, max_len=25): self.max_length = max_len; self.pad_token_id=0
        def __call__(self, text_list, padding, truncation, max_length, return_tensors, return_attention_mask):
            input_ids_batch = []
            attention_mask_batch = []
            for text in text_list if isinstance(text_list, list) else [text_list]:
                tokens = [hash(t) % 1000 + 1 for t in text.split()[:max_length]]
                padded_tokens = tokens + [self.pad_token_id] * (max_length - len(tokens))
                attn_mask = [1] * len(tokens) + [0] * (max_length - len(tokens))
                input_ids_batch.append(padded_tokens)
                attention_mask_batch.append(attn_mask)
            return {
                'input_ids': torch.tensor(input_ids_batch, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask_batch, dtype=torch.long)
            }
    
    mock_tokenizer_m2s = MockTokenizerForM2S(max_len=30)

    try:
        m2s_dataset = Map2SeqDatasetLoader(
            config=mock_config_m2s, 
            split="dev_seen", 
            tokenizer=mock_tokenizer_m2s, 
            max_instruction_length=30
        )
        print(f"Map2Seq Dataset size: {len(m2s_dataset)}")
        assert len(m2s_dataset) == 3

        if len(m2s_dataset) > 0:
            sample_item_0 = m2s_dataset[0]
            print(f"\nSample item [0]: ID {sample_item_0['id']}")
            print(f"  Instruction: {sample_item_0['instruction_text']}")
            print(f"  Tokens shape: {sample_item_0['instruction_tokens_ids'].shape}")
            print(f"  Target Pano: {sample_item_0['target_panoid']}")
            print(f"  Landmarks: {sample_item_0.get('landmarks_unfiltered')}")
            assert sample_item_0['instruction_tokens_ids'].shape == (30,)
            assert sample_item_0['target_panoid'] == "p3"
            assert "cafe" in sample_item_0.get('landmarks_unfiltered', [])
            assert sample_item_0['is_map2seq'] is True

            sample_item_1 = m2s_dataset[1]
            print(f"\nSample item [1]: ID {sample_item_1['id']}")
            print(f"  Start Heading: {sample_item_1['start_heading']} (type: {type(sample_item_1['start_heading'])})")
            assert isinstance(sample_item_1['start_heading'], float)
            assert "park" in sample_item_1.get('landmarks_unfiltered', [])

    except Exception as e:
        print(f"Error during Map2SeqDatasetLoader test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        import shutil
        if os.path.exists(dummy_dataset_root_m2s):
            shutil.rmtree(dummy_dataset_root_m2s)

    print("\nMap2SeqDatasetLoader tests completed.")
