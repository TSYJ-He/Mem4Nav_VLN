import numpy as np
import torch
from typing import Optional, Dict, Tuple, Any
def _split_by_3bits(coord: int) -> int:
    """Interleaves 0s between bits of a 21-bit coordinate for Morton encoding."""
    x = coord & 0x1fffff  # Mask to 21 bits
    x = (x | (x << 42)) & 0x1f00000000000000000ffff # x = ---- ---- ---- ---- ---- --xx xxxx xxxx xxxx xxxx xx-- ---- ---- ---- ---- ---- ---- ---- ---- ---- --xx xxxx xxxx xxxx xxxx xx
    x = (x | (x << 21)) & 0x1f000000000ffff000000000 # x = ---- ---- ---- --xx xxxx xxxx xxxx xxxx xx-- ---- ---- ---- --xx xxxx xxxx xxxx xxxx xx-- ---- ----
    x = (x | (x << 14)) & 0xc00003f00003f00003f0000  # x = --xx xxxx xx-- --xx xxxx xx-- --xx xxxx xx-- --xx xxxx xx-- --xx xxxx xx
    x = (x | (x << 7))  & 0x80c0180c0180c0180c018    # x = -x-x -x-x -x-x -x-x -x-x -x-x -x-x -x-x -x-x -x-x -x-x -x-x -x-x -x-x
    return x

def morton_encode_3d(x: int, y: int, z: int) -> int:
    """
    Encodes 3D integer coordinates into a single Morton code.
    Each coordinate should be within the range [0, 2^max_coord_bits - 1].
    """
    return (_split_by_3bits(z) << 2) | (_split_by_3bits(y) << 1) | _split_by_3bits(x)

def _compact_by_3bits(morton_val: int) -> int:
    """Extracts coordinate from its interleaved Morton representation."""
    x = morton_val & 0x80c0180c0180c0180c018
    x = (x ^ (x >> 7))   & 0xc00003f00003f00003f0000
    x = (x ^ (x >> 14))  & 0x1f000000000ffff000000000
    x = (x ^ (x >> 21))  & 0x1f00000000000000000ffff
    x = (x ^ (x >> 42))  & 0x1fffff # Mask to 21 bits
    return x

def morton_decode_3d(code: int) -> Tuple[int, int, int]:
    """Decodes a Morton code back into 3D integer coordinates."""
    x = _compact_by_3bits(code)
    y = _compact_by_3bits(code >> 1)
    z = _compact_by_3bits(code >> 2)
    return x, y, z

class OctreeLeaf:
    """
    Represents a leaf node in the Sparse Octree.
    It stores its spatial properties and the current Long-Term Memory (LTM) token
    associated with the observations within this voxel.
    """
    __slots__ = ('morton_code', 'center', 'size', 'level', 
                 'ltm_token', # Stores the 2*d_emb LTM state token (output of ReversibleTransformer)
                 'observations_count', 'last_observation_embedding_v_t')

    def __init__(self, morton_code: int, center: np.ndarray, size: float, level: int):
        self.morton_code: int = morton_code
        self.center: np.ndarray = center  # Center coordinates of the voxel [x, y, z]
        self.size: float = size           # Side length of the voxel
        self.level: int = level           # Depth level of this leaf in the octree

  
        self.ltm_token: Optional[torch.Tensor] = None 

        # Tracks basic observation statistics if needed
        self.observations_count: int = 0

        self.last_observation_embedding_v_t: Optional[torch.Tensor] = None

    def update_observation_stats(self, observation_embedding_v_t: torch.Tensor):
        """
        Updates simple statistics about observations in this leaf.
        The LTM token update is handled externally by the LTM system.
        """
        self.last_observation_embedding_v_t = observation_embedding_v_t
        self.observations_count += 1

    def set_ltm_token(self, token: torch.Tensor):
        """
        Sets or updates the Long-Term Memory (LTM) state token for this leaf.
        The provided token is expected to be the 2*d_emb dimensional output
        from the LTM's ReversibleTransformer.
        """
        if token.ndim == 0: # Should not happen if token is 2*d_emb
             print(f"Warning: Attempting to set LTM token with scalar for leaf {self.morton_code}. Token shape: {token.shape}")
        self.ltm_token = token

    def get_ltm_token(self) -> Optional[torch.Tensor]:
        """Returns the current LTM state token for this leaf."""
        return self.ltm_token

    def __repr__(self) -> str:
        ltm_token_shape = self.ltm_token.shape if self.ltm_token is not None else "None"
        return (f"OctreeLeaf(code={self.morton_code}, center={self.center.tolist()}, "
                f"level={self.level}, obs_count={self.observations_count}, "
                f"ltm_token_shape={ltm_token_shape})")

class SparseOctree:
    """
    Hierarchical Sparse Octree for voxel-level indexing of observations.
    Organizes 3D space into a hierarchy of voxels. Only leaf voxels that
    are visited or contain relevant observations are instantiated.
    Each leaf can store an associated LTM token.
    """
    def __init__(self, world_size: float, max_depth: int, 
                 # embedding_dim is not directly used by Octree itself for LTM token storage,
                 # as LTM manages its token dimensions. It's here for legacy or if OctreeLeaf
                 # directly stored raw embeddings previously.
                 embedding_dim_for_v_t: int, # Dimension of observation v_t, primarily for type hints if needed
                 device: torch.device = torch.device('cpu')):
        """
        Initializes the Sparse Octree.

        Args:
            world_size (float): The side length of the cubic world space.
            max_depth (int): The maximum depth of the octree (Lambda).
            embedding_dim_for_v_t (int): Dimensionality of the raw observation embeddings (v_t)
                                         that might be passed to update_observation_stats.
            device (torch.device): The PyTorch device.
        """
        if not (0 < max_depth <= 21):
            raise ValueError("max_depth must be between 1 and 21 for 64-bit Morton codes.")

        self.world_size: float = float(world_size)
        self.max_depth: int = int(max_depth)
        self.device: torch.device = device # Though OctreeLeaf stores tensors, Octree itself is mainly dicts/numpy

        # Leaf nodes are stored in a hash map, keyed by their Morton code.
        self.leaves: Dict[int, OctreeLeaf] = {}
        self.leaf_voxel_size: float = self.world_size / (2**self.max_depth)

    def _quantize_point(self, point: np.ndarray) -> Tuple[int, int, int]:
        """Quantizes a continuous 3D point to integer grid coordinates at max_depth."""
        normalized_point = np.clip(point / self.world_size, 0.0, 1.0 - 1e-9)
        scale = 2**self.max_depth
        quantized_coords = np.floor(normalized_point * scale).astype(np.int64)
        max_coord_val = scale - 1
        quantized_coords = np.clip(quantized_coords, 0, max_coord_val)
        return tuple(quantized_coords.tolist()) #type: ignore

    def _get_leaf_center_and_level(self, quantized_coords: Tuple[int, int, int]) -> Tuple[np.ndarray, float, int]:
        """Calculates the center, size, and level of a leaf voxel."""
        level = self.max_depth
        size = self.leaf_voxel_size
        center = (np.array(quantized_coords, dtype=np.float32) + 0.5) * size
        return center, size, level

    def _get_morton_code(self, point: np.ndarray) -> int:
        """Computes the Morton code for a leaf voxel containing the given point."""
        qx, qy, qz = self._quantize_point(point)
        return morton_encode_3d(qx, qy, qz)

    def insert_or_get_leaf(self, point_xyz_np: np.ndarray, 
                           current_observation_embedding_v_t: Optional[torch.Tensor] = None) -> OctreeLeaf:
        """
        Retrieves an existing leaf or creates a new one for the voxel containing the point.
        Updates basic observation statistics on the leaf if an embedding is provided.
        The LTM token associated with this leaf is managed and set externally via `leaf.set_ltm_token()`.

        Args:
            point_xyz_np (np.ndarray): The 3D point (x, y, z) of the observation.
            current_observation_embedding_v_t (Optional[torch.Tensor]): The d_emb-dimensional raw embedding
                                                                      of the current observation (v_t).
                                                                      Used to update `leaf.last_observation_embedding_v_t`.

        Returns:
            OctreeLeaf: The leaf node corresponding to the point.
        """
        morton_code = self._get_morton_code(point_xyz_np)

        if morton_code not in self.leaves:
            quantized_coords = self._quantize_point(point_xyz_np)
            center, size, level = self._get_leaf_center_and_level(quantized_coords)
            new_leaf = OctreeLeaf(morton_code, center, size, level)
            self.leaves[morton_code] = new_leaf
            # Initial LTM token for a new leaf is None. It will be set by the MappingModule
            # after the first LTM write operation for this key (morton_code).

        leaf = self.leaves[morton_code]
        
        if current_observation_embedding_v_t is not None:
            # Ensure the observation embedding is on the correct device if it's a tensor.
            # OctreeLeaf itself doesn't enforce device for this, but consistency is good.
            obs_emb_device = current_observation_embedding_v_t.to(self.device) if hasattr(current_observation_embedding_v_t, 'to') else current_observation_embedding_v_t
            leaf.update_observation_stats(obs_emb_device)
        
        return leaf

    def get_leaf_by_point(self, point_xyz_np: np.ndarray) -> Optional[OctreeLeaf]:
        """Retrieves the octree leaf containing the given point, if it exists."""
        morton_code = self._get_morton_code(point_xyz_np)
        return self.leaves.get(morton_code)

    def get_leaf_by_code(self, morton_code: int) -> Optional[OctreeLeaf]:
        """Retrieves an octree leaf by its Morton code, if it exists."""
        return self.leaves.get(morton_code)

    def query_radius(self, center_point_np: np.ndarray, radius: float) -> List[OctreeLeaf]:
        """
        Retrieves all instantiated leaf nodes whose centers are within a given radius.
        """
        if radius < 0: return []
        nearby_leaves = []
        radius_squared = radius * radius
        query_p = np.array(center_point_np, dtype=np.float32)

        for leaf in self.leaves.values():
            dist_squared = np.sum((leaf.center - query_p)**2)
            if dist_squared <= radius_squared:
                nearby_leaves.append(leaf)
        return nearby_leaves

    def clear(self):
        """Clears all leaves from the octree."""
        self.leaves.clear()

    def __len__(self) -> int:
        """Returns the number of instantiated leaf nodes."""
        return len(self.leaves)

    def get_all_leaves(self) -> List[OctreeLeaf]:
        """Returns a list of all instantiated leaf nodes."""
        return list(self.leaves.values())

if __name__ == '__main__':
    # Example Usage
    # Dimension of v_t (fused observation embedding from perception)
    # This is relevant for OctreeLeaf.last_observation_embedding_v_t
    d_emb_for_v_t = 384 
    test_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    octree = SparseOctree(world_size=100.0, max_depth=16, 
                          embedding_dim_for_v_t=d_emb_for_v_t, device=test_device)

    point1_np = np.array([10.0, 20.0, 30.0])
    embedding_v1 = torch.rand(d_emb_for_v_t, device=test_device)

    # Insert into octree
    leaf1 = octree.insert_or_get_leaf(point1_np, embedding_v1)
    print(f"Inserted/Got Leaf 1: {leaf1}")
    assert leaf1.ltm_token is None # Initially no LTM token
    assert torch.equal(leaf1.last_observation_embedding_v_t, embedding_v1) # type: ignore
    assert leaf1.observations_count == 1

    # Simulate LTM system providing a token for this leaf
    # LTM token dimension is typically 2 * d_emb (e.g., 2 * 128 if LTM's internal d_emb is 128)
    # Let's assume LTM d_emb used for reversible transformer input parts is 128, so token is 256D.
    # This should align with LTM's `ltm_token_dim`.
    # For testing here, let's use a dummy LTM token dimension, e.g., 512.
    dummy_ltm_token_dim = 512 
    mock_ltm_token_for_leaf1 = torch.randn(dummy_ltm_token_dim, device=test_device)
    leaf1.set_ltm_token(mock_ltm_token_for_leaf1)
    print(f"  Set LTM token for Leaf 1. Shape: {leaf1.get_ltm_token().shape}") # type: ignore
    assert torch.equal(leaf1.get_ltm_token(), mock_ltm_token_for_leaf1) # type: ignore

    # Observe again at the same location (or close enough to be in the same voxel)
    point2_np = np.array([10.1, 20.1, 30.1]) 
    embedding_v2 = torch.rand(d_emb_for_v_t, device=test_device)
    leaf2 = octree.insert_or_get_leaf(point2_np, embedding_v2)
    
    print(f"Updated Leaf 2 (should be same as Leaf 1): {leaf2}")
    assert leaf1 is leaf2 # Should be the same object
    assert leaf2.observations_count == 2
    assert torch.equal(leaf2.last_observation_embedding_v_t, embedding_v2) # type: ignore
    # LTM token would still be the one from the previous update until explicitly set again
    assert torch.equal(leaf2.get_ltm_token(), mock_ltm_token_for_leaf1) # type: ignore

    # Simulate LTM system providing an updated token for this leaf
    mock_updated_ltm_token = torch.randn(dummy_ltm_token_dim, device=test_device)
    leaf2.set_ltm_token(mock_updated_ltm_token)
    print(f"  Updated LTM token for Leaf 1/2. Shape: {leaf2.get_ltm_token().shape}") #type: ignore
    assert torch.equal(leaf2.get_ltm_token(), mock_updated_ltm_token) # type: ignore

    # Test retrieval
    retrieved_leaf = octree.get_leaf_by_code(leaf1.morton_code)
    assert retrieved_leaf is leaf1
    if retrieved_leaf:
        print(f"Retrieved Leaf by code: {retrieved_leaf.morton_code}, LTM token shape: {retrieved_leaf.get_ltm_token().shape}") # type: ignore

    print("\nSparseOctree with modified OctreeLeaf tests completed.")
