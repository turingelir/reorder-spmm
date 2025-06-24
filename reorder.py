r"""
    Matrix Reordering Module

    
    By Fatih Said Duran, 2025
"""
import cugraph
import cudf
import networkx as nx
from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix, isspmatrix_csr, triu
from scipy.sparse.csgraph import reverse_cuthill_mckee
from sksparse.cholmod import cholesky
import pymetis

import sys
import os
import gc
import numpy as np
import argparse
import signal
from typing import List, Tuple, Dict

MAX_SUBMATRIX_SIZE = 1000000  # or set via env/config
NUM_CLUSTERS = 8  # Default number of clusters for METIS, can be overridden by environment variable

class MatrixReorderer:
    """
    A class to handle matrix reordering using various methods.
    
    Attributes
    ----------
    matrix : scipy.sparse.csr_matrix
        The matrix to be reordered.
    method : callable
        The reordering method to apply.
    """
    
    def __init__(self, path:str, reordering_method:str, partitioning_method:str=None,
                num_clusters:int=NUM_CLUSTERS, repeats:int=1,
                 res_limit:float=1.0, ordering_place:str='local',
                 allow_preprocessing:bool=True, allow_metrics:bool=True,
                 aggressive_gc:bool=False
                 ):
        """
        Initialize the MatrixReorderer with a matrix file and a reordering method.
        Parameters
        ----------
        path : str
            The path to the Matrix Market file containing the matrix.
        method : str
            The reordering method to use ('amd', 'symamd', 'rcm').
        metrics : string of metric names, optional
            The metrics to compute on the matrix (e.g., 'nnz', 'density').
        """
        self.dir = os.path.dirname(os.path.abspath(path))
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist.")
        self.file_name = os.path.basename(path)
        self.reorder_method_name = reordering_method
        self.partition_method_name = partitioning_method
        
        # Build comprehensive method name including all parameters that affect output
        if partitioning_method:
            method_parts = [partitioning_method, ordering_place, reordering_method]
            # Add resolution limit for Louvain method
            if partitioning_method == 'louvain':
                method_parts.append(f"res{res_limit}")
            self.method_name = "_".join(method_parts)
        else:
            self.method_name = reordering_method

        self.conditions:dict = None

        self.ordering_place = ordering_place # 'local' or 'global' 
        print(f"Ordering place: {self.ordering_place}")
        print(f"Partitioning method: {self.partition_method_name}")
        if not(ordering_place == 'global' or (ordering_place == 'local' and partitioning_method is not None)):
            print(f"Invalid combination of ordering place '{ordering_place}' and partitioning method '{partitioning_method}'. ")
            exit(0) # Continue like nothing happened

        self.reorder = self.select_reordering_method(reordering_method) # Separate partitioning and reordering methods
        if not callable(self.reorder):
            raise ValueError(f"Method {reordering_method} is not callable.")
        self.partition = self.select_partitioning_method(partitioning_method) 
        if not callable(self.partition):
            raise ValueError(f"Partitioning method {partitioning_method} is not callable.")
        
        self.num_clusters = num_clusters
        self.res_limit = res_limit
        self.repeats = repeats  # TODO: Number of repeats for clustering methods (e.g., Louvain)

        self.PREPROCESSING = allow_preprocessing  # TODO: Flag to indicate if preprocessing is allowed (like making symmetric)
        self.METRICS = allow_metrics # TODO: Flag to indicate if metrics should be computed
        self.AGGRESSIVE_GC = aggressive_gc  # Flag to enable more aggressive garbage collection
        
        self.metrics = dict()  # Store metrics as a dictionary

        self.logger = None  # TODO: Logger for debugging and information

        self.perm = None

        self.symmetric = True
    
    def __call__(self, matrix) -> np.ndarray:
        """
        Call the reordering method directly on the matrix.

        Returns
        -------
        np.ndarray
            The permutation indices.
        """
        self.check_matrix(matrix)
        return self.reorder(matrix)[1]
    
    def __str__(self):
        """
        String representation of the MatrixReorderer.
        """
        return f"MatrixReorderer(method={self.method_name}, file={self.file_name}, conditions={self.conditions})"
    
    def hybrid_reorder(self, matrix, store_perm:bool=False) -> Tuple[csr_matrix, np.ndarray]:
        """
        Apply the selected reordering method to the matrix.

        Returns
        -------
        tuple
            The reordered matrix and the permutation indices.
        """
        perm = self.permutation(matrix)
        if store_perm:
            self.perm = perm
            
        # Force garbage collection before matrix reordering
        gc.collect()
        
        reordered_matrix = matrix[perm, :][:, perm]
        
        return reordered_matrix, perm

    def permutation(self, matrix) -> np.ndarray:
        """
        General method to compute the permutation based on the selected partitioning and reordering methods.
        Parameters
        ----------
        matrix : scipy.sparse.csr_matrix
            The matrix to reorder.
        Returns
        -------
        np.ndarray
            The permutation indices.
        """
        # If reordering is global, do it first
        if self.ordering_place == 'global':
            perm = self.reorder(matrix) # This will return the permutation indices

        # If partitioning is specified, apply it
        if self.partition_method_name:
            if self.ordering_place == 'global':
                matrix = matrix[perm, :][:, perm]

            # Find partitions of the matrix
            parts_idx = self.partition(matrix) # Arrays of indices for each partition

            num_parts = len(parts_idx)

            perm = []
            for part_id in range(num_parts):
                indices = parts_idx[part_id]
                
                if len(indices) == 0:
                    continue
                if len(indices) > MAX_SUBMATRIX_SIZE:
                    print(f"Warning: Skipping partition {part_id} with {len(indices)} nodes (too large for memory)")
                    continue
                # Extract the submatrix for the current partition
                submatrix = matrix[indices, :][:, indices]

                if self.PREPROCESSING and not self.symmetric:
                    submatrix = submatrix + submatrix.T

                # If the reordering method is local 
                if self.ordering_place == 'local':
                    subperm = self.reorder(submatrix)
                    # Map subperm indices back to global indices
                    global_subperm = indices[subperm]
                    perm.extend(global_subperm.tolist())
                else:  # If global, we already reordered the matrix
                    # Get the subperm indices from the global permutation
                    subperm = np.arange(len(indices))
                    global_subperm = indices[subperm]
                    perm.extend(global_subperm.tolist())

            return np.array(perm)
        # No partitioning, just return the global permutation
        return perm if self.ordering_place == 'global' else np.arange(matrix.shape[0])

    def amd_permutation(self, matrix) -> np.ndarray:
        """ 
        Compute AMD permutation using scikit-sparse.cholmod."""
        try:
            factor = cholesky(matrix)  # Requires symmetric positive-definite matrix
        except Exception as e:
            raise ValueError("Matrix is not positive definite.") from e
        return factor.P()    
    
    def metis_partition(self, matrix, nparts=None) -> List[np.ndarray]:
        """
        Compute METIS partitioning of the matrix.

        Parameters
        ----------
        matrix : scipy.sparse.csr_matrix
            The matrix to partition.
        nparts : int, optional
            The number of partitions to create (default is self.num_clusters).

        Returns
        -------
        List[np.ndarray]
            A list of arrays, where each array contains the indices of the nodes in that partition.
        """
        try:
            print(f"Starting METIS partitioning for matrix {self.file_name}")
            
            # Force garbage collection before starting
            gc.collect()
            
            # Validate matrix first
            if not self._validate_matrix_for_metis(matrix):
                print("Matrix validation failed, falling back to single partition")
                return [np.arange(matrix.shape[0])]
            
            # Make sure matrix is symmetric (required for METIS)
            if not self._is_symmetric(matrix):
                print("Warning: Matrix is not symmetric, making it symmetric for METIS...")
                matrix = (matrix + matrix.T) / 2
                matrix.eliminate_zeros()
                # Force garbage collection after matrix operations
                gc.collect()
            
            print("Converting to adjacency list...")
            adj = self.csr_to_adj_list(matrix)
            
            # Force garbage collection after adjacency list creation
            gc.collect()
            
            # Validate adjacency list
            if not adj or len(adj) != matrix.shape[0]:
                raise ValueError("Invalid adjacency list generated")
            
            if nparts is None:
                nparts = self.num_clusters
                
            # Ensure nparts is reasonable
            n_nodes = matrix.shape[0]
            nparts = min(nparts, max(1, n_nodes // 10))  # At least 10 nodes per partition
            if nparts <= 1:
                print("Only 1 partition needed, returning single partition")
                return [np.arange(matrix.shape[0])]
            
            print(f"Running METIS partitioning with {nparts} partitions on matrix of size {matrix.shape[0]}x{matrix.shape[1]}")
            
            # Force garbage collection before calling METIS
            gc.collect()
            
            # Add timeout protection - if METIS hangs, we'll catch it
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("METIS partitioning timed out")
            
            # Set timeout to 60 seconds
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)
            
            try:
                _, parts = pymetis.part_graph(nparts=nparts, adjacency=adj)
                signal.alarm(0)  # Cancel timeout
                
                # Force garbage collection after METIS call
                gc.collect()
                
            except TimeoutError:
                print("METIS partitioning timed out, falling back to single partition")
                return [np.arange(matrix.shape[0])]

            # Convert parts to list of arrays for each partition (standardized format)
            parts = np.array(parts)
            parts_idx = []
            for part_id in range(nparts):
                indices = np.where(parts == part_id)[0]
                if len(indices) > 0:  # Only include non-empty partitions
                    parts_idx.append(indices)
            
            # Final garbage collection before returning
            gc.collect()
            
            print(f"METIS partitioning completed successfully, created {len(parts_idx)} partitions")
            return parts_idx
            
        except Exception as e:
            print(f"METIS partitioning failed: {str(e)}")
            print("Falling back to simple partitioning method")
            # Force garbage collection before fallback
            gc.collect()
            # Fallback: use simple partitioning instead of single partition
            try:
                return self.simple_partition(matrix, nparts)
            except Exception as e2:
                print(f"Simple partitioning also failed: {str(e2)}")
                print("Using single partition as final fallback")
                return [np.arange(matrix.shape[0])]

    def louvain_partition(self, matrix) -> List[np.ndarray]:
        """
        Compute Louvain partitioning of the matrix.

        Parameters
        ----------
        matrix : scipy.sparse.csr_matrix
            The matrix to partition.

        Returns
        -------
        List[np.ndarray]
            A list of arrays, where each array contains the indices of the nodes in that partition.
        """
        # Ensure non-negative weights
        matrix.data[matrix.data < 0] = 0

        # Convert sparse matrix to cuDF edge list format for cuGraph
        coo = matrix.tocoo()
        edge_df = cudf.DataFrame({
            'src': coo.row,
            'dst': coo.col
        })

        # Create cuGraph Graph
        G = cugraph.Graph()
        G.from_cudf_edgelist(edge_df, source='src', destination='dst')

        # TODO: Repeat a number of times and take the best partitioning in terms of modularity
        best_modularity = -np.inf
        best_partition_df = None
        for _ in range(self.repeats):
            # Run Louvain community detection using cuGraph
            partition_df, modularity = cugraph.louvain(G, resolution=self.res_limit)
            if modularity > best_modularity:
                best_modularity = modularity
                best_partition_df = partition_df

        # Convert to numpy arrays for efficient processing (METIS-style)
        vertices = partition_df['vertex'].to_numpy()
        partitions = partition_df['partition'].to_numpy()
        
        # Get unique partition IDs
        unique_partitions = np.unique(partitions)
        NUM_CLUSTERS = len(unique_partitions)
        self.num_clusters = NUM_CLUSTERS
        os.environ['NUM_CLUSTERS'] = str(NUM_CLUSTERS)

        # TODO: Log modularity and number of partitions (call logger method)
        print(f"Louvain modularity: {best_modularity}, resolution: {self.res_limit}, number of partitions: {NUM_CLUSTERS}")

        # Convert to standardized format: list of arrays for each partition
        parts_idx = []
        for part_id in unique_partitions:
            # Use direct numpy indexing for optimal performance
            indices = vertices[partitions == part_id]
            if len(indices) > 0:  # Only include non-empty partitions
                parts_idx.append(indices)
        
        return parts_idx

    def metis_rcm_permutation(self, matrix, nparts=NUM_CLUSTERS) -> np.ndarray:
        """ DEPRECATED
        Compute METIS partitioning and then apply RCM on each partition."""
        adj = self.csr_to_adj_list(matrix)
        _, parts = pymetis.part_graph(nparts=nparts, adjacency=adj)

        perm = []
        parts = np.array(parts)
        for part_id in range(nparts):
            indices = np.where(parts == part_id)[0]
            if len(indices) == 0:
                continue
            if len(indices) > MAX_SUBMATRIX_SIZE:
                print(f"Warning: Skipping cluster {part_id} with {len(indices)} nodes (too large for memory)")
                continue
            submatrix = matrix[indices, :][:, indices]
            if not self.symmetric:
                submatrix = submatrix + submatrix.T  # Ensure symmetry for RCM

            # Apply RCM on submatrix
            subperm = self.best_rcm_permutation(submatrix)
            # Map subperm indices back to global indices
            global_subperm = indices[subperm]
            perm.extend(global_subperm.tolist())

        return np.array(perm) 
    
    def louvain_rcm_permutation(self, matrix): 
        """ DEPRECATED
        Compute Louvain partitioning and then apply RCM on each partition."""
        # Ensure non-negative weights
        matrix.data[matrix.data < 0] = 0

        # Convert sparse matrix to cuDF edge list format for cuGraph
        coo = matrix.tocoo()
        edge_df = cudf.DataFrame({
            'src': coo.row,
            'dst': coo.col
        })

        # Create cuGraph Graph
        G = cugraph.Graph()
        G.from_cudf_edgelist(edge_df, source='src', destination='dst')

        # Run Louvain community detection using cuGraph
        partition_df, modularity = cugraph.louvain(G, resolution=self.res_limit)

        # Convert to numpy arrays for efficient processing (METIS-style)
        vertices = partition_df['vertex'].to_numpy()
        partitions = partition_df['partition'].to_numpy()
        
        # Get unique partition IDs
        unique_partitions = np.unique(partitions)
        NUM_CLUSTERS = len(unique_partitions)
        self.num_clusters = NUM_CLUSTERS
        os.environ['NUM_CLUSTERS'] = str(NUM_CLUSTERS)  # export env var

        perm = []
        for part_id in unique_partitions:
            # Use direct numpy indexing (similar to METIS approach)
            indices = vertices[partitions == part_id]
            if len(indices) == 0:
                continue
            if len(indices) > MAX_SUBMATRIX_SIZE:
                print(f"Warning: Skipping cluster {part_id} with {len(indices)} nodes (too large for memory)")
                continue
                
            submatrix = matrix[indices, :][:, indices]

            if not self.symmetric:
                submatrix = submatrix + submatrix.T  # Ensure symmetry for RCM

            subperm = self.best_rcm_permutation(submatrix)
            global_subperm = indices[subperm]
            perm.extend(global_subperm.tolist())

        return np.array(perm)
    
    def csr_to_adj_list(self, A):
        """
        Convert CSR matrix to METIS-compatible adjacency list.

        Returns
        -------
        List[List[int]]
            An adjacency list representation of the graph.
        """
        n = A.shape[0]
        adj = []
        
        # Ensure the matrix is in the right format
        if not isspmatrix_csr(A):
            A = A.tocsr()
        
        # Remove explicit zeros
        A.eliminate_zeros()
        
        # Force garbage collection before processing large adjacency list
        gc.collect()
        
        for i in range(n):
            start_idx = A.indptr[i]
            end_idx = A.indptr[i+1]
            
            if start_idx < end_idx:
                neighbors = A.indices[start_idx:end_idx]
                # Remove self-loops and ensure neighbors are within valid range
                neighbors = neighbors[(neighbors != i) & (neighbors >= 0) & (neighbors < n)]
                # Convert to list of integers (METIS requirement)
                adj.append([int(neighbor) for neighbor in neighbors])
            else:
                # No neighbors for this node
                adj.append([])
            
            # Periodic garbage collection for very large matrices
            if self.AGGRESSIVE_GC and i > 0 and i % 50000 == 0:
                gc.collect()
            elif i > 0 and i % 100000 == 0:
                gc.collect()
        
        # Final garbage collection after adjacency list creation
        gc.collect()
        
        return adj

    def load_matrix(self, filename:str) -> csr_matrix:
        """
        Load a matrix from a Matrix Market file to a Compressed Sparse Row format.

        Parameters
        ----------
        filename : str
            The path to the Matrix Market file.

        Returns
        -------
        scipy.sparse.csr_matrix: 
            The loaded matrix in Compressed Sparse Row format.
        """
        # Force garbage collection before loading
        gc.collect()
        
        matrix = mmread(filename)
        matrix = matrix.tocsr()
        
        # Force garbage collection after loading and conversion
        gc.collect()
        
        return matrix

    def save_matrix(self, matrix:csr_matrix, filename:str):
        """
        Save a matrix to a Matrix Market file.

        Parameters
        ----------
        matrix : scipy.sparse.csr_matrix
            The matrix to save.
        filename : str
            The path to the output file.
        """
        # If the matrix is symmetric, save only the upper triangular part
        if (matrix != matrix.T).nnz == 0:
            matrix = triu(matrix)
        mmwrite(filename, matrix)

    def save_reorderred_matrix(self, matrix:csr_matrix):
        """
        Save a reordered matrix to a Matrix Market file.

        Parameters
        ----------
        matrix : scipy.sparse.csr_matrix
            The reordered matrix to save.
        filename : str
            The path to the output file.
        """
        path = os.path.join(self.dir, f"reordered_{self.method_name}_{self.file_name}")
        self.save_matrix(matrix, path)

    def bandwidth(self, A):
        A = A.tocoo()
        return np.max(np.abs(A.row - A.col))

    def parse_metrics(self, metrics:List[str]):
        """ DEPRECATED
        Parse the metrics to compute on the matrix.

        Parameters
        ----------
        metrics : List[str]
            A list of metric names to compute (e.g., 'nnz', 'density').
        
        Raises
        ------
        ValueError
            If an unknown metric is specified.
        """
        if metrics is None:
            self.metrics = []
            return
        
        self.metrics = []
        for metric in metrics:
            if metric == 'nnz':
                self.metrics.append(lambda x: x.nnz)
            elif metric == 'density':
                self.metrics.append(lambda x: x.nnz / (x.shape[0] * x.shape[1]))
            else:
                raise ValueError(f"Unknown metric: {metric}")

    def select_reordering_method(self, method_name: str) -> callable:
        """
        Select a reordering method based on the provided name.

        Returns
        -------
        callable
            A function that takes a matrix and returns a permutation array.
        """
        if method_name == 'amd':
            self.conditions = {'symmetric': True, 'square': True, 'positive definite': True}
            return self.amd_permutation
        elif method_name == 'symamd':
            self.conditions = {'symmetric': True, 'square': True}
            raise NotImplementedError("symamd is not implemented yet.")
        elif method_name == 'rcm':
            self.conditions = {'symmetric': True, 'square': True}
            return self.best_rcm_permutation
        elif method_name == None:
            self.conditions = {'symmetric': True, 'square': True}
            return lambda x: np.arange(x.shape[0]) # No reordering, return identity permutation
        else: 
            raise ValueError(f"Unknown reordering method: {method_name}")
    
    def select_partitioning_method(self, method_name: str) -> callable:
        """
        Select a partitioning method based on the provided name.

        Returns
        -------
        callable
            A function that takes a matrix and returns a list of arrays of indices for each partition.
        """
        if method_name == 'metis':
            self.conditions = {'symmetric': True, 'square': True}
            return self.metis_partition
        elif method_name == 'louvain':
            self.conditions = {'symmetric': True, 'square': True, 'positive': True}
            return self.louvain_partition # TODO: Add Leiden algorithm
        elif method_name == 'simple':
            self.conditions = {'symmetric': True, 'square': True}
            return self.simple_partition
        elif method_name is None:
            self.conditions = {'symmetric': True, 'square': True}
            return lambda x: [np.arange(x.shape[0])]  # No partitioning, return single partition with all indices
        else:
            raise ValueError(f"Unknown partitioning method: {method_name}")

        
    def evaluate_metrics(self, matrix:csr_matrix):
        """
        Evaluate the matrix with the specified metrics.

        Parameters
        ----------
        matrix : scipy.sparse.csr_matrix
            The matrix to evaluate.

        Returns
        -------
        dict
            A dictionary containing the computed metrics.
        """
        raise NotImplementedError("Metrics evaluation is not implemented yet.")
        
        
    import numpy as np


    def check_matrix(self, matrix: csr_matrix):
        """
        Check matrix according to the specified conditions for reordering methods. 

        Parameters
        ----------
        matrix : csr_matrix
            The matrix to check.

        Raises
        ------
        ValueError
            If the matrix does not meet the specified conditions.
        """
        if not isspmatrix_csr(matrix):
            raise ValueError("Matrix must be in CSR format.")
        if self.conditions.get('square', False) and matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix is not square.")
        if self.conditions.get('symmetric', False):
            if matrix.shape[0] != matrix.shape[1]:
                self.symmetric = False
            diff = (matrix - matrix.T).nnz
            if diff != 0:
                self.symmetric = False
        # TODO: Check datatype, datastructure


    
    def best_rcm_permutation(self, matrix, num_trials=5) -> np.ndarray:
        """
        Wrapper to repeatedly call reverse_cuthill_mckee and return the permutation
        that gives the lowest bandwidth.
        
        Parameters
        ----------
        matrix : scipy.sparse.csr_matrix
            The matrix to reorder.
        num_trials : int, optional
            Number of trials to run (default: 5).
            
        Returns
        -------
        np.ndarray
            The permutation indices that give the lowest bandwidth.
        """
        best_perm = None
        best_bandwidth = np.inf
        
        # Convert to COO once for efficiency
        coo = matrix.tocoo()
        
        for trial in range(num_trials):
            try:
                perm = reverse_cuthill_mckee(matrix)
                # Calculate bandwidth directly from permutation without creating reordered matrix
                bandwidth = self.bandwidth_from_permutation(coo, perm)
                
                if bandwidth < best_bandwidth:
                    best_bandwidth = bandwidth
                    best_perm = perm
            except Exception as e:
                # If RCM fails for any reason, continue with other trials
                continue
        
        # If all trials failed, return identity permutation
        if best_perm is None:
            return np.arange(matrix.shape[0])
            
        return best_perm
    
    def bandwidth_from_permutation(self, coo_matrix, perm) -> int:
        """
        Calculate bandwidth directly from permutation without creating reordered matrix.
        
        Parameters
        ----------
        coo_matrix : scipy.sparse.coo_matrix
            The matrix in COO format.
        perm : np.ndarray
            The permutation array.
            
        Returns
        -------
        int
            The bandwidth of the reordered matrix.
        """
        # Create inverse permutation for O(1) lookup
        inv_perm = np.empty_like(perm)
        inv_perm[perm] = np.arange(len(perm))
        
        # Calculate new row and column indices after permutation
        new_rows = inv_perm[coo_matrix.row]
        new_cols = inv_perm[coo_matrix.col]
        
        # Return maximum absolute difference
        return np.max(np.abs(new_rows - new_cols))
    
    def _is_symmetric(self, matrix, tol=1e-10):
        """
        Check if a sparse matrix is symmetric within a tolerance.
        
        Parameters
        ----------
        matrix : scipy.sparse matrix
            The matrix to check
        tol : float, optional
            Tolerance for symmetry check (default: 1e-10)
            
        Returns
        -------
        bool
            True if matrix is symmetric, False otherwise
        """
        if matrix.shape[0] != matrix.shape[1]:
            return False
        
        # Convert to CSR if needed
        if not isspmatrix_csr(matrix):
            matrix = matrix.tocsr()
        
        # Quick check: compare with transpose
        diff = matrix - matrix.T
        return np.abs(diff.data).max() < tol

    def _validate_matrix_for_metis(self, matrix):
        """
        Validate that a matrix is suitable for METIS partitioning.
        
        Parameters
        ----------
        matrix : scipy.sparse.csr_matrix
            The matrix to validate
            
        Returns
        -------
        bool
            True if matrix is valid for METIS, False otherwise
        """
        try:
            # Check basic properties
            if matrix.shape[0] != matrix.shape[1]:
                print(f"Matrix is not square: {matrix.shape}")
                return False
                
            if matrix.nnz == 0:
                print("Matrix has no non-zero elements")
                return False
                
            # Check for very large matrices that might cause memory issues
            n = matrix.shape[0]
            if n > 1000000:  # 1M nodes
                print(f"Warning: Large matrix size {n}x{n} may cause memory issues with METIS")
                
            # Check connectivity
            matrix_copy = matrix.copy()
            matrix_copy.eliminate_zeros()
            
            # Count nodes with at least one connection
            connected_nodes = np.sum(np.diff(matrix_copy.indptr) > 0)
            if connected_nodes < 2:
                print(f"Too few connected nodes ({connected_nodes}) for meaningful partitioning")
                return False
                
            print(f"Matrix validation passed: {n}x{n}, {matrix.nnz} nnz, {connected_nodes} connected nodes")
            return True
            
        except Exception as e:
            print(f"Matrix validation failed: {str(e)}")
            return False

    def simple_partition(self, matrix, nparts=None) -> List[np.ndarray]:
        """
        Simple partitioning method that doesn't require external libraries.
        Uses degree-based ordering and splits nodes into equal-sized partitions.
        
        Parameters
        ----------
        matrix : scipy.sparse.csr_matrix
            The matrix to partition.
        nparts : int, optional
            The number of partitions to create (default is self.num_clusters).

        Returns
        -------
        List[np.ndarray]
            A list of arrays, where each array contains the indices of the nodes in that partition.
        """
        if nparts is None:
            nparts = self.num_clusters
            
        n = matrix.shape[0]
        if nparts <= 1:
            return [np.arange(n)]
            
        # Calculate node degrees
        degrees = np.array((matrix != 0).sum(axis=1)).flatten()
        
        # Sort nodes by degree (descending)
        sorted_indices = np.argsort(degrees)[::-1]
        
        # Split into partitions
        partition_size = n // nparts
        parts_idx = []
        
        for i in range(nparts):
            start_idx = i * partition_size
            if i == nparts - 1:  # Last partition gets remaining nodes
                end_idx = n
            else:
                end_idx = (i + 1) * partition_size
            
            partition_nodes = sorted_indices[start_idx:end_idx]
            if len(partition_nodes) > 0:
                parts_idx.append(partition_nodes)
        
        print(f"Simple partitioning completed, created {len(parts_idx)} partitions")
        return parts_idx

def parse_arguments():
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        The parsed command line arguments.
    """
    global NUM_CLUSTERS
    # Set default number of clusters from environment variable or use 8
    NUM_CLUSTERS = int(os.getenv('NUM_CLUSTERS', 8))
    parser = argparse.ArgumentParser(description="Matrix Reordering Tool")
    parser.add_argument('--file', type=str, default='data/astro-ph/astro-ph.mtx',
                        help='Path to the Matrix Market file')
    
    # For backward compatibility, keep --method but map to new architecture
    parser.add_argument('--method', type=str, default='rcm', choices=['amd', 'rcm', 'metis', 'louvain'],
                        help='Reordering method: amd, rcm (pure reordering) or metis, louvain (partitioning+rcm)')
    
    # New architecture parameters
    parser.add_argument('--reorder-method', type=str, default=None, choices=['amd', 'rcm'],
                        help='Pure reordering method (overrides --method)')
    parser.add_argument('--partition-method', type=str, default=None, choices=[None, 'metis', 'louvain', 'simple'],
                        help='Partitioning method (overrides --method). Use "simple" for a safe fallback if METIS crashes')
    parser.add_argument('--ordering-place', type=str, default='local', choices=['local', 'global'],
                        help='Apply reordering locally (within partitions) or globally (default: local)')
    
    parser.add_argument('--nparts', type=int, default=NUM_CLUSTERS,
                        help='Number of partitions for METIS (default: 8)')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite existing reordered matrix file')
    parser.add_argument('--res_limit', type=float, default=1.0,
                        help='Resolution limit for Louvain method (default: 1.0)')
    parser.add_argument('--repeats', type=int, default=1,
                        help='Number of repeats for clustering methods (default: 1)')
    parser.add_argument('--allow-preprocessing', action='store_true', default=True,
                        help='Allow preprocessing of the matrix (e.g., making symmetric)')
    parser.add_argument('--allow-metrics', action='store_true', default=True,
                        help='Allow computation of metrics on the matrix')
    parser.add_argument('--aggressive-gc', action='store_true', default=False,
                        help='Enable more aggressive garbage collection to reduce memory usage (may slow down execution)')
    
    return parser.parse_args()

def main(args=None):
    """
        Main function to execute the matrix reordering.
    """
    
    # Force garbage collection at start
    gc.collect()
    
    # Map old method names to new architecture
    reorder_method = args.reorder_method
    partition_method = args.partition_method
    ordering_place = args.ordering_place
    
    # Handle backward compatibility with --method
    if args.method and not args.reorder_method and not args.partition_method:
        if args.method in ['amd', 'rcm']:
            reorder_method = args.method
            partition_method = None
            ordering_place = 'global'
        elif args.method in ['metis', 'louvain']:
            reorder_method = 'rcm'  # Default reordering for partitioning methods
            partition_method = args.method
            ordering_place = 'local'
    
    # Initialize reorderer with new architecture - this will create the proper method_name
    reorderer = MatrixReorderer(
        path=args.file, 
        reordering_method=reorder_method,
        partitioning_method=partition_method,
        num_clusters=args.nparts,
        res_limit=args.res_limit,
        ordering_place=ordering_place
    )
    
    # Use the method name from the reorderer object (includes all parameters)
    method_name = reorderer.method_name
    
    # Check if reordered matrix already exists
    output_filename = f"reordered_{method_name}_{os.path.basename(args.file)}"
    output_path = os.path.join(os.path.dirname(args.file), output_filename)
    
    if not args.overwrite and os.path.exists(output_path):
        print(f"Reordered matrix already exists: {output_filename}")
        print("Use --overwrite to overwrite.")
        return 0
    
    # Force garbage collection before loading matrix
    gc.collect()
    
    matrix = reorderer.load_matrix(args.file)

    # Check if square
    if matrix.shape[0] != matrix.shape[1]:
        print(f"Matrix {reorderer.file_name} is not square. Cannot reorder.")
        return 0

    print(f"Loaded matrix {reorderer.file_name} with shape {matrix.shape} and {matrix.nnz} non-zero entries.")

    reorderer.check_matrix(matrix)

    # Reorder the matrix
    print(f"Reordering matrix {reorderer.file_name} using method {reorderer.method_name}...")
    reordered_matrix, perm = reorderer.hybrid_reorder(matrix)

    print(f"Bandwidth before reordering: {reorderer.bandwidth(matrix)}")
    print(f"Bandwidth after reordering: {reorderer.bandwidth(reordered_matrix)}")
    
         
    # Save the reordered matrix
    reorderer.save_reorderred_matrix(reordered_matrix)
    print(f"Reordered matrix saved as: reordered_{method_name}_{os.path.basename(args.file)}")

    return 0



if __name__ == "__main__":
    args = parse_arguments()
    main(args)
    sys.exit(0)