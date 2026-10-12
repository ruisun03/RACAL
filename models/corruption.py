from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch

from utils.graph_utils import adjacency_to_tensor


class StructuredCorruptionGenerator:
    """Generate graph-aware structured negatives for time-window contrast."""

    def __init__(
        self,
        modes: Sequence[str],
        num_negatives_per_mode: int = 1,
        num_window_negatives: int | None = None,
        max_lag: int = 3,
        lag_min: int = 1,
        lag_max: int | None = None,
        intervention_scale: float = 1.5,
        root_node_policy: str = "random_non_isolated",
        donor_policy: str = "random_batch_except_self",
        subgraph_hops_up: int = 1,
        subgraph_hops_down: int = 1,
        max_subgraph_size: int = 6,
        replace_ratio_min: float = 0.5,
        replace_ratio_max: float = 1.0,
        splice_start_ratio_min: float = 0.33,
        splice_start_ratio_max: float = 0.8,
        mode_probs: Dict[str, float] | None = None,
    ) -> None:
        self.modes = list(modes)
        resolved_lag_max = max_lag if lag_max is None else lag_max
        self.max_lag = max(1, int(max_lag))
        self.lag_min = max(1, int(lag_min))
        self.lag_max = max(self.lag_min, int(resolved_lag_max))
        self.intervention_scale = intervention_scale
        self.root_node_policy = root_node_policy
        self.donor_policy = donor_policy
        self.subgraph_hops_up = max(0, int(subgraph_hops_up))
        self.subgraph_hops_down = max(0, int(subgraph_hops_down))
        self.max_subgraph_size = max(1, int(max_subgraph_size))
        self.replace_ratio_min = float(replace_ratio_min)
        self.replace_ratio_max = float(replace_ratio_max)
        self.splice_start_ratio_min = float(splice_start_ratio_min)
        self.splice_start_ratio_max = float(splice_start_ratio_max)
        if num_window_negatives is None:
            num_window_negatives = num_negatives_per_mode
        self.num_window_negatives = max(1, int(num_window_negatives))
        self.mode_probabilities = None if mode_probs is None else {str(k): float(v) for k, v in mode_probs.items()}

        if self.root_node_policy != "random_non_isolated":
            raise ValueError(f"Unsupported root_node_policy: {self.root_node_policy}")
        if self.donor_policy != "random_batch_except_self":
            raise ValueError(f"Unsupported donor_policy: {self.donor_policy}")
        if not 0.0 < self.replace_ratio_min <= self.replace_ratio_max:
            raise ValueError("replace_ratio_min and replace_ratio_max must satisfy 0 < min <= max.")
        if self.splice_start_ratio_min < 0.0 or self.splice_start_ratio_max > 1.0:
            raise ValueError("splice_start_ratio_min/max must lie in [0, 1].")
        if self.splice_start_ratio_min > self.splice_start_ratio_max:
            raise ValueError("splice_start_ratio_min must be <= splice_start_ratio_max.")

    @staticmethod
    def _donor_indices(batch_size: int, device: torch.device) -> torch.Tensor:
        """Select donor windows by cyclic left shift within the batch."""

        if batch_size <= 1:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        return torch.roll(torch.arange(batch_size, device=device), shifts=1)

    def _resolve_indices(
        self,
        sample_indices: torch.Tensor,
        node_indices: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sample_indices = sample_indices.to(device=device, dtype=torch.long).view(-1)
        node_indices = node_indices.to(device=device, dtype=torch.long).view(-1)
        if sample_indices.shape != node_indices.shape:
            raise ValueError("sample_indices and node_indices must have the same shape.")
        return sample_indices, node_indices

    @staticmethod
    def _parents_for_node(adjacency: torch.Tensor, node_idx: int) -> torch.Tensor:
        parents = torch.where(adjacency[:, node_idx] > 0)[0]
        if parents.numel() > 0:
            return parents
        return torch.tensor([node_idx], device=adjacency.device, dtype=torch.long)

    @staticmethod
    def _incoming_neighbors(adjacency: torch.Tensor, node_idx: int) -> torch.Tensor:
        return torch.where(adjacency[:, node_idx] > 0)[0]

    @staticmethod
    def _outgoing_neighbors(adjacency: torch.Tensor, node_idx: int) -> torch.Tensor:
        return torch.where(adjacency[node_idx, :] > 0)[0]

    @staticmethod
    def _indices_to_tensor(indices: Sequence[int], device: torch.device) -> torch.Tensor:
        if not indices:
            return torch.empty((0,), device=device, dtype=torch.long)
        return torch.tensor(list(indices), device=device, dtype=torch.long)

    def _sample_index_list(
        self,
        indices: Sequence[int],
        sample_size: int,
        device: torch.device,
        deterministic: bool,
    ) -> List[int]:
        index_list = list(indices)
        if sample_size <= 0 or not index_list:
            return []
        if len(index_list) <= sample_size:
            return index_list
        if deterministic:
            return index_list[:sample_size]
        order = torch.randperm(len(index_list), device=device)[:sample_size].tolist()
        return [index_list[idx] for idx in order]

    def _candidate_root_nodes(self, adjacency: torch.Tensor) -> torch.Tensor:
        in_degree = (adjacency > 0).sum(dim=0)
        out_degree = (adjacency > 0).sum(dim=1)
        candidates = torch.where((in_degree + out_degree) > 0)[0]
        if candidates.numel() > 0:
            return candidates
        return torch.arange(adjacency.shape[0], device=adjacency.device, dtype=torch.long)

    def _sample_root_node(
        self,
        adjacency: torch.Tensor,
        sample_idx: int,
        negative_idx: int,
        deterministic: bool,
    ) -> int:
        candidates = self._candidate_root_nodes(adjacency)
        if deterministic:
            position = (sample_idx + negative_idx) % max(int(candidates.numel()), 1)
            return int(candidates[position].item())
        sampled = int(torch.randint(0, candidates.numel(), (1,), device=adjacency.device).item())
        return int(candidates[sampled].item())

    def _expand_neighbors(
        self,
        adjacency: torch.Tensor,
        seed_nodes: Sequence[int],
        hops: int,
        direction: str,
    ) -> List[int]:
        visited = set(int(node) for node in seed_nodes)
        frontier = list(visited)
        discovered: List[int] = []

        for _ in range(max(0, hops)):
            next_frontier: List[int] = []
            for node in frontier:
                if direction == "up":
                    neighbors = self._incoming_neighbors(adjacency, node)
                elif direction == "down":
                    neighbors = self._outgoing_neighbors(adjacency, node)
                else:
                    raise ValueError(f"Unsupported direction: {direction}")

                for neighbor in neighbors.tolist():
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    discovered.append(neighbor)
                    next_frontier.append(neighbor)
            frontier = next_frontier
            if not frontier:
                break

        return discovered

    def _collect_local_subgraph(
        self,
        adjacency: torch.Tensor,
        root_node_idx: int,
        deterministic: bool,
    ) -> Dict[str, torch.Tensor]:
        device = adjacency.device
        parent_candidates = self._expand_neighbors(
            adjacency=adjacency,
            seed_nodes=[root_node_idx],
            hops=self.subgraph_hops_up,
            direction="up",
        )
        child_candidates = self._expand_neighbors(
            adjacency=adjacency,
            seed_nodes=[root_node_idx],
            hops=self.subgraph_hops_down,
            direction="down",
        )

        parent_set = set(parent_candidates)
        child_candidates = [
            node_idx for node_idx in child_candidates if node_idx != root_node_idx and node_idx not in parent_set
        ]

        remaining_budget = max(0, self.max_subgraph_size - 1)
        selected_parents = self._sample_index_list(
            parent_candidates,
            sample_size=min(len(parent_candidates), remaining_budget),
            device=device,
            deterministic=deterministic,
        )
        remaining_budget = max(0, remaining_budget - len(selected_parents))
        selected_children = self._sample_index_list(
            child_candidates,
            sample_size=min(len(child_candidates), remaining_budget),
            device=device,
            deterministic=deterministic,
        )

        nodes = [root_node_idx] + selected_parents + selected_children
        return {
            "root": self._indices_to_tensor([root_node_idx], device=device),
            "parents": self._indices_to_tensor(selected_parents, device=device),
            "children": self._indices_to_tensor(selected_children, device=device),
            "nodes": self._indices_to_tensor(nodes, device=device),
        }

    def _sample_mode(self, sample_idx: int, negative_idx: int, device: torch.device, deterministic: bool) -> str:
        if deterministic:
            mode_idx = (sample_idx + negative_idx) % len(self.modes)
            return self.modes[mode_idx]

        if self.mode_probabilities:
            weights = torch.tensor(
                [self.mode_probabilities.get(mode, 0.0) for mode in self.modes],
                device=device,
                dtype=torch.float32,
            )
            weight_sum = float(weights.sum().item())
            if weight_sum > 0.0:
                mode_idx = int(torch.multinomial(weights, num_samples=1).item())
                return self.modes[mode_idx]

        mode_idx = int(torch.randint(0, len(self.modes), (1,), device=device).item())
        return self.modes[mode_idx]

    def _sample_donor_index(
        self,
        batch_size: int,
        sample_idx: int,
        device: torch.device,
        deterministic: bool,
    ) -> int:
        if batch_size <= 1:
            return sample_idx
        if deterministic:
            donors = self._donor_indices(batch_size, device)
            return int(donors[sample_idx].item())

        candidates = torch.arange(batch_size, device=device, dtype=torch.long)
        candidates = candidates[candidates != sample_idx]
        donor_position = int(torch.randint(0, candidates.numel(), (1,), device=device).item())
        return int(candidates[donor_position].item())

    def _sample_replace_count(
        self,
        candidate_count: int,
        row_idx: int,
        device: torch.device,
        deterministic: bool,
    ) -> int:
        if candidate_count <= 1:
            return 1

        min_replace = max(1, math.ceil(self.replace_ratio_min * candidate_count))
        max_replace = max(min_replace, math.ceil(self.replace_ratio_max * candidate_count))
        max_replace = min(candidate_count, max_replace)

        if deterministic or min_replace == max_replace:
            span = max_replace - min_replace + 1
            return min_replace + (row_idx % span)
        return int(torch.randint(min_replace, max_replace + 1, (1,), device=device).item())

    def _select_replacement_nodes(
        self,
        subgraph: Dict[str, torch.Tensor],
        row_idx: int,
        deterministic: bool,
    ) -> torch.Tensor:
        device = subgraph["nodes"].device
        candidate_count = int(subgraph["nodes"].numel())
        replace_count = self._sample_replace_count(
            candidate_count=candidate_count,
            row_idx=row_idx,
            device=device,
            deterministic=deterministic,
        )
        extra_budget = max(0, replace_count - 1)
        selected_parents = self._sample_index_list(
            subgraph["parents"].tolist(),
            sample_size=min(int(subgraph["parents"].numel()), extra_budget),
            device=device,
            deterministic=deterministic,
        )
        extra_budget = max(0, extra_budget - len(selected_parents))
        selected_children = self._sample_index_list(
            subgraph["children"].tolist(),
            sample_size=min(int(subgraph["children"].numel()), extra_budget),
            device=device,
            deterministic=deterministic,
        )
        selected = [int(subgraph["root"][0].item())] + selected_parents + selected_children
        return self._indices_to_tensor(selected, device=device)

    def parent_replacement_for_node(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        sample_indices: torch.Tensor,
        node_indices: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Replace only the target node's parent trajectories with donor traces."""

        del deterministic

        adjacency_t = adjacency_to_tensor(adjacency, device=x.device)
        sample_indices, node_indices = self._resolve_indices(sample_indices, node_indices, x.device)
        donors = self._donor_indices(x.shape[0], x.device)
        out = x[sample_indices].clone()

        for row_idx, (sample_idx, node_idx) in enumerate(zip(sample_indices.tolist(), node_indices.tolist())):
            donor_idx = int(donors[sample_idx].item())
            parent_nodes = self._parents_for_node(adjacency_t, node_idx)
            out[row_idx, :, parent_nodes] = x[donor_idx, :, parent_nodes]
        return out

    def lag_mismatch_for_node(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        sample_indices: torch.Tensor,
        node_indices: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Apply lag mismatch only to the target node's parent trajectories."""

        adjacency_t = adjacency_to_tensor(adjacency, device=x.device)
        sample_indices, node_indices = self._resolve_indices(sample_indices, node_indices, x.device)
        out = x[sample_indices].clone()
        window_size = x.shape[1]
        max_lag = max(1, min(self.max_lag, window_size - 1))

        for row_idx, node_idx in enumerate(node_indices.tolist()):
            target_nodes = self._parents_for_node(adjacency_t, node_idx)
            if deterministic:
                lag = 1 + (row_idx % max_lag)
            else:
                lag = int(torch.randint(1, max_lag + 1, (1,), device=x.device).item())
            out[row_idx, :, target_nodes] = torch.roll(out[row_idx, :, target_nodes], shifts=lag, dims=0)
        return out

    def cross_window_splice_for_node(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        sample_indices: torch.Tensor,
        node_indices: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Splice only the target node and its local parent subgraph from a donor."""

        adjacency_t = adjacency_to_tensor(adjacency, device=x.device)
        sample_indices, node_indices = self._resolve_indices(sample_indices, node_indices, x.device)
        donors = self._donor_indices(x.shape[0], x.device)
        out = x[sample_indices].clone()
        window_size = x.shape[1]
        splice_min = window_size // 3
        splice_max = max(splice_min + 1, window_size - 1)
        splice_span = max(1, splice_max - splice_min)

        for row_idx, (sample_idx, node_idx) in enumerate(zip(sample_indices.tolist(), node_indices.tolist())):
            donor_idx = int(donors[sample_idx].item())
            parents = self._parents_for_node(adjacency_t, node_idx)
            splice_nodes = torch.unique(torch.cat([parents, torch.tensor([node_idx], device=x.device)]))
            if deterministic:
                splice_start = splice_min + (row_idx % splice_span)
            else:
                splice_start = int(torch.randint(splice_min, splice_max, (1,), device=x.device).item())
            out[row_idx, splice_start:, splice_nodes] = x[donor_idx, splice_start:, splice_nodes]
        return out

    def local_intervention_on_node(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        sample_indices: torch.Tensor,
        node_indices: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Inject a local perturbation on the target node only."""

        del adjacency

        sample_indices, node_indices = self._resolve_indices(sample_indices, node_indices, x.device)
        out = x[sample_indices].clone()
        window_size = x.shape[1]
        segment_length = max(2, window_size // 4)
        deterministic_start = max(0, window_size - segment_length)

        for row_idx, node_idx in enumerate(node_indices.tolist()):
            if deterministic:
                start = deterministic_start
            else:
                start = int(
                    torch.randint(
                        max(1, window_size - segment_length),
                        window_size,
                        (1,),
                        device=x.device,
                    ).item()
                )
            end = min(window_size, start + segment_length)
            node_std = torch.clamp(out[row_idx, :, node_idx].std(), min=0.5)
            if deterministic:
                pattern = torch.linspace(-1.0, 1.0, steps=end - start, device=x.device, dtype=x.dtype)
                perturbation = self.intervention_scale * node_std * pattern
            else:
                perturbation = self.intervention_scale * node_std * torch.randn(end - start, device=x.device)
            out[row_idx, start:end, node_idx] = out[row_idx, start:end, node_idx] + perturbation
        return out

    def random_noise_for_window(
        self,
        x: torch.Tensor,
        sample_indices: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Create non-structured negatives by perturbing the whole window."""

        sample_indices = sample_indices.to(device=x.device, dtype=torch.long).view(-1)
        out = x[sample_indices].clone()
        scale = torch.clamp(out.std(dim=(1, 2), keepdim=True), min=0.5)
        if deterministic:
            pattern = torch.linspace(-1.0, 1.0, steps=out.shape[1], device=x.device, dtype=x.dtype)
            noise = pattern.view(1, -1, 1).expand_as(out)
        else:
            noise = torch.randn_like(out)
        return out + self.intervention_scale * scale * noise

    def random_window_replacement(
        self,
        x: torch.Tensor,
        sample_indices: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Create non-structured negatives by using unrelated donor windows."""

        sample_indices = sample_indices.to(device=x.device, dtype=torch.long).view(-1)
        donor_indices = []
        for sample_idx in sample_indices.tolist():
            donor_indices.append(
                self._sample_donor_index(
                    batch_size=x.shape[0],
                    sample_idx=int(sample_idx),
                    device=x.device,
                    deterministic=deterministic,
                )
            )
        donors = torch.tensor(donor_indices, device=x.device, dtype=torch.long)
        return x[donors].clone()

    def subgraph_parent_replacement_for_node(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        sample_indices: torch.Tensor,
        node_indices: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Replace a root-centered local subgraph with donor trajectories."""

        adjacency_t = adjacency_to_tensor(adjacency, device=x.device)
        sample_indices, node_indices = self._resolve_indices(sample_indices, node_indices, x.device)
        out = x[sample_indices].clone()
        subgraph_sizes = []

        for row_idx, (sample_idx, root_idx) in enumerate(zip(sample_indices.tolist(), node_indices.tolist())):
            donor_idx = self._sample_donor_index(
                batch_size=x.shape[0],
                sample_idx=sample_idx,
                device=x.device,
                deterministic=deterministic,
            )
            subgraph = self._collect_local_subgraph(
                adjacency=adjacency_t,
                root_node_idx=root_idx,
                deterministic=deterministic,
            )
            replacement_nodes = self._select_replacement_nodes(
                subgraph=subgraph,
                row_idx=row_idx,
                deterministic=deterministic,
            )
            out[row_idx, :, replacement_nodes] = x[donor_idx, :, replacement_nodes]
            subgraph_sizes.append(int(subgraph["nodes"].numel()))

        size_tensor = torch.tensor(subgraph_sizes, device=x.device, dtype=torch.long)
        return out, size_tensor

    def subgraph_lag_desync_for_node(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        sample_indices: torch.Tensor,
        node_indices: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Break local temporal consistency by desynchronizing a root-centered subgraph."""

        adjacency_t = adjacency_to_tensor(adjacency, device=x.device)
        sample_indices, node_indices = self._resolve_indices(sample_indices, node_indices, x.device)
        out = x[sample_indices].clone()
        subgraph_sizes = []
        window_size = x.shape[1]
        effective_lag_max = max(1, min(self.lag_max, window_size - 1))
        effective_lag_min = min(self.lag_min, effective_lag_max)

        for row_idx, root_idx in enumerate(node_indices.tolist()):
            subgraph = self._collect_local_subgraph(
                adjacency=adjacency_t,
                root_node_idx=root_idx,
                deterministic=deterministic,
            )
            parent_nodes = subgraph["parents"]
            child_nodes = subgraph["children"]
            root_nodes = subgraph["root"]

            if window_size > 1:
                if deterministic:
                    lag_span = effective_lag_max - effective_lag_min + 1
                    parent_lag = effective_lag_min + (row_idx % lag_span)
                    child_lag = effective_lag_min + ((row_idx + 1) % lag_span)
                else:
                    parent_lag = int(
                        torch.randint(effective_lag_min, effective_lag_max + 1, (1,), device=x.device).item()
                    )
                    child_lag = int(
                        torch.randint(effective_lag_min, effective_lag_max + 1, (1,), device=x.device).item()
                    )

                if parent_nodes.numel() > 0:
                    out[row_idx, :, parent_nodes] = torch.roll(
                        out[row_idx, :, parent_nodes],
                        shifts=parent_lag,
                        dims=0,
                    )
                if child_nodes.numel() > 0:
                    out[row_idx, :, child_nodes] = torch.roll(
                        out[row_idx, :, child_nodes],
                        shifts=-child_lag,
                        dims=0,
                    )
                if parent_nodes.numel() == 0 and child_nodes.numel() == 0:
                    out[row_idx, :, root_nodes] = torch.roll(
                        out[row_idx, :, root_nodes],
                        shifts=parent_lag,
                        dims=0,
                    )

            subgraph_sizes.append(int(subgraph["nodes"].numel()))

        size_tensor = torch.tensor(subgraph_sizes, device=x.device, dtype=torch.long)
        return out, size_tensor

    def subgraph_splice_for_node(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        sample_indices: torch.Tensor,
        node_indices: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Splice the local subgraph from another batch window into the current one."""

        adjacency_t = adjacency_to_tensor(adjacency, device=x.device)
        sample_indices, node_indices = self._resolve_indices(sample_indices, node_indices, x.device)
        out = x[sample_indices].clone()
        subgraph_sizes = []
        window_size = x.shape[1]

        lower = int(self.splice_start_ratio_min * window_size)
        upper = int(self.splice_start_ratio_max * window_size)
        lower = min(max(1, lower), max(window_size - 1, 1))
        upper = min(max(lower, upper), max(window_size - 1, 1))

        for row_idx, (sample_idx, root_idx) in enumerate(zip(sample_indices.tolist(), node_indices.tolist())):
            donor_idx = self._sample_donor_index(
                batch_size=x.shape[0],
                sample_idx=sample_idx,
                device=x.device,
                deterministic=deterministic,
            )
            subgraph = self._collect_local_subgraph(
                adjacency=adjacency_t,
                root_node_idx=root_idx,
                deterministic=deterministic,
            )
            if window_size > 1:
                if deterministic:
                    splice_span = upper - lower + 1
                    splice_start = lower + (row_idx % splice_span)
                else:
                    splice_start = int(torch.randint(lower, upper + 1, (1,), device=x.device).item())
                out[row_idx, splice_start:, subgraph["nodes"]] = x[donor_idx, splice_start:, subgraph["nodes"]]
            subgraph_sizes.append(int(subgraph["nodes"].numel()))

        size_tensor = torch.tensor(subgraph_sizes, device=x.device, dtype=torch.long)
        return out, size_tensor

    def _apply_node_mode(
        self,
        mode: str,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        sample_indices: torch.Tensor,
        node_indices: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        if mode == "random_noise":
            return self.random_noise_for_window(
                x,
                sample_indices=sample_indices,
                deterministic=deterministic,
            )
        if mode == "random_window":
            return self.random_window_replacement(
                x,
                sample_indices=sample_indices,
                deterministic=deterministic,
            )
        if mode == "parent_replacement":
            return self.parent_replacement_for_node(
                x,
                adjacency,
                sample_indices=sample_indices,
                node_indices=node_indices,
                deterministic=deterministic,
            )
        if mode == "lag_mismatch":
            return self.lag_mismatch_for_node(
                x,
                adjacency,
                sample_indices=sample_indices,
                node_indices=node_indices,
                deterministic=deterministic,
            )
        if mode == "cross_window_splice":
            return self.cross_window_splice_for_node(
                x,
                adjacency,
                sample_indices=sample_indices,
                node_indices=node_indices,
                deterministic=deterministic,
            )
        if mode == "local_intervention":
            return self.local_intervention_on_node(
                x,
                adjacency,
                sample_indices=sample_indices,
                node_indices=node_indices,
                deterministic=deterministic,
            )
        raise ValueError(f"Unsupported node-targeted corruption mode: {mode}")

    def _apply_subgraph_mode(
        self,
        mode: str,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        sample_indices: torch.Tensor,
        node_indices: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if mode == "subgraph_parent_replacement":
            return self.subgraph_parent_replacement_for_node(
                x,
                adjacency,
                sample_indices=sample_indices,
                node_indices=node_indices,
                deterministic=deterministic,
            )
        if mode == "subgraph_lag_desync":
            return self.subgraph_lag_desync_for_node(
                x,
                adjacency,
                sample_indices=sample_indices,
                node_indices=node_indices,
                deterministic=deterministic,
            )
        if mode == "subgraph_splice":
            return self.subgraph_splice_for_node(
                x,
                adjacency,
                sample_indices=sample_indices,
                node_indices=node_indices,
                deterministic=deterministic,
            )
        raise ValueError(f"Unsupported subgraph corruption mode: {mode}")

    def generate_window_corruptions(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, List[str], torch.Tensor, torch.Tensor]:
        """Generate graph-aware negatives for each window in the batch."""

        if x.ndim != 3:
            raise ValueError("x must have shape [B, T, N].")
        if not self.modes:
            raise ValueError("At least one corruption mode must be enabled.")
        if x.shape[0] == 0:
            empty_long = x.new_empty((0,), dtype=torch.long)
            return x.new_empty((0, x.shape[1], x.shape[2])), [], empty_long, empty_long

        batch_size = x.shape[0]
        adjacency_t = adjacency_to_tensor(adjacency, device=x.device)

        corrupted_windows = []
        mode_names: List[str] = []
        root_node_indices = []
        subgraph_sizes = []

        for sample_idx in range(batch_size):
            sample_indices = torch.tensor([sample_idx], device=x.device, dtype=torch.long)
            for negative_idx in range(self.num_window_negatives):
                root_idx = self._sample_root_node(
                    adjacency=adjacency_t,
                    sample_idx=sample_idx,
                    negative_idx=negative_idx,
                    deterministic=deterministic,
                )
                mode = self._sample_mode(
                    sample_idx=sample_idx,
                    negative_idx=negative_idx,
                    device=x.device,
                    deterministic=deterministic,
                )
                node_indices = torch.tensor([root_idx], device=x.device, dtype=torch.long)

                if mode.startswith("subgraph_"):
                    corrupted, size_tensor = self._apply_subgraph_mode(
                        mode,
                        x,
                        adjacency_t,
                        sample_indices=sample_indices,
                        node_indices=node_indices,
                        deterministic=deterministic,
                    )
                    subgraph_size = int(size_tensor[0].item())
                else:
                    corrupted = self._apply_node_mode(
                        mode,
                        x,
                        adjacency_t,
                        sample_indices=sample_indices,
                        node_indices=node_indices,
                        deterministic=deterministic,
                    )
                    subgraph_size = 1

                corrupted_windows.append(corrupted)
                mode_names.append(mode)
                root_node_indices.append(root_idx)
                subgraph_sizes.append(subgraph_size)

        root_nodes = torch.tensor(root_node_indices, device=x.device, dtype=torch.long)
        subgraph_sizes_t = torch.tensor(subgraph_sizes, device=x.device, dtype=torch.long)
        return torch.cat(corrupted_windows, dim=0), mode_names, root_nodes, subgraph_sizes_t

    def generate_node_corruptions(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        sample_indices: torch.Tensor,
        node_indices: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, List[str]]:
        """Backward-compatible node-targeted corruption interface."""

        sample_indices, node_indices = self._resolve_indices(sample_indices, node_indices, x.device)
        if sample_indices.numel() == 0:
            return x.new_empty((0, x.shape[1], x.shape[2])), []
        if not self.modes:
            raise ValueError("At least one node-targeted corruption mode must be enabled.")

        adjacency_t = adjacency_to_tensor(adjacency, device=x.device)
        corrupted_windows = []
        mode_names: List[str] = []

        for row_idx in range(sample_indices.shape[0]):
            mode = self._sample_mode(
                sample_idx=int(sample_indices[row_idx].item()),
                negative_idx=row_idx,
                device=x.device,
                deterministic=deterministic,
            )
            if mode.startswith("subgraph_"):
                corrupted, _ = self._apply_subgraph_mode(
                    mode,
                    x,
                    adjacency_t,
                    sample_indices=sample_indices[row_idx : row_idx + 1],
                    node_indices=node_indices[row_idx : row_idx + 1],
                    deterministic=deterministic,
                )
            else:
                corrupted = self._apply_node_mode(
                    mode,
                    x,
                    adjacency_t,
                    sample_indices=sample_indices[row_idx : row_idx + 1],
                    node_indices=node_indices[row_idx : row_idx + 1],
                    deterministic=deterministic,
                )
            corrupted_windows.append(corrupted)
            mode_names.append(mode)

        return torch.cat(corrupted_windows, dim=0), mode_names
