"""RDKit helpers used by the Markush abstraction pipeline."""

from __future__ import annotations

import random
from collections import deque

try:  # pragma: no cover - import guard
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError as exc:  # pragma: no cover
    raise ImportError('markushgen requires rdkit; install with `pip install -e ".[gen]"`') from exc

from .config import MarkushConfig

_BOND_TYPES = {
    Chem.BondType.SINGLE: "single",
    Chem.BondType.DOUBLE: "double",
    Chem.BondType.TRIPLE: "triple",
    Chem.BondType.AROMATIC: "aromatic",
}


def load_molecule(smiles: str) -> Chem.Mol | None:
    """Parse a SMILES string and keep the largest heavy-atom fragment."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = largest_fragment(mol)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    Chem.SanitizeMol(mol)
    return mol


def largest_fragment(mol: Chem.Mol) -> Chem.Mol | None:
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not frags:
        return None
    return max(frags, key=lambda frag: frag.GetNumHeavyAtoms())


def canonical_smiles(mol: Chem.Mol) -> str:
    try:
        return Chem.MolToSmiles(mol)
    except Exception:  # pragma: no cover - defensive
        return ""


def scaffold_smiles(mol: Chem.Mol) -> str:
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:  # pragma: no cover - defensive
        return ""


def adjacency(mol: Chem.Mol) -> dict[int, set[int]]:
    graph: dict[int, set[int]] = {atom.GetIdx(): set() for atom in mol.GetAtoms()}
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        graph[a].add(b)
        graph[b].add(a)
    return graph


def bond_type(bond: Chem.Bond) -> str:
    if bond.GetIsAromatic():
        return "aromatic"
    return _BOND_TYPES.get(bond.GetBondType(), "single")


def bond_order(bond: Chem.Bond) -> int:
    return {"single": 1, "double": 2, "triple": 3, "aromatic": 1}.get(bond_type(bond), 1)


def murcko_atoms(mol: Chem.Mol) -> set[int]:
    """Atoms of the Bemis-Murcko scaffold: rings plus the linkers between them."""
    ring_atoms = {atom.GetIdx() for atom in mol.GetAtoms() if atom.IsInRing()}
    graph = adjacency(mol)
    degree = {idx: len(neighbours) for idx, neighbours in graph.items()}
    removed: set[int] = set()
    queue: deque[int] = deque(idx for idx in graph if idx not in ring_atoms and degree[idx] <= 1)
    while queue:
        idx = queue.popleft()
        if idx in removed:
            continue
        removed.add(idx)
        for neighbour in graph[idx]:
            degree[neighbour] -= 1
            if neighbour not in removed and neighbour not in ring_atoms and degree[neighbour] <= 1:
                queue.append(neighbour)
    return set(graph) - removed


def ring_systems(mol: Chem.Mol) -> list[set[int]]:
    """Fused ring systems (rings sharing at least two atoms)."""
    ring_info = mol.GetRingInfo()
    rings = [set(ring) for ring in ring_info.AtomRings()]
    parent = list(range(len(rings)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if len(rings[i] & rings[j]) >= 2:
                union(i, j)

    groups: dict[int, set[int]] = {}
    for i, ring in enumerate(rings):
        groups.setdefault(find(i), set()).update(ring)
    return list(groups.values())


def connected_components(atoms: set[int], graph: dict[int, set[int]]) -> list[set[int]]:
    components: list[set[int]] = []
    seen: set[int] = set()
    for start in atoms:
        if start in seen:
            continue
        component: set[int] = set()
        queue = deque([start])
        seen.add(start)
        while queue:
            idx = queue.popleft()
            component.add(idx)
            for neighbour in graph[idx]:
                if neighbour in atoms and neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
        components.append(component)
    return components


def select_core(mol: Chem.Mol, config: MarkushConfig, rng: random.Random) -> set[int]:
    """Choose the retained skeleton atoms."""
    num_atoms = mol.GetNumAtoms()
    if config.core_policy == "murcko":
        return murcko_atoms(mol)

    target = max(config.min_core_atoms, round(config.core_retention * num_atoms))
    target = min(target, num_atoms)
    ring_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.IsInRing()]
    start = rng.choice(ring_atoms) if ring_atoms else rng.randrange(num_atoms)
    graph = adjacency(mol)
    order: list[int] = []
    seen = {start}
    queue: deque[int] = deque([start])
    while queue:
        idx = queue.popleft()
        order.append(idx)
        neighbours = sorted(graph[idx])
        rng.shuffle(neighbours)
        for neighbour in neighbours:
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append(neighbour)
    return set(order[:target])
