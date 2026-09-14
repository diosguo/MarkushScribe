"""Stochastic SMILES -> Markush IR abstraction.

The transformation keeps a connected skeleton and abstracts the rest into
Markush variables: terminal/whole branches become R groups, ring atoms become
M variables, acyclic linkers become L variables, isolated ring systems become
A/B circle placeholders, and repeated chains can collapse into an `(R)n` group.
The output is a plain MarkushRender IR document; it carries no coordinates and
lets CoordGen lay the result out.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any

from rdkit import Chem

from . import molecule as molutil
from .config import MarkushConfig
from .ir import IREdge, IRNode, MarkushIR, atom_symbol

CutEdge = tuple[int, int, int]


def _bfs_tree(
    root: int, component: set[int], graph: dict[int, set[int]]
) -> tuple[dict[int, int | None], dict[int, int]]:
    parent: dict[int, int | None] = {root: None}
    depth: dict[int, int] = {root: 0}
    queue: deque[int] = deque([root])
    while queue:
        idx = queue.popleft()
        for neighbour in sorted(graph[idx]):
            if neighbour in component and neighbour not in parent:
                parent[neighbour] = idx
                depth[neighbour] = depth[idx] + 1
                queue.append(neighbour)
    return parent, depth


def _is_repeat_chain(
    mol: Chem.Mol,
    component: set[int],
    graph: dict[int, set[int]],
    bond_lookup: dict[frozenset[int], Chem.Bond],
) -> bool:
    if len(component) < 2:
        return False
    elements = {mol.GetAtomWithIdx(idx).GetSymbol() for idx in component}
    if len(elements) != 1 or next(iter(elements)) == "H":
        return False
    for idx in component:
        if mol.GetAtomWithIdx(idx).IsInRing():
            return False
        internal = [n for n in graph[idx] if n in component]
        if len(internal) > 2:
            return False
        for neighbour in internal:
            if molutil.bond_type(bond_lookup[frozenset((idx, neighbour))]) != "single":
                return False
    return True


def _fragment_smiles(mol: Chem.Mol, atoms: set[int]) -> str | None:
    try:
        fragment = Chem.RWMol(mol)
        for idx in sorted(set(range(mol.GetNumAtoms())) - atoms, reverse=True):
            fragment.RemoveAtom(idx)
        result = fragment.GetMol()
        Chem.SanitizeMol(result)
        return Chem.MolToSmiles(result)
    except Exception:
        return None


def abstract_molecule(
    mol: Chem.Mol,
    config: MarkushConfig,
    rng: random.Random,
    *,
    source_smiles: str = "",
    source_id: str = "",
) -> MarkushIR | None:
    """Abstract ``mol`` into a :class:`MarkushIR`, or ``None`` when unsuitable."""
    num_atoms = mol.GetNumAtoms()
    if num_atoms < config.min_core_atoms:
        return None

    graph = molutil.adjacency(mol)
    bond_lookup = {
        frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())): bond for bond in mol.GetBonds()
    }
    core = molutil.select_core(mol, config, rng)
    if len(core) < config.min_core_atoms:
        return None

    keep_atoms = set(core)
    cut_edges: list[CutEdge] = []
    cut_component: dict[tuple[int, int], tuple[set[int], bool]] = {}

    removed = set(range(num_atoms)) - core
    for component in molutil.connected_components(removed, graph):
        attachments = [
            (core_neighbour, atom, bond_lookup[frozenset((atom, core_neighbour))])
            for atom in component
            for core_neighbour in graph[atom]
            if core_neighbour in core
        ]
        if len(attachments) != 1:
            keep_atoms |= component
            continue
        keep_side, root, bond = attachments[0]
        order = molutil.bond_order(bond)
        has_ring = any(mol.GetAtomWithIdx(idx).IsInRing() for idx in component)
        whole = has_ring or rng.random() < config.whole_branch_prob
        if whole:
            cut_edges.append((keep_side, root, order))
            cut_component[(keep_side, root)] = (component, True)
            continue

        parent, depth = _bfs_tree(root, component, graph)
        keep_length = rng.choice(config.keep_length_choices)
        keep_atoms |= {idx for idx in component if depth[idx] < keep_length}
        if keep_length == 0:
            cut_edges.append((keep_side, root, order))
            cut_component[(keep_side, root)] = (component, True)
        else:
            for idx in component:
                if depth[idx] != keep_length - 1:
                    continue
                for neighbour in sorted(graph[idx]):
                    if neighbour in component and parent.get(neighbour) == idx:
                        cut_edges.append(
                            (
                                idx,
                                neighbour,
                                molutil.bond_order(bond_lookup[frozenset((idx, neighbour))]),
                            )
                        )
                        cut_component[(idx, neighbour)] = (component, False)

    if len(set(range(num_atoms)) - keep_atoms) > config.max_abstraction_ratio * num_atoms:
        return None
    if not cut_edges and not keep_atoms:
        return None

    cut_edges.sort(key=lambda edge: (edge[0], edge[1]))

    placeholder_groups: list[set[int]] = []
    placeholder_of: dict[int, int] = {}
    if config.p_ring_placeholder > 0:
        candidates = []
        for system in molutil.ring_systems(mol):
            if not system <= keep_atoms or not 3 <= len(system) <= 8:
                continue
            if not any(neighbour not in system for atom in system for neighbour in graph[atom]):
                continue
            candidates.append(system)
        rng.shuffle(candidates)
        for system in candidates:
            if len(placeholder_groups) >= config.max_placeholders:
                break
            if rng.random() >= config.p_ring_placeholder:
                continue
            placeholder_of.update({atom: len(placeholder_groups) for atom in system})
            placeholder_groups.append(system)

    m_atoms: dict[int, str] = {}
    l_atoms: dict[int, str] = {}
    counters: dict[str, int] = {}
    placeholder_systems = set().union(*placeholder_groups) if placeholder_groups else set()

    def next_symbol(family: str) -> str:
        counters[family] = counters.get(family, 0) + 1
        return f"{family}{counters[family]}"

    if config.p_ring_atom_m > 0:
        candidates = [
            idx
            for idx in sorted(keep_atoms)
            if idx not in placeholder_systems and mol.GetAtomWithIdx(idx).IsInRing()
        ]
        rng.shuffle(candidates)
        for idx in candidates:
            if len(m_atoms) >= config.max_m_vars:
                break
            if rng.random() < config.p_ring_atom_m:
                m_atoms[idx] = next_symbol(rng.choice(config.m_families))

    if config.p_linker_l > 0:
        candidates = [
            idx
            for idx in sorted(keep_atoms)
            if idx not in placeholder_systems
            and not mol.GetAtomWithIdx(idx).IsInRing()
            and len(graph[idx]) == 2
        ]
        rng.shuffle(candidates)
        for idx in candidates:
            if len(l_atoms) >= config.max_l_vars:
                break
            if rng.random() < config.p_linker_l:
                l_atoms[idx] = next_symbol(rng.choice(config.l_families))

    nodes: list[IRNode] = []

    def add_node(**fields: Any) -> int:
        node_id = len(nodes)
        nodes.append(IRNode(id=node_id, **fields))
        return node_id

    placeholder_node: dict[int, int] = {}
    for group_index, pid in enumerate(config.placeholder_ids):
        if group_index >= len(placeholder_groups):
            break
        placeholder_node[group_index] = add_node(kind="variable", symbol=pid, variable_type="ring")

    atom_node: dict[int, int] = {}
    h_delta = {atom: 0 for atom in keep_atoms}
    for keep, _cut, order in cut_edges:
        h_delta[keep] = h_delta.get(keep, 0) + max(0, order - 1)

    for idx in sorted(keep_atoms - placeholder_systems - set(m_atoms) - set(l_atoms)):
        atom = mol.GetAtomWithIdx(idx)
        h_count = max(0, atom.GetTotalNumHs() + h_delta.get(idx, 0))
        atom_node[idx] = add_node(
            kind="atom",
            symbol=atom_symbol(
                atom.GetSymbol(), h_count or None, atom.GetFormalCharge(), atom.GetIsotope() or None
            ),
            element=atom.GetSymbol(),
            h_count=h_count or None,
            charge=atom.GetFormalCharge(),
            isotope=atom.GetIsotope() or None,
            aromatic=atom.GetIsAromatic(),
        )

    var_node: dict[int, int] = {}
    for idx in sorted(set(m_atoms) | set(l_atoms)):
        if idx in m_atoms:
            var_node[idx] = add_node(
                kind="variable", symbol=m_atoms[idx], variable_type="ring_atom"
            )
        else:
            var_node[idx] = add_node(kind="variable", symbol=l_atoms[idx], variable_type="linker")

    def resolve(idx: int) -> int | None:
        if idx in placeholder_of:
            return placeholder_node[placeholder_of[idx]]
        if idx in var_node:
            return var_node[idx]
        return atom_node.get(idx)

    edges: list[IREdge] = []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in placeholder_of and placeholder_of.get(b) == placeholder_of.get(a):
            continue
        node_a, node_b = resolve(a), resolve(b)
        if node_a is None or node_b is None or node_a == node_b:
            continue
        edges.append(IREdge(source=node_a, target=node_b, bond_type=molutil.bond_type(bond)))

    rgroup_definitions: dict[str, list[str]] = {}
    for keep, cut, _order in cut_edges:
        component, whole = cut_component[(keep, cut)]
        variable_bond = rng.random() < config.p_variable_bond
        if (
            whole
            and rng.random() < config.p_repeat_chain
            and _is_repeat_chain(mol, component, graph, bond_lookup)
        ):
            family = rng.choice(config.repeat_families)
            symbol = next_symbol(family)
            multiplicity = rng.choice(config.multiplicity_choices)
            if family in config.l_families:
                rnode = add_node(
                    kind="variable",
                    symbol=symbol,
                    variable_type="linker",
                    multiplicity=multiplicity,
                )
            elif family in config.m_families:
                rnode = add_node(
                    kind="variable",
                    symbol=symbol,
                    variable_type="ring_atom",
                    multiplicity=multiplicity,
                )
            else:
                rnode = add_node(kind="rgroup", symbol=symbol, multiplicity=multiplicity)
                rgroup_definitions[symbol] = [
                    smiles for smiles in [_fragment_smiles(mol, component)] if smiles is not None
                ]
        else:
            family = rng.choice(config.r_families)
            symbol = next_symbol(family)
            multiplicity = (
                rng.choice(config.multiplicity_choices)
                if rng.random() < config.p_multiplicity
                else None
            )
            rnode = add_node(kind="rgroup", symbol=symbol, multiplicity=multiplicity)
            rgroup_definitions[symbol] = [
                smiles for smiles in [_fragment_smiles(mol, component)] if smiles is not None
            ]
        source_node = resolve(keep)
        if source_node is None:
            continue
        edges.append(
            IREdge(
                source=source_node,
                target=rnode,
                bond_type="wavy" if variable_bond else "single",
                variable=variable_bond,
            )
        )

    kind_counts: dict[str, int] = {}
    for node in nodes:
        kind_counts[node.kind] = kind_counts.get(node.kind, 0) + 1

    ir = MarkushIR(
        nodes=nodes,
        edges=edges,
        rgroup_definitions=rgroup_definitions,
        meta={
            "generator": "markushscribe.markushgen",
            "source_smiles": source_smiles,
            "source_id": source_id,
            "canonical_smiles": molutil.canonical_smiles(mol),
            "scaffold_smiles": molutil.scaffold_smiles(mol),
            "seed": config.seed,
            "core_policy": config.core_policy,
            "core_retention": config.core_retention,
            "n_atoms_input": num_atoms,
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "node_kind_counts": kind_counts,
            "abstraction_ratio": round((num_atoms - len(keep_atoms)) / num_atoms, 4),
            "rgroup_fragments": rgroup_definitions,
        },
    )
    ir.validate()
    return ir
