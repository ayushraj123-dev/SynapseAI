"""Knowledge Graph Construction and Visualization Module for SynapseAI.

Task 4 Implementation:
- In-memory directed graph construction (NetworkX DiGraph) preserving rich node/edge metadata.
- Strict endpoint validation: relationships referencing non-existent nodes are rejected and reported.
- Stable entity IDs and relationship IDs preserved from Task 3 extraction.
- Thematic integration: nodes track theme memberships without synthesizing artificial edges.
- Comprehensive graph statistics: node/edge counts, weakly connected components, distribution by type/theme.
- Interactive visualization using PyVis with dark-mode styling, physics layout, and HTML escaping.
- Decoupled storage abstraction: independent graph representation exportable to Dict/JSON.
"""

from collections import Counter
import html
import json
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
from pyvis.network import Network

from src.entity_extraction import (
    Entity,
    EntityMention,
    ExtractionResult,
    Relationship,
    RelationshipEvidence,
    TopicTheme,
)


# Color palette for entity types (matching SynapseAI dark slate aesthetic)
ENTITY_TYPE_COLORS: Dict[str, str] = {
    "CONCEPT": "#38bdf8",       # Sky Blue
    "TECHNOLOGY": "#6366f1",    # Indigo
    "ORGANIZATION": "#10b981",  # Emerald Green
    "PERSON": "#f59e0b",        # Amber
    "LOCATION": "#ec4899",      # Rose Pink
    "METHOD": "#8b5cf6",        # Violet / Purple
    "THEME": "#a855f7",         # Amethyst / Purple
}
DEFAULT_NODE_COLOR: str = "#94a3b8"  # Slate Gray


class KnowledgeGraphBuilder:
    """Constructs, validates, and visualizes directed knowledge graphs from extracted records."""

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self.themes: List[TopicTheme] = []
        self.missing_endpoints: List[Dict[str, Any]] = []

    def clear(self) -> None:
        """Clear the current graph and validation state."""
        self.graph.clear()
        self.themes = []
        self.missing_endpoints = []

    def build_from_extraction(
        self,
        extraction_result: Union[ExtractionResult, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Construct the knowledge graph from Task 3 extraction output.

        Args:
            extraction_result: ExtractionResult object or its dictionary representation.

        Returns:
            Dictionary of graph construction statistics and validation metrics.
        """
        self.clear()

        # Handle ExtractionResult object or serialized dictionary
        if isinstance(extraction_result, ExtractionResult):
            entities_dict = extraction_result.entities
            relationships_dict = extraction_result.relationships
            themes_list = extraction_result.themes
        elif isinstance(extraction_result, dict):
            entities_dict = extraction_result.get("entities", {})
            relationships_dict = extraction_result.get("relationships", {})
            raw_themes = extraction_result.get("themes", [])
            themes_list = []
            for t in raw_themes:
                if isinstance(t, TopicTheme):
                    themes_list.append(t)
                elif isinstance(t, dict):
                    themes_list.append(TopicTheme(
                        theme_id=t.get("theme_id", ""),
                        name=t.get("name", ""),
                        description=t.get("description", ""),
                        entity_ids=t.get("entity_ids", []),
                        source_chunk_ids=t.get("source_chunk_ids", []),
                        confidence=float(t.get("confidence", 1.0)),
                        document_filenames=t.get("document_filenames", []),
                    ))
        else:
            raise ValueError(f"Unsupported extraction result type: {type(extraction_result)}")

        self.themes = themes_list

        # Step 1: Build entity mapping from theme membership (theme membership is not an edge)
        entity_theme_names: Dict[str, List[str]] = {}
        for theme in self.themes:
            for eid in theme.entity_ids:
                entity_theme_names.setdefault(eid, []).append(theme.name)

        # Step 2: Add Entity Nodes
        for eid, ent in entities_dict.items():
            if isinstance(ent, Entity):
                name = ent.name
                etype = ent.entity_type
                aliases = ent.aliases
                description = ent.description
                mentions = ent.mention_count
                source_refs = [r.to_dict() if hasattr(r, "to_dict") else r for r in ent.source_references]
            elif isinstance(ent, dict):
                eid = ent.get("entity_id", eid)
                name = ent.get("name", eid)
                etype = ent.get("entity_type", "CONCEPT")
                aliases = ent.get("aliases", [])
                description = ent.get("description", "")
                mentions = ent.get("mention_count", 1)
                source_refs = ent.get("source_references", [])
            else:
                continue

            node_themes = entity_theme_names.get(eid, [])

            self.graph.add_node(
                eid,
                entity_id=eid,
                name=name,
                entity_type=etype,
                aliases=aliases,
                description=description,
                mention_count=mentions,
                source_references=source_refs,
                themes=node_themes,
            )

        # Build canonical resolution map from node IDs, names, and aliases
        name_to_eid: Dict[str, str] = {}
        for nid, data in self.graph.nodes(data=True):
            name_to_eid[nid.lower()] = nid
            name = data.get("name", "")
            if name:
                name_to_eid[name.lower()] = nid
            for alias in data.get("aliases", []):
                if alias:
                    name_to_eid[alias.lower()] = nid

        # Step 3: Add Directed Relationship Edges with Strict Endpoint Validation
        for rid, rel in relationships_dict.items():
            if isinstance(rel, Relationship):
                subj_id = rel.subject_id
                obj_id = rel.object_id
                pred = rel.predicate
                conf = rel.confidence
                evidence = [e.to_dict() if hasattr(e, "to_dict") else e for e in rel.evidence]
                rel_id = rel.relationship_id
            elif isinstance(rel, dict):
                subj_id = rel.get("subject_id", "")
                obj_id = rel.get("object_id", "")
                pred = rel.get("predicate", "RELATES_TO")
                conf = float(rel.get("confidence", 1.0))
                evidence = rel.get("evidence", [])
                rel_id = rel.get("relationship_id", rid)
            else:
                continue

            # Check if endpoints need canonical resolution via name or alias
            if not self.graph.has_node(subj_id) and subj_id.lower() in name_to_eid:
                subj_id = name_to_eid[subj_id.lower()]
            if not self.graph.has_node(obj_id) and obj_id.lower() in name_to_eid:
                obj_id = name_to_eid[obj_id.lower()]

            # Validate that both subject and object exist as nodes
            has_subj = self.graph.has_node(subj_id)
            has_obj = self.graph.has_node(obj_id)

            if not (has_subj and has_obj):
                self.missing_endpoints.append({
                    "relationship_id": rel_id,
                    "subject_id": subj_id,
                    "object_id": obj_id,
                    "predicate": pred,
                    "subject_exists": has_subj,
                    "object_exists": has_obj,
                    "reason": f"Missing endpoint(s): subject_exists={has_subj}, object_exists={has_obj}",
                })
                continue

            # Reject self-loops
            if subj_id == obj_id:
                continue

            # Multi-predicate safe representation: preserve distinct relationships between the same pair
            if self.graph.has_edge(subj_id, obj_id):
                existing_data = self.graph[subj_id][obj_id]
                existing_pred = existing_data.get("predicate", "")
                all_preds = list(existing_data.get("all_predicates", [existing_pred]))
                if pred not in all_preds:
                    all_preds.append(pred)
                combined_pred = " / ".join(all_preds)
                existing_ev = list(existing_data.get("evidence", []))
                for ev in evidence:
                    if ev not in existing_ev:
                        existing_ev.append(ev)
                self.graph[subj_id][obj_id]["predicate"] = combined_pred
                self.graph[subj_id][obj_id]["all_predicates"] = all_preds
                self.graph[subj_id][obj_id]["evidence"] = existing_ev
                self.graph[subj_id][obj_id]["confidence"] = max(existing_data.get("confidence", 0.0), conf)
            else:
                self.graph.add_edge(
                    subj_id,
                    obj_id,
                    relationship_id=rel_id,
                    predicate=pred,
                    all_predicates=[pred],
                    confidence=conf,
                    evidence=evidence,
                    edge_type="FACTUAL",
                )

        return self.get_graph_statistics()

    def build_graph(
        self,
        entities: Union[List[Dict[str, Any]], Dict[str, Any]],
        relations: Union[List[Dict[str, Any]], Dict[str, Any]],
        themes: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Backward-compatible graph construction method for raw list/dict inputs.

        Args:
            entities: Entity list or dictionary.
            relations: Relationship list or dictionary.
            themes: Optional list of topic themes.

        Returns:
            Graph summary dictionary.
        """
        # Convert list of entities to dict if needed
        if isinstance(entities, list):
            ent_dict = {
                e.get("entity_id", f"ent_{i}"): e for i, e in enumerate(entities) if isinstance(e, dict)
            }
        else:
            ent_dict = entities

        # Convert list of relations to dict if needed
        if isinstance(relations, list):
            rel_dict = {
                r.get("relationship_id", f"rel_{i}"): r for i, r in enumerate(relations) if isinstance(r, dict)
            }
        else:
            rel_dict = relations

        fake_res = {
            "entities": ent_dict,
            "relationships": rel_dict,
            "themes": themes or [],
        }
        return self.build_from_extraction(fake_res)

    def get_graph_statistics(self) -> Dict[str, Any]:
        """Compute structural, topological, and validation metrics for the current graph.

        Returns:
            Dictionary containing node count, edge count, connected components,
            distribution by type/theme, density, and missing endpoint counts.
        """
        num_nodes = self.graph.number_of_nodes()
        num_edges = self.graph.number_of_edges()

        if num_nodes == 0:
            return {
                "node_count": 0,
                "edge_count": 0,
                "connected_components": 0,
                "density": 0.0,
                "nodes_by_type": {},
                "nodes_by_theme": {},
                "missing_endpoints_count": len(self.missing_endpoints),
                "missing_endpoints": list(self.missing_endpoints),
            }

        # Calculate weakly connected components for directed graph
        num_components = nx.number_weakly_connected_components(self.graph)
        density = round(float(nx.density(self.graph)), 4)

        # Count nodes by type
        type_counts: Counter = Counter()
        for _, attrs in self.graph.nodes(data=True):
            type_counts[attrs.get("entity_type", "UNKNOWN")] += 1

        # Count nodes by theme
        theme_counts: Counter = Counter()
        for _, attrs in self.graph.nodes(data=True):
            for t in attrs.get("themes", []):
                theme_counts[t] += 1

        # Count factual edges vs structural edges
        factual_edges = sum(1 for _, _, d in self.graph.edges(data=True) if d.get("edge_type", "FACTUAL") == "FACTUAL")
        structural_edges = num_edges - factual_edges

        # Unique predicates count
        unique_predicates = set()
        for _, _, d in self.graph.edges(data=True):
            for p in d.get("all_predicates", [d.get("predicate", "")]):
                if p:
                    unique_predicates.add(p)

        isolated = list(nx.isolates(self.graph))

        return {
            "node_count": num_nodes,
            "edge_count": num_edges,
            "factual_edge_count": factual_edges,
            "structural_edge_count": structural_edges,
            "unique_predicates_count": len(unique_predicates),
            "unique_predicates": sorted(list(unique_predicates)),
            "isolated_nodes_count": len(isolated),
            "isolated_nodes": isolated,
            "connected_components": num_components,
            "density": density,
            "nodes_by_type": dict(type_counts),
            "nodes_by_theme": dict(theme_counts),
            "missing_endpoints_count": len(self.missing_endpoints),
            "missing_endpoints": list(self.missing_endpoints),
        }

    def get_visualization_data(self) -> Dict[str, Any]:
        """Format the graph nodes and edges for frontend visualization.

        Returns:
            Dictionary containing serializable 'nodes' and 'edges' arrays.
        """
        nodes = []
        for node_id, data in self.graph.nodes(data=True):
            nodes.append({
                "id": node_id,
                "label": data.get("name", node_id),
                "entity_type": data.get("entity_type", "CONCEPT"),
                "aliases": data.get("aliases", []),
                "description": data.get("description", ""),
                "mention_count": data.get("mention_count", 1),
                "themes": data.get("themes", []),
                "source_references": data.get("source_references", []),
            })

        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append({
                "from": u,
                "to": v,
                "relationship_id": data.get("relationship_id", ""),
                "label": data.get("predicate", "RELATES_TO"),
                "confidence": data.get("confidence", 1.0),
                "evidence": data.get("evidence", []),
            })

        return {"nodes": nodes, "edges": edges}

    def get_subgraph(
        self,
        filter_types: Optional[List[str]] = None,
        filter_theme: Optional[str] = None,
        focus_node: Optional[str] = None,
        include_structural: bool = False,
    ) -> nx.DiGraph:
        """Create a filtered view or ego-subgraph based on user selections.

        Args:
            filter_types: If provided, only include nodes with matching entity types.
            filter_theme: If provided, only include nodes belonging to this theme.
            focus_node: If provided, only include this node and its immediate neighbors (1-hop).
            include_structural: If True, include TopicTheme nodes and BELONGS_TO structural edges.

        Returns:
            Filtered NetworkX DiGraph.
        """
        if self.graph.number_of_nodes() == 0:
            return nx.DiGraph()

        candidate_nodes: Set[str] = set(self.graph.nodes())

        # Filter by entity type
        if filter_types and "ALL" not in filter_types:
            type_set = set(filter_types)
            candidate_nodes = {
                n for n in candidate_nodes
                if self.graph.nodes[n].get("entity_type") in type_set
            }

        # Filter by theme
        if filter_theme and filter_theme != "ALL":
            candidate_nodes = {
                n for n in candidate_nodes
                if filter_theme in self.graph.nodes[n].get("themes", [])
            }

        # Focus on ego-network (selected node + in/out neighbors)
        if focus_node and self.graph.has_node(focus_node):
            in_neighbors = set(self.graph.predecessors(focus_node))
            out_neighbors = set(self.graph.successors(focus_node))
            ego_neighborhood = {focus_node} | in_neighbors | out_neighbors
            candidate_nodes = candidate_nodes & ego_neighborhood
            # Ensure the focus node itself is retained even if type/theme filtered
            candidate_nodes.add(focus_node)

        sub = self.graph.subgraph(candidate_nodes).copy()

        # Optionally inject TopicTheme nodes and structural BELONGS_TO edges
        if include_structural and self.themes:
            for theme in self.themes:
                if filter_theme and filter_theme != "ALL" and theme.name != filter_theme:
                    continue
                member_eids = [eid for eid in theme.entity_ids if eid in candidate_nodes]
                if member_eids:
                    theme_node_id = f"theme_node_{theme.theme_id}"
                    sub.add_node(
                        theme_node_id,
                        entity_id=theme_node_id,
                        name=theme.name,
                        entity_type="THEME",
                        aliases=[theme.name],
                        description=theme.description,
                        mention_count=len(member_eids),
                        source_references=[],
                        themes=[theme.name],
                        is_structural=True,
                    )
                    for eid in member_eids:
                        sub.add_edge(
                            eid,
                            theme_node_id,
                            relationship_id=f"struct_{eid}_{theme.theme_id}",
                            predicate="BELONGS_TO",
                            all_predicates=["BELONGS_TO"],
                            confidence=theme.confidence,
                            evidence=[{
                                "evidence_text": f"Entity '{self.graph.nodes[eid].get('name', eid)}' belongs to topic theme: {theme.name}",
                                "source_filename": "Thematic Clustering",
                            }],
                            edge_type="STRUCTURAL",
                            is_structural=True,
                        )

        return sub

    def generate_pyvis_html(
        self,
        filter_types: Optional[List[str]] = None,
        filter_theme: Optional[str] = None,
        focus_node: Optional[str] = None,
        highlight_nodes: Optional[List[str]] = None,
        highlight_edges: Optional[List[Tuple[str, str]]] = None,
        height: str = "520px",
        theme: str = "dark",
        include_structural: bool = False,
    ) -> str:
        """Generate an interactive HTML visualization of the knowledge graph using PyVis.

        Supports dynamic query-driven graph highlighting: dims unrelated nodes and edges,
        styles evidence paths with glowing accents, and centers camera focus on evidence.
        Adapts backgrounds, edges, and font colors to light or dark theme mode.

        Args:
            filter_types: Optional list of entity types to display.
            filter_theme: Optional topic theme name to filter nodes.
            focus_node: Optional entity ID to focus on with immediate neighbors.
            highlight_nodes: Optional list of node IDs or entity names to highlight.
            highlight_edges: Optional list of (subject, object) edge tuples to highlight.
            height: Height of the embedded canvas (CSS string).
            theme: Active theme mode ('dark' or 'light').
            include_structural: If True, include TopicTheme hubs and BELONGS_TO structural links.

        Returns:
            Self-contained HTML string with interactive VisJS canvas.
        """
        sub = self.get_subgraph(
            filter_types=filter_types,
            filter_theme=filter_theme,
            focus_node=focus_node,
            include_structural=include_structural,
        )

        is_light = (theme == "light")
        empty_bg = "#f4f6fc" if is_light else "#0d1117"
        empty_border = "#dce3f2" if is_light else "#30363d"
        empty_color = "#64748b" if is_light else "#8b949e"

        if sub.number_of_nodes() == 0:
            return (
                f'<div style="height: 380px; display: flex; align-items: center; justify-content: center; '
                f'background-color: {empty_bg}; color: {empty_color}; border: 1px dashed {empty_border}; border-radius: 8px;">'
                '<span>No nodes match the selected filters.</span></div>'
            )

        # Build lookup sets for highlighted nodes and edges
        hl_node_set: Set[str] = set()
        matched_hl_ids: List[str] = []
        if highlight_nodes:
            lower_targets = {h.lower() for h in highlight_nodes if h}
            for nid, ndata in sub.nodes(data=True):
                nid_lower = nid.lower()
                name_lower = ndata.get("name", "").lower()
                if nid in highlight_nodes or nid_lower in lower_targets or name_lower in lower_targets:
                    hl_node_set.add(nid)
                    if nid not in matched_hl_ids:
                        matched_hl_ids.append(nid)

        hl_edge_set: Set[Tuple[str, str]] = set()
        if highlight_edges:
            for u, v in highlight_edges:
                hl_edge_set.add((u, v))
                hl_edge_set.add((v, u))
                # Map names to IDs if applicable
                u_ids = [nid for nid, d in sub.nodes(data=True) if d.get("name", "").lower() == u.lower() or nid.lower() == u.lower()]
                v_ids = [nid for nid, d in sub.nodes(data=True) if d.get("name", "").lower() == v.lower() or nid.lower() == v.lower()]
                for uid in u_ids:
                    for vid in v_ids:
                        hl_edge_set.add((uid, vid))
                        hl_edge_set.add((vid, uid))

        has_active_highlight = bool(hl_node_set)

        # Theme-aware colors
        bg_col = "#f4f6fc" if is_light else "#0d1117"
        font_col = "#172033" if is_light else "#f8fafc"
        edge_default_col = "#6366f1" if is_light else "#818cf8"
        edge_stroke_col = "#ffffff" if is_light else "#0d1117"
        edge_font_col = "#1e293b" if is_light else "#f8fafc"

        # Initialize PyVis network
        net = Network(
            height=height,
            width="100%",
            directed=True,
            bgcolor=bg_col,
            font_color=font_col,
        )

        # Configure physics for stable, responsive layout
        net.set_options(
            json.dumps({
                "nodes": {
                    "font": {"size": 13, "color": font_col, "face": "Segoe UI, sans-serif"},
                    "borderWidth": 1.5,
                    "borderWidthSelected": 3.0,
                    "shadow": True,
                },
                "edges": {
                    "color": {"color": edge_default_col, "highlight": "#8b5cf6", "hover": "#a855f7"},
                    "arrows": {"to": {"enabled": True, "scaleFactor": 0.8}},
                    "font": {"size": 11, "color": edge_font_col, "align": "middle", "strokeWidth": 3, "strokeColor": edge_stroke_col},
                    "smooth": {"type": "continuous", "roundness": 0.2},
                },
                "physics": {
                    "solver": "forceAtlas2Based",
                    "forceAtlas2Based": {
                        "gravitationalConstant": -45,
                        "centralGravity": 0.015,
                        "springLength": 120,
                        "springConstant": 0.08,
                        "damping": 0.4,
                    },
                    "minVelocity": 0.75,
                    "stabilization": {"enabled": True, "iterations": 150},
                },
                "interaction": {
                    "hover": True,
                    "navigationButtons": True,
                    "keyboard": True,
                    "tooltipDelay": 150,
                },
            })
        )

        # Add Nodes to PyVis with custom styling and tooltips
        for nid, data in sub.nodes(data=True):
            name = data.get("name", nid)
            etype = data.get("entity_type", "CONCEPT")
            mentions = data.get("mention_count", 1)
            desc = data.get("description", "")
            themes = data.get("themes", [])
            source_refs = data.get("source_references", [])
            is_struct_node = bool(data.get("is_structural"))

            is_highlighted = nid in hl_node_set

            if has_active_highlight:
                if is_highlighted:
                    node_size = min(22 + (mentions * 3), 42)
                    color = {
                        "background": "#38bdf8",
                        "border": "#ffffff",
                        "highlight": {"background": "#0284c7", "border": "#ffffff"},
                    }
                    font = {"size": 14, "color": "#ffffff", "face": "Segoe UI, sans-serif", "bold": True}
                    border_width = 3.5
                else:
                    node_size = max(10, min(14 + mentions, 20))
                    color = {
                        "background": "rgba(45, 55, 72, 0.30)",
                        "border": "rgba(71, 85, 105, 0.20)",
                        "highlight": {"background": "rgba(45, 55, 72, 0.60)", "border": "#94a3b8"},
                    }
                    font = {"size": 10, "color": "rgba(148, 163, 184, 0.40)", "face": "Segoe UI, sans-serif"}
                    border_width = 1.0
            else:
                base_color = ENTITY_TYPE_COLORS.get(etype, DEFAULT_NODE_COLOR)
                node_size = min(16 + (mentions * 3), 36)
                if nid == focus_node:
                    node_size += 6
                    base_color = "#38bdf8"
                elif is_struct_node:
                    base_color = "#a855f7"
                    node_size = 28
                color = base_color
                font = {"size": 13, "color": "#1e293b" if is_light else "#f8fafc", "face": "Segoe UI, sans-serif"}
                border_width = 1.5

            clean_name = html.escape(name)
            clean_etype = html.escape(etype)
            clean_desc = html.escape(desc[:180] + "..." if len(desc) > 180 else desc)
            clean_themes = html.escape(", ".join(themes) if themes else "None")
            ref_summary = f"{len(source_refs)} source mention(s)"
            hl_badge = " <b style='color:#38bdf8'>[RELEVANT EVIDENCE]</b>" if is_highlighted else ""
            struct_badge = " <b style='color:#a855f7'>[TOPIC THEME HUB]</b>" if is_struct_node else ""

            tooltip_html = (
                f"<b>{clean_name}</b> [{clean_etype}]{hl_badge}{struct_badge}<br/>"
                f"<i>{clean_desc}</i><br/><br/>"
                f"• Mentions: {mentions}<br/>"
                f"• Themes: {clean_themes}<br/>"
                f"• Provenance: {ref_summary}"
            )

            net.add_node(
                nid,
                label=name,
                title=tooltip_html,
                color=color,
                size=node_size,
                shape="diamond" if is_struct_node else "dot",
                font=font,
                borderWidth=border_width,
            )

        # Add Edges to PyVis with highlight awareness and high contrast
        for u, v, data in sub.edges(data=True):
            pred = data.get("predicate", "RELATES_TO")
            conf = data.get("confidence", 1.0)
            evidence = data.get("evidence", [])
            is_struct_edge = bool(data.get("is_structural"))

            is_edge_highlighted = (u, v) in hl_edge_set or (v, u) in hl_edge_set
            if not is_edge_highlighted and has_active_highlight:
                # If both endpoints are highlighted in small graph, also highlight edge
                if u in hl_node_set and v in hl_node_set and len(hl_node_set) <= 8:
                    is_edge_highlighted = True

            if has_active_highlight:
                if is_edge_highlighted:
                    edge_color = {"color": "#38bdf8", "highlight": "#0284c7", "hover": "#bae6fd"}
                    edge_width = 3.5
                    edge_font = {"size": 11, "color": "#38bdf8", "align": "middle", "strokeWidth": 3, "strokeColor": "#0d1117"}
                    arrows = {"to": {"enabled": True, "scaleFactor": 1.0}}
                else:
                    edge_color = {"color": "rgba(71, 85, 105, 0.15)", "highlight": "rgba(71, 85, 105, 0.25)", "hover": "rgba(71, 85, 105, 0.35)"}
                    edge_width = 0.8
                    edge_font = {"size": 8, "color": "rgba(100, 116, 139, 0.35)", "align": "middle", "strokeWidth": 1, "strokeColor": "#0d1117"}
                    arrows = {"to": {"enabled": True, "scaleFactor": 0.4}}
            else:
                if is_struct_edge:
                    edge_base = "#a855f7" if is_light else "#c084fc"
                    edge_hl = "#d946ef"
                    edge_hover = "#e879f9"
                else:
                    edge_base = "#6366f1" if is_light else "#818cf8"
                    edge_hl = "#0284c7" if is_light else "#38bdf8"
                    edge_hover = "#4f46e5" if is_light else "#a5b4fc"

                edge_color = {"color": edge_base, "highlight": edge_hl, "hover": edge_hover}
                edge_width = 1.8 if is_struct_edge else 2.2
                font_c = "#1e293b" if is_light else "#f8fafc"
                stroke_c = "#ffffff" if is_light else "#0d1117"
                edge_font = {"size": 11, "color": font_c, "align": "middle", "strokeWidth": 3, "strokeColor": stroke_c}
                arrows = {"to": {"enabled": True, "scaleFactor": 0.8}}

            ev_text = ""
            if evidence:
                first_ev = evidence[0]
                ev_str = first_ev.get("evidence_text", "") if isinstance(first_ev, dict) else ""
                if ev_str:
                    ev_text = f"<br/><i>Evidence:</i> &quot;{html.escape(ev_str[:160])}...&quot;"

            kind_badge = " [STRUCTURAL]" if is_struct_edge else ""
            edge_tooltip = (
                f"<b>{html.escape(pred)}</b>{kind_badge}<br/>"
                f"Confidence: {conf:.2f}"
                f"{ev_text}"
            )

            net.add_edge(
                u,
                v,
                label=pred,
                title=edge_tooltip,
                color=edge_color,
                width=edge_width,
                font=edge_font,
                arrows=arrows,
            )

        html_content = net.generate_html()

        # If evidence nodes are highlighted, inject auto-zoom VisJS script
        if matched_hl_ids:
            nodes_json = json.dumps(matched_hl_ids)
            auto_fit_script = f"""
            <script type="text/javascript">
            window.addEventListener('load', function() {{
                setTimeout(function() {{
                    if (typeof network !== 'undefined') {{
                        var targetNodes = {nodes_json};
                        if (targetNodes && targetNodes.length > 0) {{
                            network.fit({{
                                nodes: targetNodes,
                                animation: {{ duration: 1000, easingFunction: 'easeInOutQuad' }}
                            }});
                        }}
                    }}
                }}, 750);
            }});
            </script>
            """
            html_content = html_content.replace("</body>", f"{auto_fit_script}\n</body>")

        return html_content

    def export_to_dict(self) -> Dict[str, Any]:
        """Export the graph and metadata to a decoupled dictionary format.

        This format is independent of the in-memory NetworkX implementation
        and can be stored locally or exported to a graph database (e.g. Neo4j).
        """
        vis_data = self.get_visualization_data()
        return {
            "graph_backend": "networkx",
            "nodes": vis_data["nodes"],
            "edges": vis_data["edges"],
            "themes": [t.to_dict() if hasattr(t, "to_dict") else t for t in self.themes],
            "statistics": self.get_graph_statistics(),
        }

    def export_to_json(self, indent: int = 2) -> str:
        """Export the graph structure to formatted JSON."""
        return json.dumps(self.export_to_dict(), indent=indent)

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Restore graph state from a previously exported dictionary."""
        self.clear()
        nodes_list = data.get("nodes", [])
        edges_list = data.get("edges", [])
        themes_list = data.get("themes", [])

        # Add nodes
        for n in nodes_list:
            nid = n.get("id")
            if not nid:
                continue
            self.graph.add_node(
                nid,
                entity_id=nid,
                name=n.get("label", nid),
                entity_type=n.get("entity_type", "CONCEPT"),
                aliases=n.get("aliases", []),
                description=n.get("description", ""),
                mention_count=n.get("mention_count", 1),
                source_references=n.get("source_references", []),
                themes=n.get("themes", []),
            )

        # Add edges
        for e in edges_list:
            u = e.get("from")
            v = e.get("to")
            if self.graph.has_node(u) and self.graph.has_node(v):
                self.graph.add_edge(
                    u,
                    v,
                    relationship_id=e.get("relationship_id", ""),
                    predicate=e.get("label", "RELATES_TO"),
                    confidence=float(e.get("confidence", 1.0)),
                    evidence=e.get("evidence", []),
                )
            else:
                self.missing_endpoints.append({
                    "from": u,
                    "to": v,
                    "reason": "Missing endpoint node during deserialization",
                })

    def generate_3d_graph_html(
        self,
        filter_types: Optional[List[str]] = None,
        filter_theme: Optional[str] = None,
        focus_node: Optional[str] = None,
        highlight_nodes: Optional[List[str]] = None,
        highlight_edges: Optional[List[Tuple[str, str]]] = None,
        height: str = "560px",
        theme: str = "dark",
        include_structural: bool = False,
    ) -> str:
        """Generate an immersive, interactive 3D WebGL knowledge graph visualization.

        Uses Three.js and 3D Force-Directed Graph with spatial camera animation, volumetric
        glowing nodes, directional particle streams on active evidence paths, and glassmorphic HUD.
        Includes a native Canvas 3D fallback for completely offline environments.
        Supports both 'dark' and 'light' futuristic visual themes.
        """
        sub = self.get_subgraph(
            filter_types=filter_types,
            filter_theme=filter_theme,
            focus_node=focus_node,
            include_structural=include_structural,
        )

        is_light = (theme == "light")
        empty_bg = "radial-gradient(circle, #f8fafc 0%, #edf2f9 100%)" if is_light else "radial-gradient(circle, #0f172a 0%, #060913 100%)"
        empty_border = "rgba(99, 102, 241, 0.25)" if is_light else "rgba(255,255,255,0.1)"
        empty_text = "#64748b" if is_light else "#94a3b8"

        if sub.number_of_nodes() == 0:
            return (
                f'<div style="height: 400px; display: flex; align-items: center; justify-content: center; '
                f'background: {empty_bg}; color: {empty_text}; '
                f'border: 1px dashed {empty_border}; border-radius: 12px; font-family: sans-serif;">'
                f'<span>No nodes match the selected filters.</span></div>'
            )

        # Highlight sets
        hl_node_set: Set[str] = set()
        if highlight_nodes:
            lower_targets = {h.lower() for h in highlight_nodes if h}
            for nid, ndata in sub.nodes(data=True):
                if nid in highlight_nodes or nid.lower() in lower_targets or ndata.get("name", "").lower() in lower_targets:
                    hl_node_set.add(nid)

        hl_edge_set: Set[Tuple[str, str]] = set()
        if highlight_edges:
            for u, v in highlight_edges:
                hl_edge_set.add((u, v))
                hl_edge_set.add((v, u))

        has_active_highlight = bool(hl_node_set)

        # Build 3D nodes payload
        nodes_payload = []
        for nid, data in sub.nodes(data=True):
            name = data.get("name", nid)
            etype = data.get("entity_type", "CONCEPT")
            mentions = data.get("mention_count", 1)
            desc = data.get("description", "")
            themes = data.get("themes", [])
            is_hl = nid in hl_node_set
            is_struct_node = bool(data.get("is_structural"))

            base_color = ENTITY_TYPE_COLORS.get(etype, DEFAULT_NODE_COLOR)
            node_color = "#38bdf8" if is_hl else base_color
            node_val = max(14, min(20 + mentions * 4, 38))
            if is_hl:
                node_val += 10
            elif is_struct_node:
                node_val = 30
                node_color = "#a855f7"

            nodes_payload.append({
                "id": nid,
                "name": name,
                "type": etype,
                "description": desc,
                "val": node_val,
                "color": node_color,
                "mentions": mentions,
                "themes": themes,
                "highlighted": is_hl,
                "dimmed": has_active_highlight and not is_hl,
                "is_structural": is_struct_node,
            })

        # Build 3D links payload with guaranteed visibility and high contrast
        links_payload = []
        for u, v, data in sub.edges(data=True):
            pred = data.get("predicate", "RELATES_TO")
            conf = float(data.get("confidence", 1.0))
            ev_list = data.get("evidence", [])
            first_ev = ev_list[0] if ev_list else {}
            ev_text = first_ev.get("evidence_text", "") if isinstance(first_ev, dict) else getattr(first_ev, "evidence_text", "")

            is_hl = (u, v) in hl_edge_set or (v, u) in hl_edge_set
            is_struct = bool(data.get("is_structural"))

            if is_light:
                if is_hl:
                    link_color = "#0284c7"
                elif is_struct:
                    link_color = "#a855f7"
                else:
                    link_color = "#6366f1" if not has_active_highlight else "#94a3b8"
            else:
                if is_hl:
                    link_color = "#38bdf8"
                elif is_struct:
                    link_color = "#c084fc"
                else:
                    link_color = "#818cf8" if not has_active_highlight else "#64748b"

            link_width = 3.5 if is_hl else (1.8 if is_struct else 2.2)
            particles = 4 if is_hl else (2 if not has_active_highlight else 0)

            links_payload.append({
                "source": u,
                "target": v,
                "predicate": pred,
                "confidence": conf,
                "evidence": ev_text,
                "color": link_color,
                "width": link_width,
                "particles": particles,
                "highlighted": is_hl,
                "is_structural": is_struct,
            })

        graph_data_json = json.dumps({"nodes": nodes_payload, "links": links_payload})
        clean_height = height if "px" in height else f"{height}px"

        # Theme design tokens
        body_bg = "radial-gradient(circle at 50% 50%, #ffffff 0%, #edf2f9 100%)" if is_light else "radial-gradient(circle at 50% 50%, #0d1527 0%, #050811 100%)"
        three_bg = "#f4f6fc" if is_light else "#050811"
        body_color = "#172033" if is_light else "#e2e8f0"
        hud_bg = "rgba(255, 255, 255, 0.88)" if is_light else "rgba(15, 23, 42, 0.75)"
        hud_border = "rgba(99, 102, 241, 0.22)" if is_light else "rgba(255, 255, 255, 0.1)"
        hud_shadow = "0 8px 32px 0 rgba(0, 0, 0, 0.08)" if is_light else "0 8px 32px 0 rgba(0, 0, 0, 0.37)"
        hud_title_color = "#6366f1" if is_light else "#38bdf8"
        hud_info_color = "#334155" if is_light else "#cbd5e1"
        ctrl_bg = "rgba(255, 255, 255, 0.95)" if is_light else "rgba(20, 28, 48, 0.95)"
        ctrl_border = "rgba(148, 163, 184, 0.6)" if is_light else "rgba(139, 92, 246, 0.4)"
        ctrl_color = "#0f172a" if is_light else "#f8fafc"
        node_card_bg = "rgba(255, 255, 255, 0.94)" if is_light else "rgba(15, 23, 42, 0.85)"
        node_card_border = "rgba(99, 102, 241, 0.35)" if is_light else "rgba(56, 189, 248, 0.4)"
        node_card_shadow = "0 12px 36px rgba(0, 0, 0, 0.12)" if is_light else "0 12px 36px rgba(0, 0, 0, 0.5)"
        nc_title_color = "#0f172a" if is_light else "#f8fafc"
        nc_desc_color = "#475569" if is_light else "#94a3b8"
        fallback_clear_color = "#f4f6fc" if is_light else "#050811"
        link_arrow_default = "#4f46e5" if is_light else "#a5b4fc"
        link_particle_default = "#818cf8" if is_light else "#c7d2fe"
        fallback_stroke_color = "#6366f1" if is_light else "#818cf8"
        fallback_text_color = "#1e293b" if is_light else "#cbd5e1"
        link_opacity_val = "0.85" if is_light else "0.90"
        amb_light_val = "1.0" if is_light else "0.85"

        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
      background: {body_bg};
      overflow: hidden;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      color: {body_color};
      width: 100%;
      height: {clean_height};
      position: relative;
    }}
    #3d-graph {{
      width: 100%;
      height: 100%;
    }}
    /* HUD Overlay */
    .hud-panel {{
      position: absolute;
      top: 14px;
      left: 14px;
      background: {hud_bg};
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid {hud_border};
      border-radius: 10px;
      padding: 10px 14px;
      font-size: 11px;
      pointer-events: none;
      box-shadow: {hud_shadow};
      max-width: 280px;
      z-index: 10;
    }}
    .hud-title {{
      font-weight: 700;
      color: {hud_title_color};
      letter-spacing: 0.04em;
      text-transform: uppercase;
      font-size: 10px;
      margin-bottom: 3px;
      display: flex;
      align-items: center;
      gap: 5px;
    }}
    .hud-info {{
      color: {hud_info_color};
      font-size: 11px;
      line-height: 1.4;
    }}
    /* Floating Controls */
    .controls-overlay {{
      position: absolute;
      top: 14px;
      right: 14px;
      display: flex;
      gap: 6px;
      z-index: 10;
    }}
    .ctrl-btn {{
      background: {ctrl_bg};
      border: 1px solid {ctrl_border};
      color: {ctrl_color};
      padding: 6px 10px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
      backdrop-filter: blur(8px);
      transition: all 0.2s ease;
    }}
    .ctrl-btn:hover {{
      color: #6366f1;
      background: rgba(99, 102, 241, 0.15);
      border-color: #6366f1;
      box-shadow: 0 0 12px rgba(99, 102, 241, 0.25);
    }}
    /* Node Detail Card */
    .node-card {{
      position: absolute;
      bottom: 14px;
      left: 14px;
      background: {node_card_bg};
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
      border: 1px solid {node_card_border};
      border-radius: 10px;
      padding: 12px 16px;
      font-size: 12px;
      box-shadow: {node_card_shadow};
      max-width: 320px;
      display: none;
      z-index: 15;
      animation: fadeIn 0.25s ease forwards;
    }}
    @keyframes fadeIn {{
      from {{ opacity: 0; transform: translateY(6px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    .nc-badge {{
      display: inline-block;
      font-size: 9px;
      font-weight: 700;
      padding: 2px 6px;
      border-radius: 4px;
      background: rgba(99, 102, 241, 0.15);
      color: #6366f1;
      border: 1px solid rgba(99, 102, 241, 0.3);
      margin-bottom: 4px;
      text-transform: uppercase;
    }}
    .nc-title {{
      font-size: 14px;
      font-weight: 700;
      color: {nc_title_color};
      margin-bottom: 4px;
    }}
    .nc-desc {{
      font-size: 11px;
      color: {nc_desc_color};
      line-height: 1.4;
      margin-bottom: 6px;
    }}
    .nc-meta {{
      font-size: 10px;
      color: #64748b;
      border-top: 1px solid rgba(148, 163, 184, 0.15);
      padding-top: 5px;
    }}
    /* Fallback Canvas */
    #fallback-canvas {{
      display: none;
      width: 100%;
      height: 100%;
    }}
  </style>
  <script src="https://unpkg.com/three@0.160.0/build/three.min.js"></script>
  <script src="https://unpkg.com/3d-force-graph@1.73.3/dist/3d-force-graph.min.js"></script>
</head>
<body>
  <div id="3d-graph"></div>
  <canvas id="fallback-canvas"></canvas>

  <div class="hud-panel">
    <div class="hud-title"><span>🌌</span> Synapse 3D Knowledge Space</div>
    <div class="hud-info">
      <b>{len(nodes_payload)}</b> nodes • <b>{len(links_payload)}</b> connections<br/>
      <span style="color:#64748b;">Rotate: Left Click • Pan: Right Click • Zoom: Scroll</span>
    </div>
  </div>

  <div class="controls-overlay">
    <button class="ctrl-btn" id="btn-reset" title="Reset camera to center">⟲ Reset Camera</button>
    <button class="ctrl-btn" id="btn-auto" title="Toggle gentle rotation">⚡ Auto-Orbit</button>
  </div>

  <div class="node-card" id="node-inspector">
    <span class="nc-badge" id="nc-type">CONCEPT</span>
    <div class="nc-title" id="nc-name">Entity Name</div>
    <div class="nc-desc" id="nc-desc">Entity description excerpt...</div>
    <div class="nc-meta" id="nc-meta">Mentions: 1 • Connected concepts</div>
  </div>

  <script>
    const gData = {graph_data_json};
    let graphInstance = null;
    let isAutoRotating = true;

    function init3DGraph() {{
      if (typeof ForceGraph3D === 'undefined' || typeof THREE === 'undefined') {{
        runFallbackCanvas();
        return;
      }}

      try {{
        const elem = document.getElementById('3d-graph');
        graphInstance = ForceGraph3D()(elem)
          .graphData(gData)
          .backgroundColor('{three_bg}')
          .showNavInfo(false)
          .nodeId('id')
          .nodeLabel(node => `<div style="background:rgba(15,23,42,0.92);padding:6px 10px;border-radius:8px;border:1px solid #38bdf8;font-size:12px;color:#fff;box-shadow:0 4px 16px rgba(0,0,0,0.4);"><b>${{node.name}}</b> [${{node.type}}]<br/><span style="color:#94a3b8;font-size:10px;">${{node.themes.join(', ') || 'No theme'}}</span></div>`)
          .nodeColor(node => node.dimmed ? 'rgba(71, 85, 105, 0.35)' : node.color)
          .nodeVal(node => node.val)
          .nodeResolution(16)
          .linkSource('source')
          .linkTarget('target')
          .linkOpacity({link_opacity_val})
          .linkWidth(link => link.width || 2.0)
          .linkResolution(8)
          .linkColor(link => link.color)
          .linkDirectionalArrowLength(link => link.highlighted ? 6.0 : 4.5)
          .linkDirectionalArrowRelPos(0.92)
          .linkDirectionalArrowColor(link => link.highlighted ? '#38bdf8' : '{link_arrow_default}')
          .linkDirectionalArrowResolution(8)
          .linkDirectionalParticles(link => link.particles !== undefined ? link.particles : (link.highlighted ? 4 : 2))
          .linkDirectionalParticleSpeed(0.006)
          .linkDirectionalParticleWidth(link => link.highlighted ? 3.5 : 2.2)
          .linkDirectionalParticleColor(link => link.highlighted ? '#38bdf8' : '{link_particle_default}')
          .linkLabel(link => `<div style="background:rgba(15,23,42,0.92);padding:5px 9px;border-radius:6px;border:1px solid #818cf8;font-size:11px;color:#e2e8f0;box-shadow:0 4px 12px rgba(0,0,0,0.4);"><b>${{link.predicate}}</b> (conf: ${{link.confidence.toFixed(2)}})</div>`)
          .onNodeClick(node => {{
            // Smoothly fly camera to clicked node
            const distance = 80;
            const distRatio = 1 + distance / Math.hypot(node.x, node.y, node.z);
            graphInstance.cameraPosition(
              {{ x: node.x * distRatio, y: node.y * distRatio, z: node.z * distRatio }},
              node,
              1500
            );
            showNodeCard(node);
          }})
          .onNodeHover(node => {{
            elem.style.cursor = node ? 'pointer' : 'default';
            if (node) {{
              showNodeCard(node);
            }}
          }});

        // Ensure bright, uniform 3D lighting so link cylinders and node spheres pop vividly
        setTimeout(() => {{
          if (graphInstance && graphInstance.scene) {{
            const scene = graphInstance.scene();
            const amb = new THREE.AmbientLight(0xffffff, {amb_light_val});
            scene.add(amb);
            const hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 0.6);
            scene.add(hemi);
          }}
        }}, 60);

        // Gentle auto-rotation
        let angle = 0;
        setInterval(() => {{
          if (isAutoRotating && graphInstance) {{
            angle += Math.PI / 1800;
            const dist = 360;
            graphInstance.cameraPosition({{
              x: dist * Math.sin(angle),
              z: dist * Math.cos(angle)
            }});
          }}
        }}, 35);

        // Auto-focus on evidence nodes if query highlights exist
        const highlightedNodes = gData.nodes.filter(n => n.highlighted);
        if (highlightedNodes.length > 0) {{
          setTimeout(() => {{
            isAutoRotating = false;
            let avgX = 0, avgY = 0, avgZ = 0;
            highlightedNodes.forEach(n => {{ avgX += (n.x || 0); avgY += (n.y || 0); avgZ += (n.z || 0); }});
            avgX /= highlightedNodes.length;
            avgY /= highlightedNodes.length;
            avgZ /= highlightedNodes.length;

            graphInstance.cameraPosition(
              {{ x: avgX + 60, y: avgY + 40, z: avgZ + 140 }},
              {{ x: avgX, y: avgY, z: avgZ }},
              2000
            );
          }}, 800);
        }}

      }} catch (err) {{
        console.warn('3D Force Graph failed, using fallback:', err);
        runFallbackCanvas();
      }}
    }}

    function showNodeCard(node) {{
      const card = document.getElementById('node-inspector');
      document.getElementById('nc-name').innerText = node.name;
      document.getElementById('nc-type').innerText = node.type;
      document.getElementById('nc-desc').innerText = node.description ? (node.description.slice(0, 160) + '...') : 'No explicit description.';
      document.getElementById('nc-meta').innerText = `Mentions: ${{node.mentions}} • Themes: ${{node.themes.join(', ') || 'General'}}`;
      card.style.display = 'block';
    }}

    document.getElementById('btn-reset').addEventListener('click', () => {{
      if (graphInstance) {{
        isAutoRotating = false;
        graphInstance.cameraPosition({{ x: 0, y: 0, z: 320 }}, {{ x: 0, y: 0, z: 0 }}, 1200);
      }}
    }});

    document.getElementById('btn-auto').addEventListener('click', () => {{
      isAutoRotating = !isAutoRotating;
      document.getElementById('btn-auto').style.borderColor = isAutoRotating ? '#38bdf8' : 'rgba(255,255,255,0.15)';
    }});

    // Offline HTML5 Canvas 3D Perspective Fallback
    function runFallbackCanvas() {{
      document.getElementById('3d-graph').style.display = 'none';
      const canvas = document.getElementById('fallback-canvas');
      canvas.style.display = 'block';
      const ctx = canvas.getContext('2d');
      canvas.width = canvas.parentElement.clientWidth || 800;
      canvas.height = canvas.parentElement.clientHeight || 500;

      // Initialize node coordinates with spherical distribution
      const nodes = gData.nodes.map((n, i) => {{
        const phi = Math.acos(-1 + (2 * i) / Math.max(1, gData.nodes.length));
        const theta = Math.sqrt(Math.max(1, gData.nodes.length) * Math.PI) * phi;
        const radius = 130 + (i % 3) * 35;
        return {{
          ...n,
          x: radius * Math.cos(theta) * Math.sin(phi),
          y: radius * Math.sin(theta) * Math.sin(phi),
          z: radius * Math.cos(phi),
        }};
      }});

      let rotX = 0.003, rotY = 0.005;
      function render() {{
        ctx.fillStyle = '{fallback_clear_color}';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const cx = canvas.width / 2;
        const cy = canvas.height / 2;
        const fov = 350;

        // Rotate nodes
        nodes.forEach(n => {{
          const cosY = Math.cos(rotY), sinY = Math.sin(rotY);
          const x1 = n.x * cosY - n.z * sinY;
          const z1 = n.z * cosY + n.x * sinY;
          n.x = x1; n.z = z1;
        }});

        // Draw connections with high contrast and directional arrows
        gData.links.forEach(l => {{
          const sId = l.source.id || l.source;
          const tId = l.target.id || l.target;
          const s = nodes.find(n => n.id === sId);
          const t = nodes.find(n => n.id === tId);
          if (s && t && s.z + fov > 0 && t.z + fov > 0) {{
            const sScale = fov / (fov + s.z);
            const tScale = fov / (fov + t.z);
            const sx = cx + s.x * sScale, sy = cy + s.y * sScale;
            const tx = cx + t.x * tScale, ty = cy + t.y * tScale;

            ctx.lineWidth = l.highlighted ? 3.0 : (l.is_structural ? 1.8 : 2.2);
            ctx.strokeStyle = l.color || (l.highlighted ? '#38bdf8' : '{fallback_stroke_color}');
            ctx.beginPath();
            ctx.moveTo(sx, sy);
            ctx.lineTo(tx, ty);
            ctx.stroke();

            // Draw directional arrowhead on the canvas line
            const midX = (sx + tx) / 2;
            const midY = (sy + ty) / 2;
            const angle = Math.atan2(ty - sy, tx - sx);
            const arrowLen = 7 * ((sScale + tScale) / 2);
            ctx.fillStyle = ctx.strokeStyle;
            ctx.beginPath();
            ctx.moveTo(midX, midY);
            ctx.lineTo(midX - arrowLen * Math.cos(angle - Math.PI / 6), midY - arrowLen * Math.sin(angle - Math.PI / 6));
            ctx.lineTo(midX - arrowLen * Math.cos(angle + Math.PI / 6), midY - arrowLen * Math.sin(angle + Math.PI / 6));
            ctx.closePath();
            ctx.fill();
          }}
        }});

        // Draw nodes
        nodes.forEach(n => {{
          if (n.z + fov > 0) {{
            const scale = fov / (fov + n.z);
            const x = cx + n.x * scale;
            const y = cy + n.y * scale;
            const radius = Math.max(3, (n.val / 3.5) * scale);

            ctx.beginPath();
            ctx.arc(x, y, radius, 0, Math.PI * 2);
            ctx.fillStyle = n.highlighted ? '#38bdf8' : n.color;
            ctx.shadowBlur = n.highlighted ? 15 : 6;
            ctx.shadowColor = n.highlighted ? '#38bdf8' : n.color;
            ctx.fill();
            ctx.shadowBlur = 0;

            if (scale > 0.8 || n.highlighted) {{
              ctx.font = `${{Math.round(11 * scale)}}px sans-serif`;
              ctx.fillStyle = n.highlighted ? '#fff' : '{fallback_text_color}';
              ctx.fillText(n.name, x + radius + 4, y + 4);
            }}
          }}
        }});

        requestAnimationFrame(render);
      }}
      render();
    }}
    window.addEventListener('load', () => {{
      setTimeout(init3DGraph, 100);
    }});
  </script>
</body>
</html>
        """

    @staticmethod
    def generate_empty_state_3d_html(height: str = "400px", theme: str = "dark") -> str:
        """Generate a cinematic 3D synaptic knowledge constellation for empty/welcome state.

        Runs with 100% offline Canvas 3D particle physics, mouse parallax, and luminous filaments.
        Adapts surface lighting and glow to current theme ('dark' or 'light').
        """
        clean_height = height if "px" in height else f"{height}px"
        is_light = (theme == "light")

        body_bg = "radial-gradient(circle at 50% 50%, #ffffff 0%, #edf2fb 100%)" if is_light else "radial-gradient(circle at 50% 50%, #0d1527 0%, #060913 100%)"
        body_color = "#172033" if is_light else "#e2e8f0"
        center_bg = "rgba(255, 255, 255, 0.88)" if is_light else "rgba(15, 23, 42, 0.72)"
        center_border = "rgba(99, 102, 241, 0.25)" if is_light else "rgba(255, 255, 255, 0.12)"
        center_shadow = "0 20px 50px rgba(0, 0, 0, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.9)" if is_light else "0 20px 50px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.15)"
        badge_bg = "rgba(99, 102, 241, 0.12)" if is_light else "rgba(56, 189, 248, 0.12)"
        badge_border = "rgba(99, 102, 241, 0.3)" if is_light else "rgba(56, 189, 248, 0.3)"
        badge_color = "#6366f1" if is_light else "#38bdf8"
        headline_grad = "linear-gradient(135deg, #0f172a 30%, #475569 100%)" if is_light else "linear-gradient(135deg, #ffffff 30%, #94a3b8 100%)"
        subtext_color = "#475569" if is_light else "#94a3b8"
        pill_bg = "rgba(241, 245, 249, 0.85)" if is_light else "rgba(30, 41, 59, 0.6)"
        pill_border = "rgba(203, 213, 225, 0.8)" if is_light else "rgba(255, 255, 255, 0.08)"
        pill_color = "#334155" if is_light else "#cbd5e1"
        canvas_clear = "#f4f6fc" if is_light else "#060913"

        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
      background: {body_bg};
      overflow: hidden;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      color: {body_color};
      width: 100%;
      height: {clean_height};
      position: relative;
    }}
    canvas {{
      width: 100%;
      height: 100%;
      display: block;
    }}
    .center-overlay {{
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      text-align: center;
      background: {center_bg};
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid {center_border};
      border-radius: 16px;
      padding: 24px 32px;
      max-width: 540px;
      width: 90%;
      box-shadow: {center_shadow};
      pointer-events: none;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      padding: 4px 10px;
      border-radius: 9999px;
      background: {badge_bg};
      color: {badge_color};
      border: 1px solid {badge_border};
      margin-bottom: 12px;
    }}
    .headline {{
      font-size: 20px;
      font-weight: 800;
      color: #f8fafc;
      letter-spacing: -0.02em;
      margin-bottom: 8px;
      line-height: 1.25;
      background: {headline_grad};
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    .subtext {{
      font-size: 12.5px;
      color: {subtext_color};
      line-height: 1.5;
      margin-bottom: 14px;
    }}
    .feature-pills {{
      display: flex;
      justify-content: center;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .pill {{
      font-size: 10px;
      font-weight: 600;
      padding: 3px 8px;
      border-radius: 6px;
      background: {pill_bg};
      border: 1px solid {pill_border};
      color: {pill_color};
    }}
  </style>
</head>
<body>
  <canvas id="c"></canvas>
  <div class="center-overlay">
    <div class="badge"><span>🧠</span> Synaptic Knowledge Graph Engine</div>
    <div class="headline">Your documents. Connected knowledge. Clear answers.</div>
    <div class="subtext">
      Transform scattered documents into an explorable 3D knowledge universe. Discover themes across sources, trace relationships between entities, and inspect the evidence behind every answer.
    </div>
    <div class="feature-pills">
      <span class="pill">🕸️ 3D Spatial Topology</span>
      <span class="pill">🔍 Hybrid Vector & Graph Routing</span>
      <span class="pill">📚 Verifiable Provenance</span>
      <span class="pill">⚡ Neo4j & ChromaDB</span>
    </div>
  </div>

  <script>
    const canvas = document.getElementById('c');
    const ctx = canvas.getContext('2d');
    let width = canvas.width = window.innerWidth;
    let height = canvas.height = window.innerHeight;

    window.addEventListener('resize', () => {{
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
    }});

    // Generate orbiting synaptic nodes
    const nodeCount = 55;
    const nodes = [];
    const colors = ['#8B5CF6', '#6366F1', '#22D3EE', '#EC4899', '#2DD4BF'];

    for (let i = 0; i < nodeCount; i++) {{
      const radius = 180 + Math.random() * 160;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos((Math.random() * 2) - 1);
      nodes.push({{
        x: radius * Math.sin(phi) * Math.cos(theta),
        y: radius * Math.sin(phi) * Math.sin(theta),
        z: radius * Math.cos(phi),
        color: colors[i % colors.length],
        size: 2.5 + Math.random() * 3,
        orbitSpeed: (0.001 + Math.random() * 0.0015) * (Math.random() > 0.5 ? 1 : -1)
      }});
    }}

    let mouseX = 0, mouseY = 0;
    window.addEventListener('mousemove', (e) => {{
      mouseX = (e.clientX - width / 2) * 0.0003;
      mouseY = (e.clientY - height / 2) * 0.0003;
    }});

    let rotX = 0.0012, rotY = 0.002;
    function render() {{
      ctx.fillStyle = '{canvas_clear}';
      ctx.fillRect(0, 0, width, height);

      const cx = width / 2;
      const cy = height / 2;
      const fov = 380;

      // Update positions with mouse parallax and orbit
      nodes.forEach(n => {{
        const cosY = Math.cos(rotY + mouseX), sinY = Math.sin(rotY + mouseX);
        const x1 = n.x * cosY - n.z * sinY;
        const z1 = n.z * cosY + n.x * sinY;

        const cosX = Math.cos(rotX + mouseY), sinX = Math.sin(rotX + mouseY);
        const y2 = n.y * cosX - z1 * sinX;
        const z2 = z1 * cosX + n.y * sinX;

        n.x = x1; n.y = y2; n.z = z2;
      }});

      // Draw filaments between nearby nodes
      for (let i = 0; i < nodes.length; i++) {{
        for (let j = i + 1; j < nodes.length; j++) {{
          const a = nodes[i], b = nodes[j];
          const dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
          const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
          if (dist < 110) {{
            const aScale = fov / (fov + a.z);
            const bScale = fov / (fov + b.z);
            if (a.z + fov > 0 && b.z + fov > 0) {{
              const ax = cx + a.x * aScale, ay = cy + a.y * aScale;
              const bx = cx + b.x * bScale, by = cy + b.y * bScale;
              const alpha = (1 - dist / 110) * 0.35;
              ctx.strokeStyle = `rgba(99, 102, 241, ${{alpha}})`;
              ctx.lineWidth = 1;
              ctx.beginPath();
              ctx.moveTo(ax, ay);
              ctx.lineTo(bx, by);
              ctx.stroke();
            }}
          }}
        }}
      }}

      // Draw glowing nodes
      nodes.forEach(n => {{
        if (n.z + fov > 0) {{
          const scale = fov / (fov + n.z);
          const x = cx + n.x * scale;
          const y = cy + n.y * scale;
          const rad = n.size * scale;

          ctx.beginPath();
          ctx.arc(x, y, rad, 0, Math.PI * 2);
          ctx.fillStyle = n.color;
          ctx.shadowBlur = 10;
          ctx.shadowColor = n.color;
          ctx.fill();
          ctx.shadowBlur = 0;
        }}
      }});

      requestAnimationFrame(render);
    }}
    render();
  </script>
</body>
</html>
        """

    @staticmethod
    def generate_intro_animation_html(theme: str = "dark") -> str:
        """Generate a cinematic 1.8-second entrance sequence introducing the futuristic SynapseAI workspace.

        Features ambient emergence, brand gradient reveal (electric violet, indigo, cyan),
        synaptic knowledge network formation, and capability highlights. Respects prefers-reduced-motion.
        """
        is_light = (theme == "light")
        bg_style = "radial-gradient(circle at 50% 40%, #eef2fb 0%, #f4f6fc 55%, #e2e8f0 100%)" if is_light else "radial-gradient(circle at 50% 40%, #151e38 0%, #0b1020 50%, #060913 100%)"
        text_color = "#172033" if is_light else "#f8fafc"
        sub_color = "#64748b" if is_light else "#94a3b8"
        card_bg = "rgba(255, 255, 255, 0.85)" if is_light else "rgba(17, 24, 43, 0.85)"
        card_border = "rgba(99, 102, 241, 0.3)" if is_light else "rgba(139, 92, 246, 0.35)"
        glow_shadow = "0 25px 60px rgba(99, 102, 241, 0.18)" if is_light else "0 25px 60px rgba(0, 0, 0, 0.7)"

        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
      background: {bg_style};
      color: {text_color};
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      height: 380px;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
      position: relative;
    }}
    canvas {{
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      z-index: 1;
    }}
    .intro-container {{
      position: relative;
      z-index: 10;
      text-align: center;
      max-width: 680px;
      width: 90%;
      padding: 30px 40px;
      background: {card_bg};
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      border: 1px solid {card_border};
      border-radius: 20px;
      box-shadow: {glow_shadow};
      animation: floatIn 1.2s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }}
    @keyframes floatIn {{
      0% {{ opacity: 0; transform: translateY(24px) scale(0.96); }}
      100% {{ opacity: 1; transform: translateY(0) scale(1); }}
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 12px;
      border-radius: 9999px;
      background: linear-gradient(90deg, rgba(139, 92, 246, 0.15), rgba(34, 211, 238, 0.15));
      border: 1px solid rgba(139, 92, 246, 0.4);
      color: #8b5cf6;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 12px;
    }}
    .brand-title {{
      font-size: 38px;
      font-weight: 900;
      letter-spacing: -0.03em;
      line-height: 1.1;
      margin-bottom: 10px;
      background: linear-gradient(120deg, #8B5CF6 0%, #6366F1 45%, #22D3EE 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      text-shadow: 0 0 30px rgba(139, 92, 246, 0.25);
    }}
    .tagline {{
      font-size: 14px;
      color: {sub_color};
      line-height: 1.5;
      margin-bottom: 20px;
      font-weight: 400;
    }}
    .feature-grid {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
      margin-bottom: 22px;
    }}
    .feat-card {{
      background: rgba(139, 92, 246, 0.08);
      border: 1px solid rgba(139, 92, 246, 0.2);
      border-radius: 10px;
      padding: 8px 6px;
      font-size: 11px;
      font-weight: 600;
      color: {text_color};
      transition: all 0.2s ease;
    }}
    .progress-bar-container {{
      width: 100%;
      height: 4px;
      background: rgba(99, 102, 241, 0.15);
      border-radius: 9999px;
      overflow: hidden;
      margin-top: 10px;
    }}
    .progress-fill {{
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #8B5CF6, #6366F1, #22D3EE);
      border-radius: 9999px;
      animation: fillProgress 1.8s ease-in-out forwards;
    }}
    @keyframes fillProgress {{
      0% {{ width: 0%; }}
      100% {{ width: 100%; }}
    }}
    @media (prefers-reduced-motion: reduce) {{
      .intro-container {{ animation: none; opacity: 1; }}
      .progress-fill {{ width: 100%; animation: none; }}
    }}
  </style>
</head>
<body>
  <canvas id="c-intro"></canvas>

  <div class="intro-container">
    <div class="badge"><span>⚡</span> Hypnexis AI Innovation Lab</div>
    <div class="brand-title">SYNAPSE AI</div>
    <p class="tagline">
      The Topic-Augmented Knowledge Graph & Neural Intelligence Workspace.<br/>
      Unifying Cross-Document Synthesis, Vector Embeddings, and Graph RAG.
    </p>

    <div class="feature-grid">
      <div class="feat-card">📄 Ingestion & Chunks</div>
      <div class="feat-card">🕸️ 3D Knowledge Graph</div>
      <div class="feat-card">🔍 Vector Semantic RAG</div>
      <div class="feat-card">⚡ Grounded Answers</div>
    </div>

    <div class="progress-bar-container">
      <div class="progress-fill"></div>
    </div>
  </div>

  <script>
    const canvas = document.getElementById('c-intro');
    const ctx = canvas.getContext('2d');
    let w = canvas.width = window.innerWidth;
    let h = canvas.height = window.innerHeight;

    window.addEventListener('resize', () => {{
      w = canvas.width = window.innerWidth;
      h = canvas.height = window.innerHeight;
    }});

    const particles = [];
    const colors = ['#8B5CF6', '#6366F1', '#22D3EE', '#EC4899', '#2DD4BF'];
    for (let i = 0; i < 40; i++) {{
      particles.push({{
        x: Math.random() * w,
        y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.8,
        vy: (Math.random() - 0.5) * 0.8,
        radius: 1.5 + Math.random() * 2.5,
        color: colors[i % colors.length]
      }});
    }}

    function animate() {{
      ctx.clearRect(0, 0, w, h);

      for (let i = 0; i < particles.length; i++) {{
        for (let j = i + 1; j < particles.length; j++) {{
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 100) {{
            ctx.beginPath();
            ctx.strokeStyle = `rgba(99, 102, 241, ${{0.25 * (1 - dist / 100)}})`;
            ctx.lineWidth = 1;
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.stroke();
          }}
        }}
      }}

      particles.forEach(p => {{
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0 || p.x > w) p.vx *= -1;
        if (p.y < 0 || p.y > h) p.vy *= -1;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.shadowBlur = 8;
        ctx.shadowColor = p.color;
        ctx.fill();
        ctx.shadowBlur = 0;
      }});

      requestAnimationFrame(animate);
    }}
    animate();
  </script>
</body>
</html>
        """


