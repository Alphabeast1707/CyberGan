"""
CyberGAN — Arena: Network Graph Model
Represents the simulated enterprise network as a directed graph
with nodes (hosts), edges (connections), services, and vulnerabilities.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import yaml


class NodeType(Enum):
    """Type of network host."""
    ROUTER = "router"
    SERVER = "server"
    ENDPOINT = "endpoint"


class NodeState(Enum):
    """State of a node from the simulation perspective."""
    UNKNOWN = 0      # Red hasn't scanned it yet
    SCANNED = 1      # Red knows it exists and its services
    VULNERABLE = 2   # Red has identified exploitable vulns
    COMPROMISED = 3  # Red has gained access
    ISOLATED = 4     # Blue has quarantined it
    RESTORED = 5     # Blue has cleaned and restored it


@dataclass
class Service:
    """A network service running on a host."""
    name: str
    port: int
    version: str
    is_patched: bool = False


@dataclass
class Vulnerability:
    """A vulnerability on a specific node/service."""
    cve: str
    service: str
    severity: float       # CVSS score (0-10)
    description: str
    is_patched: bool = False
    is_exploited: bool = False
    times_exploited: int = 0


@dataclass
class NetworkNode:
    """A host in the network graph."""
    id: str
    node_type: NodeType
    services: list[Service] = field(default_factory=list)
    vulnerabilities: list[Vulnerability] = field(default_factory=list)
    state: NodeState = NodeState.UNKNOWN
    has_backdoor: bool = False
    has_honeypot: bool = False
    firewall_rules: list[int] = field(default_factory=list)  # allowed ports

    @property
    def is_compromised(self) -> bool:
        return self.state == NodeState.COMPROMISED

    @property
    def is_isolated(self) -> bool:
        return self.state == NodeState.ISOLATED

    @property
    def unpatched_vulns(self) -> list[Vulnerability]:
        return [v for v in self.vulnerabilities if not v.is_patched]

    @property
    def exploitable_vulns(self) -> list[Vulnerability]:
        return [v for v in self.vulnerabilities if not v.is_patched and not v.is_exploited]

    def health_score(self) -> float:
        """0.0 = fully compromised, 1.0 = perfectly healthy."""
        if self.state == NodeState.COMPROMISED:
            return 0.0
        if self.state == NodeState.ISOLATED:
            return 0.3
        total = len(self.vulnerabilities)
        if total == 0:
            return 1.0
        patched = sum(1 for v in self.vulnerabilities if v.is_patched)
        return patched / total

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.node_type.value,
            "state": self.state.name,
            "health": self.health_score(),
            "services": [{"name": s.name, "port": s.port, "version": s.version} for s in self.services],
            "vulns": [{"cve": v.cve, "severity": v.severity, "patched": v.is_patched, "exploited": v.is_exploited} for v in self.vulnerabilities],
            "has_backdoor": self.has_backdoor,
            "has_honeypot": self.has_honeypot,
        }


@dataclass
class NetworkEdge:
    """A connection between two hosts."""
    source: str
    target: str
    firewall_ports: list[int] = field(default_factory=list)
    is_blocked: bool = False

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "firewall_ports": self.firewall_ports,
            "blocked": self.is_blocked,
        }


class NetworkGraph:
    """
    The full simulated network topology.
    Directed graph of NetworkNodes connected by NetworkEdges.
    """

    def __init__(self):
        self.nodes: dict[str, NetworkNode] = {}
        self.edges: list[NetworkEdge] = []
        self._adjacency: dict[str, list[str]] = {}  # node_id -> [neighbor_ids]
        self._initial_state: Optional[dict] = None

    @classmethod
    def from_config(cls, config_path: str) -> NetworkGraph:
        """Build a NetworkGraph from a YAML config file."""
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config.get("network", {}))

    @classmethod
    def from_dict(cls, network_config: dict) -> NetworkGraph:
        """Build from a parsed config dictionary."""
        graph = cls()

        for node_cfg in network_config.get("nodes", []):
            services = [
                Service(
                    name=s["name"],
                    port=s["port"],
                    version=s["version"],
                )
                for s in node_cfg.get("services", [])
            ]
            vulns = [
                Vulnerability(
                    cve=v["cve"],
                    service=v["service"],
                    severity=v["severity"],
                    description=v["description"],
                )
                for v in node_cfg.get("vulnerabilities", [])
            ]
            node = NetworkNode(
                id=node_cfg["id"],
                node_type=NodeType(node_cfg["type"]),
                services=services,
                vulnerabilities=vulns,
            )
            graph.add_node(node)

        for conn in network_config.get("connections", []):
            edge = NetworkEdge(
                source=conn["from"],
                target=conn["to"],
                firewall_ports=conn.get("firewall", []),
            )
            graph.add_edge(edge)
            # Add reverse edge (bidirectional connectivity)
            reverse = NetworkEdge(
                source=conn["to"],
                target=conn["from"],
                firewall_ports=conn.get("firewall", []),
            )
            graph.add_edge(reverse)

        # Save initial state for reset
        graph._save_initial_state()
        return graph

    def add_node(self, node: NetworkNode):
        self.nodes[node.id] = node
        if node.id not in self._adjacency:
            self._adjacency[node.id] = []

    def add_edge(self, edge: NetworkEdge):
        self.edges.append(edge)
        if edge.source not in self._adjacency:
            self._adjacency[edge.source] = []
        if edge.target not in self._adjacency[edge.source]:
            self._adjacency[edge.source].append(edge.target)

    def neighbors(self, node_id: str) -> list[str]:
        """Get all reachable neighbors (not blocked edges)."""
        all_neighbors = self._adjacency.get(node_id, [])
        reachable = []
        for n in all_neighbors:
            edge = self.get_edge(node_id, n)
            if edge and not edge.is_blocked:
                reachable.append(n)
        return reachable

    def get_edge(self, source: str, target: str) -> Optional[NetworkEdge]:
        """Find the edge between two nodes."""
        for e in self.edges:
            if e.source == source and e.target == target:
                return e
        return None

    def get_node_index(self, node_id: str) -> int:
        """Get numeric index for a node ID."""
        return list(self.nodes.keys()).index(node_id)

    def get_node_by_index(self, index: int) -> Optional[NetworkNode]:
        """Get node by its numeric index."""
        keys = list(self.nodes.keys())
        if 0 <= index < len(keys):
            return self.nodes[keys[index]]
        return None

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_vulns(self) -> int:
        """Max vulnerabilities per any single node."""
        if not self.nodes:
            return 0
        return max(len(n.vulnerabilities) for n in self.nodes.values())

    @property
    def total_vulns(self) -> int:
        return sum(len(n.vulnerabilities) for n in self.nodes.values())

    def get_stats(self) -> dict:
        """Get current network health statistics."""
        total = self.total_vulns
        patched = sum(
            1 for n in self.nodes.values()
            for v in n.vulnerabilities if v.is_patched
        )
        compromised = sum(
            1 for n in self.nodes.values() if n.is_compromised
        )
        isolated = sum(
            1 for n in self.nodes.values() if n.is_isolated
        )
        avg_health = sum(n.health_score() for n in self.nodes.values()) / max(len(self.nodes), 1)
        return {
            "total_nodes": self.num_nodes,
            "total_vulns": total,
            "patched_vulns": patched,
            "vuln_coverage": patched / max(total, 1),
            "compromised": compromised,
            "isolated": isolated,
            "avg_health": avg_health,
        }

    def _save_initial_state(self):
        """Deep copy the initial state for reset."""
        self._initial_state = {}
        for nid, node in self.nodes.items():
            self._initial_state[nid] = {
                "state": node.state,
                "has_backdoor": node.has_backdoor,
                "has_honeypot": node.has_honeypot,
                "vulns": [(v.is_patched, v.is_exploited, v.times_exploited) for v in node.vulnerabilities],
            }
        self._initial_edges = [(e.source, e.target, e.is_blocked) for e in self.edges]

    def reset(self):
        """Reset all nodes and edges to initial state."""
        if self._initial_state is None:
            return
        for nid, state in self._initial_state.items():
            node = self.nodes[nid]
            node.state = state["state"]
            node.has_backdoor = state["has_backdoor"]
            node.has_honeypot = state["has_honeypot"]
            for v, (patched, exploited, count) in zip(node.vulnerabilities, state["vulns"]):
                v.is_patched = patched
                v.is_exploited = exploited
                v.times_exploited = count
        for edge, (src, tgt, blocked) in zip(self.edges, self._initial_edges):
            edge.is_blocked = blocked

    def to_dict(self) -> dict:
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
            "stats": self.get_stats(),
        }
