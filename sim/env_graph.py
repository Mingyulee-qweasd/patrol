"""그래프 기반 실내 환경 로더 — 병원(위상 그래프 + 점유격자).

v1/v2의 폴리라인(route) 환경과 별개. 로봇이 그래프 노드를 따라 순찰하고,
점유격자로 충돌·자유공간을 판정. 임무는 복도(엣지) 근방 자유공간에 배치.
"""
from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np
import yaml
import networkx as nx


@dataclass
class Robot:
    id: int
    role: str
    speed: float
    caps: frozenset
    sense_r: float
    reliable_r: float


@dataclass
class GraphEnv:
    name: str
    G: nx.Graph                 # 위상 그래프 (노드 attr: x,y / 엣지 attr: length)
    occ: np.ndarray             # 점유격자 (1=벽)
    res: float                  # m/cell
    origin: tuple               # (ox, oy) 월드 좌하단
    robots: list                # [Robot]
    task_cfg: dict
    node_xy: dict = field(default_factory=dict)   # nid -> np.array([x,y])
    _free_pts: np.ndarray = None                  # 임무 배치용 자유공간 샘플 (월드좌표)

    # ── 좌표 변환 ──
    def world_to_cell(self, xy):
        return (int((xy[1] - self.origin[1]) / self.res),
                int((xy[0] - self.origin[0]) / self.res))

    def is_free(self, xy) -> bool:
        i, j = self.world_to_cell(xy)
        H, W = self.occ.shape
        if not (0 <= i < H and 0 <= j < W):
            return False
        return self.occ[i, j] == 0

    def line_blocked(self, a, b) -> bool:
        """a→b 직선이 벽을 지나나 (시야 차단·이동 충돌 판정)."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = np.linalg.norm(b - a)
        for t in np.linspace(0, 1, max(2, int(d / (self.res * 2)))):
            if not self.is_free(a + t * (b - a)):
                return True
        return False

    # ── 그래프 순찰 유틸 ──
    def shortest(self, u, v):
        return nx.shortest_path(self.G, u, v, weight="length")

    def nearest_node(self, xy):
        xy = np.asarray(xy, float)
        return min(self.G.nodes, key=lambda n: np.hypot(*(self.node_xy[n] - xy)))

    def total_edge_len(self):
        return sum(d["length"] for _, _, d in self.G.edges(data=True))

    # ── 임무 배치용 자유공간 ──
    def free_points(self, rng, k=1):
        """복도 근방 자유공간에서 k개 좌표 무작위 추출."""
        idx = rng.integers(0, len(self._free_pts), size=k)
        return self._free_pts[idx]


def load_graph_env(cfg_path: str | Path) -> GraphEnv:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    root = Path(cfg_path).resolve().parents[2]   # repo 루트 (exp/configs/x.yaml → ../../)
    mdir = root / cfg["map"]["dir"]
    gj = json.loads((mdir / cfg["map"]["graph"]).read_text())
    occ = np.load(mdir / cfg["map"]["occupancy"])
    meta = json.loads((mdir / cfg["map"]["meta"]).read_text())
    res = meta["res_m"]; origin = tuple(meta["origin_m"])

    G = nx.Graph()
    node_xy = {}
    for n in gj["nodes"]:
        G.add_node(n["id"], x=n["x"], y=n["y"])
        node_xy[n["id"]] = np.array([n["x"], n["y"]], float)
    for e in gj["edges"]:
        G.add_edge(e["u"], e["v"], length=e["length"])

    robots = [Robot(r["id"], r["role"], r["speed_mps"], frozenset(r["caps"]),
                    r["sense_radius_m"], r["reliable_radius_m"]) for r in cfg["robots"]]

    e = GraphEnv(name=cfg["name"], G=G, occ=occ, res=res, origin=origin,
                 robots=robots, task_cfg=cfg["tasks"], node_xy=node_xy)

    # 자유공간 샘플: 그래프 노드 근방(복도)만 — 임무가 로봇이 닿는 곳에 생기게
    H, W = occ.shape
    free_ij = np.argwhere(occ == 0)
    fx = origin[0] + free_ij[:, 1] * res
    fy = origin[1] + free_ij[:, 0] * res
    fpts = np.c_[fx, fy]
    # 그래프에서 3m 이내인 자유셀만 (방·복도 등 순찰 도달권)
    nodes = np.array([node_xy[n] for n in G.nodes])
    from scipy.spatial import cKDTree
    tree = cKDTree(nodes)
    d, _ = tree.query(fpts, k=1)
    e._free_pts = fpts[d < 4.0]
    return e


if __name__ == "__main__":
    import sys
    env = load_graph_env(sys.argv[1] if len(sys.argv) > 1 else "exp/configs/hospital.yaml")
    print(f"환경: {env.name}")
    print(f"  그래프: 노드 {env.G.number_of_nodes()}, 엣지 {env.G.number_of_edges()}, "
          f"연결성분 {nx.number_connected_components(env.G)}")
    print(f"  복도 총연장: {env.total_edge_len():.0f}m")
    print(f"  격자: {env.occ.shape}, 자유공간 임무배치점 {len(env._free_pts):,}")
    print(f"  로봇 {len(env.robots)}대:")
    for r in env.robots:
        print(f"    {r.role}: 속도 {r.speed}m/s, 능력 {set(r.caps) or '{인지만}'}, 감지 {r.sense_r}m")
    print(f"  임무 타입 {len(env.task_cfg['types'])}종")
