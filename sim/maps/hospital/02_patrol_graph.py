"""병원 순찰용 위상 지도 — 자동추출 그래프(graph_hospital)를 순찰 목적으로 정리.

원본 보존, 새 산출: patrol_graph.json, patrol_overview.png.
정리 기준(순찰 관점):
 1) 짧은 막다른 잔가지(방 안쪽 spur) 반복 제거 — dead-end 가지치기
 2) 가까운 노드 병합(1.5m) — 순찰 지점 성기게
 3) 최대 연결성분 + 끊긴 조각을 최근접으로 재연결 → 하나로
 4) 방 앞 접근점은 유지(복도에 붙은 짧은 가지 1단은 남김)
실행: conda patrol.
"""
import json
from pathlib import Path
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree

HERE = Path(__file__).parent
G0 = json.loads((HERE / "graph_hospital.json").read_text())
SPUR_LEN = 1.2      # 이보다 짧은 막다른 가지 제거 (m)
MERGE_R = 1.5       # 노드 병합 반경 (m)
RECONNECT_MAX = 6.0 # 끊긴 조각 재연결 최대 거리 (m)


def build(nodes, edges):
    G = nx.Graph()
    for n in nodes:
        G.add_node(n["id"], x=n["x"], y=n["y"])
    for e in edges:
        if G.has_node(e["u"]) and G.has_node(e["v"]):
            G.add_edge(e["u"], e["v"], length=e["length"])
    return G


def prune_spurs(G, max_len):
    """차수1(끝점)이면서 유일 엣지가 짧으면 제거, 반복."""
    changed = True
    while changed:
        changed = False
        for n in list(G.nodes):
            if G.degree(n) == 1:
                nb = next(iter(G[n]))
                if G[n][nb]["length"] < max_len:
                    G.remove_node(n); changed = True
    return G


def merge_close(G, r):
    """가까운 노드쌍을 반복 병합(합쳐진 위치=평균)."""
    changed = True
    while changed:
        changed = False
        ids = list(G.nodes)
        pts = np.array([[G.nodes[i]["x"], G.nodes[i]["y"]] for i in ids])
        tree = cKDTree(pts)
        pairs = tree.query_pairs(r)
        if pairs:
            i, j = min(pairs)  # 한 쌍씩 안전 병합
            a, b = ids[i], ids[j]
            # b를 a로 흡수
            G.nodes[a]["x"] = (G.nodes[a]["x"] + G.nodes[b]["x"]) / 2
            G.nodes[a]["y"] = (G.nodes[a]["y"] + G.nodes[b]["y"]) / 2
            for nb in list(G[b]):
                if nb != a:
                    d = np.hypot(G.nodes[a]["x"]-G.nodes[nb]["x"], G.nodes[a]["y"]-G.nodes[nb]["y"])
                    G.add_edge(a, nb, length=float(d))
            G.remove_node(b); changed = True
    return G


def connect_components(G, max_d):
    """끊긴 조각들을 최근접 노드쌍으로 이어 하나로."""
    while nx.number_connected_components(G) > 1:
        comps = list(nx.connected_components(G))
        comps.sort(key=len, reverse=True)
        main = comps[0]
        best = None
        mainpts = np.array([[G.nodes[i]["x"], G.nodes[i]["y"]] for i in main])
        main_ids = list(main)
        mtree = cKDTree(mainpts)
        for comp in comps[1:]:
            for c in comp:
                d, idx = mtree.query([G.nodes[c]["x"], G.nodes[c]["y"]])
                if best is None or d < best[0]:
                    best = (d, c, main_ids[idx])
        if best is None or best[0] > max_d:
            # 남은 작은 조각은 버림 (연결 불가)
            for comp in comps[1:]:
                if best is None or best[0] > max_d:
                    G.remove_nodes_from(comp)
            break
        _, a, b = best
        G.add_edge(a, b, length=float(best[0]))
    return G


def main():
    G = build(G0["nodes"], G0["edges"])
    print(f"원본: 노드 {G.number_of_nodes()}, 엣지 {G.number_of_edges()}, "
          f"조각 {nx.number_connected_components(G)}")
    G = prune_spurs(G, SPUR_LEN)
    print(f"가지치기 후: 노드 {G.number_of_nodes()}")
    G = merge_close(G, MERGE_R)
    print(f"병합 후: 노드 {G.number_of_nodes()}")
    G = connect_components(G, RECONNECT_MAX)
    print(f"연결 후: 노드 {G.number_of_nodes()}, 엣지 {G.number_of_edges()}, "
          f"조각 {nx.number_connected_components(G)}")

    # id 재부여
    remap = {old: i for i, old in enumerate(G.nodes)}
    out = {"res_m": G0["res_m"], "origin_m": G0["origin_m"],
           "nodes": [{"id": remap[n], "x": G.nodes[n]["x"], "y": G.nodes[n]["y"]} for n in G.nodes],
           "edges": [{"u": remap[a], "v": remap[b], "length": G[a][b]["length"]} for a, b in G.edges]}
    (HERE / "patrol_graph.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, matplotlib.font_manager as fm
    fp = fm.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc")
    occ = np.load(HERE / "occ_hospital.npy"); H, W = occ.shape
    ox, oy = G0["origin_m"]; RES = G0["res_m"]
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(occ, origin="lower", extent=[ox, ox+W*RES, oy, oy+H*RES], cmap="binary", alpha=0.5)
    for a, b in G.edges:
        ax.plot([G.nodes[a]["x"], G.nodes[b]["x"]], [G.nodes[a]["y"], G.nodes[b]["y"]],
                "g-", lw=2.2, alpha=0.8)
    for n in G.nodes:
        ax.plot(G.nodes[n]["x"], G.nodes[n]["y"], "ro", ms=7)
    ax.set_title(f"병원 순찰 위상 지도 (정리판) — 노드 {G.number_of_nodes()}, 엣지 {G.number_of_edges()}, "
                 f"1조각 연결\n빨강=순찰 지점, 초록=순찰 경로", fontproperties=fp, fontsize=15)
    ax.set_xlabel("x (m)", fontproperties=fp); ax.set_ylabel("y (m)", fontproperties=fp)
    ax.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(HERE / "patrol_overview.png", dpi=95)
    print("저장: patrol_graph.json, patrol_overview.png")


if __name__ == "__main__":
    main()
