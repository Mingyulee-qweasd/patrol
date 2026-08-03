"""병원 위상 지도(topological map) — 점유격자에서 복도 골격 → 노드·엣지 그래프.

자유공간(로봇 반경 팽창 여집합) → 최대연결성분 → 골격화 → 분기/끝점=노드,
골격 픽셀 추적으로 실제 엣지(벽 관통 없음). 산출: graph_hospital.json, map_overview.png.
실행: conda patrol.
"""
import json
from pathlib import Path
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize
import networkx as nx

HERE = Path(__file__).parent
RES = 0.05
ROBOT_R = 0.30
occ = np.load(HERE / "occ_hospital.npy")
meta = json.loads((HERE / "hospital_meta.json").read_text())
ox, oy = meta["origin_m"]
H, W = occ.shape


def main():
    # 1) 벽 팽창→여집합=자유공간, 최대 연결성분
    infl = ndimage.binary_dilation(occ, iterations=int(ROBOT_R / RES))
    free = ~infl
    lab, n = ndimage.label(free)
    if n:
        big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
        free = lab == big
    print(f"자유공간 {free.sum():,}셀 (성분 {n}개 중 최대)")

    # 2) 골격화 → 복도 중심선
    skel = skeletonize(free)
    print(f"골격 {skel.sum():,}픽셀")

    # 3) 노드 = 분기(이웃≥3) 또는 끝점(이웃1)
    nbr = ndimage.convolve(skel.astype(int), np.ones((3, 3)), mode="constant") - 1
    node_mask = skel & ((nbr >= 3) | (nbr == 1))
    ny, nx_ = np.where(node_mask)
    # 근접 노드 병합(0.8m)
    node_px = list(zip(ny, nx_))
    merged, used = [], np.zeros(len(node_px), bool)
    for i, (y, x) in enumerate(node_px):
        if used[i]:
            continue
        grp = [(y, x)]; used[i] = True
        for j in range(i+1, len(node_px)):
            if not used[j] and abs(node_px[j][0]-y) < 0.8/RES and abs(node_px[j][1]-x) < 0.8/RES:
                grp.append(node_px[j]); used[j] = True
        my = int(np.mean([g[0] for g in grp])); mx = int(np.mean([g[1] for g in grp]))
        merged.append((my, mx))
    print(f"노드 {len(merged)}개 (병합 후)")

    # 4) 엣지 = 골격을 따라 노드→노드 추적 (BFS, 벽 관통 없음)
    node_of = {}
    for idx, (y, x) in enumerate(merged):
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                node_of[(y+dy, x+dx)] = idx
    skel_set = set(zip(*np.where(skel)))
    G = nx.Graph()
    for idx, (y, x) in enumerate(merged):
        G.add_node(idx, x=float(ox + x*RES), y=float(oy + y*RES))
    # 각 노드에서 골격 팔로우
    for idx, (y0, x0) in enumerate(merged):
        starts = [(y0+dy, x0+dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                  if (y0+dy, x0+dx) in skel_set and (dy, dx) != (0, 0)]
        for s in starts:
            prev = (y0, x0); cur = s; steps = 0
            while steps < 4000:
                if cur in node_of and node_of[cur] != idx:
                    j = node_of[cur]
                    d = np.hypot(merged[j][0]-y0, merged[j][1]-x0) * RES
                    if not G.has_edge(idx, j):
                        G.add_edge(idx, j, length=float(d))
                    break
                nxts = [(cur[0]+dy, cur[1]+dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                        if (cur[0]+dy, cur[1]+dx) in skel_set
                        and (cur[0]+dy, cur[1]+dx) != prev and (dy, dx) != (0, 0)]
                if not nxts:
                    break
                prev, cur = cur, nxts[0]; steps += 1
    # 고립 노드 제거
    G.remove_nodes_from([n for n in list(G.nodes) if G.degree(n) == 0])
    print(f"그래프: 노드 {G.number_of_nodes()}, 엣지 {G.number_of_edges()}, "
          f"연결성분 {nx.number_connected_components(G)}")

    # 5) 저장
    gj = {"res_m": RES, "origin_m": [ox, oy],
          "nodes": [{"id": i, "x": G.nodes[i]["x"], "y": G.nodes[i]["y"]} for i in G.nodes],
          "edges": [{"u": a, "v": b, "length": G[a][b]["length"]} for a, b in G.edges]}
    (HERE / "graph_hospital.json").write_text(json.dumps(gj, ensure_ascii=False, indent=1))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, matplotlib.font_manager as fm
    fp = fm.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc")
    fig, ax = plt.subplots(figsize=(16, 10))
    ext = [ox, ox + W*RES, oy, oy + H*RES]
    ax.imshow(occ, origin="lower", extent=ext, cmap="binary", alpha=0.5)
    for a, b in G.edges:
        ax.plot([G.nodes[a]["x"], G.nodes[b]["x"]], [G.nodes[a]["y"], G.nodes[b]["y"]],
                "b-", lw=1.8, alpha=0.7)
    for i in G.nodes:
        ax.plot(G.nodes[i]["x"], G.nodes[i]["y"], "ro", ms=5)
    ax.set_title(f"병원 위상 지도 — 노드 {G.number_of_nodes()}, 엣지 {G.number_of_edges()}\n"
                 f"빨강=순찰 지점, 파랑=복도 연결 (벽 위 격자)", fontproperties=fp, fontsize=15)
    ax.set_xlabel("x (m)", fontproperties=fp); ax.set_ylabel("y (m)", fontproperties=fp)
    ax.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(HERE / "map_overview.png", dpi=95)
    print("저장: graph_hospital.json, map_overview.png")


if __name__ == "__main__":
    main()
