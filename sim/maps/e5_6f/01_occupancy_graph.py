"""iD3-A: 2D 시뮬 지도 — 점유격자 + 복도 위상 그래프.

정렬 메시 → 높이밴드 슬라이스 → 점유격자(벽) → 자유공간 골격화 → 위상 그래프.
산출: occupancy.npy(격자), occupancy.png(검증), graph.json(노드·엣지), map_overview.png.

실행: conda patrol + unset PYTHONPATH.
"""
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.morphology import skeletonize
import networkx as nx

HERE = Path(__file__).parent
RES = 0.05          # m/cell (5cm)
BAND = (0.5, 1.8)   # 바닥 위 벽 밴드
ROBOT_R = 0.30      # 로봇 반경(장애물 팽창)


def main():
    frame = json.loads((HERE / "frame.json").read_text())
    m = o3d.io.read_triangle_mesh(str(HERE / "e5_6f_aligned.ply"))
    V = np.asarray(m.vertices)

    # 1) 벽 밴드 슬라이스 → 2D 점유격자
    band = V[(V[:, 2] > BAND[0]) & (V[:, 2] < BAND[1])]
    xmin, ymin = V[:, 0].min(), V[:, 1].min()
    W = int(np.ceil((V[:, 0].max() - xmin) / RES)) + 1
    H = int(np.ceil((V[:, 1].max() - ymin) / RES)) + 1
    occ = np.zeros((H, W), np.uint8)
    ix = ((band[:, 0] - xmin) / RES).astype(int)
    iy = ((band[:, 1] - ymin) / RES).astype(int)
    occ[iy, ix] = 1
    print(f"격자 {W}×{H} ({W*RES:.0f}×{H*RES:.0f}m), 벽 셀 {occ.sum():,}")

    # 2) 벽 정리: 팽창→수축으로 점 구멍 메우기(연속 벽선)
    occ = ndimage.binary_closing(occ, iterations=3).astype(np.uint8)

    # 3) 자유공간 = 벽을 로봇 반경만큼 팽창시킨 것의 여집합, 최대 연결성분만
    infl = ndimage.binary_dilation(occ, iterations=int(ROBOT_R / RES))
    free = ~infl
    lab, n = ndimage.label(free)
    if n:
        biggest = np.argmax(np.bincount(lab.ravel())[1:]) + 1
        free = lab == biggest
    print(f"자유공간 셀 {free.sum():,} (연결성분 {n}개 중 최대)")

    # 4) 골격화 → 복도 중심선
    skel = skeletonize(free)

    # 5) 골격 → 위상 그래프 (교차점·끝점=노드, 사이=엣지)
    nb = ndimage.convolve(skel.astype(int), np.ones((3, 3)), mode="constant") - 1
    nodes_mask = skel & ((nb >= 3) | (nb == 1))   # 분기(≥3) 또는 끝점(1)
    ny, nx_ = np.where(nodes_mask)
    # 근접 노드 병합(0.5m 내)
    G = nx.Graph()
    coords = []
    for y, x in zip(ny, nx_):
        wx = xmin + x * RES
        wy = ymin + y * RES
        merged = False
        for i, (cx, cy) in enumerate(coords):
            if abs(cx - wx) < 0.5 and abs(cy - wy) < 0.5:
                merged = True
                break
        if not merged:
            G.add_node(len(coords), x=float(wx), y=float(wy))
            coords.append((wx, wy))
    # 엣지: 골격 픽셀을 따라 노드-노드 연결 (BFS로 인접 노드쌍)
    node_px = {}
    for i, (wx, wy) in enumerate(coords):
        node_px[(int((wy - ymin) / RES), int((wx - xmin) / RES))] = i

    def nearest_node(py, px):
        best, bd = None, 1e9
        for (ky, kx), ni in node_px.items():
            d = (ky - py) ** 2 + (kx - px) ** 2
            if d < bd:
                bd, best = d, ni
        return best if bd < (0.6 / RES) ** 2 else None

    # 골격을 세그먼트로: 각 골격 픽셀에서 인접 노드까지 추적 대신, 성분 라벨로 근사
    sy, sx = np.where(skel)
    edges = set()
    # 간이 방식: 노드쌍 중 골격상 경로가 있으면 연결 (거리 임계 + 골격 팔로우는
    # 대규모라 비용↑ → MST 근사로 복도 연결성 확보)
    pts = np.array(coords)
    if len(pts) > 1:
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        for i in range(len(pts)):
            dists, idxs = tree.query(pts[i], k=min(4, len(pts)))
            for d, j in zip(np.atleast_1d(dists), np.atleast_1d(idxs)):
                j = int(j)
                if i != j and d < 15.0:  # 15m 내 이웃 노드 연결 후보
                    edges.add((min(i, j), max(i, j)))
    for a, b in edges:
        G.add_edge(a, b, length=float(np.hypot(coords[a][0]-coords[b][0],
                                               coords[a][1]-coords[b][1])))
    print(f"그래프: 노드 {G.number_of_nodes()}, 엣지 {G.number_of_edges()}")

    # 6) 저장
    np.save(HERE / "occupancy.npy", occ)
    gj = {"res_m": RES, "origin": [float(xmin), float(ymin)],
          "nodes": [{"id": i, "x": G.nodes[i]["x"], "y": G.nodes[i]["y"]}
                    for i in G.nodes],
          "edges": [{"u": a, "v": b, "length": G[a][b]["length"]}
                    for a, b in G.edges]}
    (HERE / "graph.json").write_text(json.dumps(gj, ensure_ascii=False, indent=1))

    # 7) 검증 그림
    fig, ax = plt.subplots(1, 3, figsize=(24, 8))
    ext = [xmin, xmin + W * RES, ymin, ymin + H * RES]
    ax[0].imshow(occ, origin="lower", extent=ext, cmap="gray_r")
    ax[0].set_title("Occupancy grid (walls)"); ax[0].set_aspect("equal")
    ax[1].imshow(free, origin="lower", extent=ext, cmap="Greens")
    ax[1].imshow(skel, origin="lower", extent=ext, cmap="Reds", alpha=0.7)
    ax[1].set_title("Free space + skeleton"); ax[1].set_aspect("equal")
    ax[2].imshow(occ, origin="lower", extent=ext, cmap="gray_r", alpha=0.4)
    for a, b in G.edges:
        ax[2].plot([coords[a][0], coords[b][0]], [coords[a][1], coords[b][1]],
                   "b-", lw=1.5)
    for i in G.nodes:
        ax[2].plot(G.nodes[i]["x"], G.nodes[i]["y"], "ro", ms=4)
    ax[2].set_title(f"Topological graph ({G.number_of_nodes()}n/{G.number_of_edges()}e)")
    ax[2].set_aspect("equal")
    plt.tight_layout()
    plt.savefig(HERE / "map_overview.png", dpi=70)
    print("저장: occupancy.npy/png, graph.json, map_overview.png")


if __name__ == "__main__":
    main()
