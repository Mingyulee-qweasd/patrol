"""병원 USD → 2D 점유격자. 삼각형-평면 슬라이싱(CAD용 정석).

CAD는 정점이 모서리에만 있어 정점 래스터화로는 벽이 비어 나옴 → 각 삼각형이
높이 평면 z=Z0을 가로지르면 그 교차 선분을 2D 격자에 그림(윤곽 슬라이스).
usd-core, GUI 불필요. 실행: conda patrol.
"""
import sys
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom
from skimage.draw import line as skline
from scipy import ndimage

USD = sys.argv[1] if len(sys.argv) > 1 else \
    "/tmp/claude-1000/-home-aprl/4ae66b54-1fb0-4f07-a64d-69ee74271a6d/scratchpad/hospital.usd"
OUT = Path("/home/aprl/patrol/sim/maps/hospital"); OUT.mkdir(parents=True, exist_ok=True)
RES = 0.05
Z0 = 1.0    # 슬라이스 높이(바닥 위 1m)


def tris_of(mesh, M):
    pts = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    idx = mesh.GetFaceVertexIndicesAttr().Get()
    if not pts or not counts or not idx:
        return None
    P = np.array([[p[0], p[1], p[2], 1.0] for p in pts]) @ M   # 월드
    P = P[:, :3]
    counts = np.array(counts); idx = np.array(idx)
    tris = []
    off = 0
    for c in counts:
        f = idx[off:off+c]; off += c
        for k in range(1, c-1):           # 팬 삼각화
            tris.append((f[0], f[k], f[k+1]))
    return P, np.array(tris)


def main():
    stage = Usd.Stage.Open(USD)
    print("스테이지 열림. 삼각형 슬라이싱...")
    segs = []   # (x0,y0,x1,y1) 월드 선분
    zsamp = []
    n_mesh = 0
    for prim in stage.Traverse(Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        M = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default())).reshape(4, 4)
        r = tris_of(UsdGeom.Mesh(prim), M)
        if r is None:
            continue
        P, tris = r
        zsamp.append(P[:, 2])
        n_mesh += 1
    allz = np.concatenate(zsamp)
    floor = float(np.percentile(allz, 2))   # 최솟값 아닌 2퍼센타일(이상치 배제)
    zslice = floor + Z0
    print(f"메시 {n_mesh}, 바닥≈{floor:.2f} (2%ile), 슬라이스 z={zslice:.2f}")

    # 두 번째 순회로 실제 슬라이스 (메모리 절약)
    xs, ys = [], []
    for prim in stage.Traverse(Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        M = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default())).reshape(4, 4)
        r = tris_of(UsdGeom.Mesh(prim), M)
        if r is None:
            continue
        P, tris = r
        for a, b, c in tris:
            v = P[[a, b, c]]
            z = v[:, 2]
            if z.min() >= zslice or z.max() <= zslice:
                continue                     # 평면 안 가로지름
            # z=zslice 가로지르는 두 모서리의 교점
            cross = []
            for i, j in [(0, 1), (1, 2), (2, 0)]:
                zi, zj = z[i], z[j]
                if (zi - zslice) * (zj - zslice) < 0:
                    t = (zslice - zi) / (zj - zi)
                    cross.append(v[i] + t * (v[j] - v[i]))
            if len(cross) == 2:
                segs.append((cross[0][0], cross[0][1], cross[1][0], cross[1][1]))
    segs = np.array(segs)
    print(f"슬라이스 선분 {len(segs):,}개")

    xmin, ymin = segs[:, [0, 2]].min(), segs[:, [1, 3]].min()
    xmax, ymax = segs[:, [0, 2]].max(), segs[:, [1, 3]].max()
    W = int((xmax - xmin) / RES) + 2; H = int((ymax - ymin) / RES) + 2
    occ = np.zeros((H, W), np.uint8)
    for x0, y0, x1, y1 in segs:
        rr, cc = skline(int((y0-ymin)/RES), int((x0-xmin)/RES),
                        int((y1-ymin)/RES), int((x1-xmin)/RES))
        rr = np.clip(rr, 0, H-1); cc = np.clip(cc, 0, W-1)
        occ[rr, cc] = 1
    occ = ndimage.binary_closing(occ, iterations=1).astype(np.uint8)
    np.save(OUT / "occ_hospital.npy", occ)
    meta = {"res_m": RES, "origin_m": [float(xmin), float(ymin)],
            "size_m": [float(xmax-xmin), float(ymax-ymin)]}
    import json; (OUT / "hospital_meta.json").write_text(json.dumps(meta, indent=1))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, matplotlib.font_manager as fm
    fp = fm.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc")
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.imshow(occ, origin="lower", extent=[0, W*RES, 0, H*RES], cmap="binary")
    ax.set_title(f"병원 2D 점유격자 (삼각형 슬라이스 z={Z0}m) — {W*RES:.0f}x{H*RES:.0f}m\n"
                 f"검정=벽/구조, 흰=이동가능", fontproperties=fp, fontsize=15)
    ax.set_xlabel("x (m)", fontproperties=fp); ax.set_ylabel("y (m)", fontproperties=fp)
    ax.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(OUT / "hospital_occupancy.png", dpi=95)
    print(f"저장: occ_hospital.npy, hospital_occupancy.png ({W}x{H}, 벽 {occ.sum()/(W*H)*100:.1f}%)")


if __name__ == "__main__":
    main()
