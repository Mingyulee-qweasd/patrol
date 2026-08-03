"""iD3-0: E5 6층 메시 공유 기반 — 정렬·청소·좌표 프레임.

전체 메시 로드 → 바닥 검출 → PCA 수평 정렬(복도축=x) → 노이즈 제거 →
정렬 메시(.ply) + 좌표 프레임(frame.json) 저장. 트랙 A(2D 지도)·B(Isaac) 공통 원재료.

실행: conda patrol + unset PYTHONPATH. 관례상 스캔 단위=미터.
"""
import json
from pathlib import Path

import numpy as np
import open3d as o3d

HERE = Path(__file__).parent
RAW = HERE / "e5_6f_raw.ply"


def main():
    print(f"로드: {RAW} ...")
    m = o3d.io.read_triangle_mesh(str(RAW))
    V = np.asarray(m.vertices)
    print(f"  정점 {len(V):,} / 면 {len(np.asarray(m.triangles)):,}")

    # 1) 바닥 Z 검출: 하위 구간 최빈 (바닥 평면이 가장 조밀)
    zh, ze = np.histogram(V[:, 2], bins=120)
    floor_z = float(ze[np.argmax(zh[:40])])
    print(f"  바닥 Z ≈ {floor_z:.2f}")

    # 2) 벽 밴드(바닥+0.3~1.8m)로 수평 주축 찾기 → PCA 회전
    band = (V[:, 2] > floor_z + 0.3) & (V[:, 2] < floor_z + 1.8)
    XY = V[band, :2]
    XY0 = XY - XY.mean(0)
    _, evecs = np.linalg.eigh(np.cov(XY0.T))
    R2 = evecs[:, ::-1]            # 큰 고유값 축 = 긴 복도 → x축
    if np.linalg.det(R2) < 0:      # 반사 방지(우수 좌표계 유지)
        R2[:, 1] *= -1
    R = np.eye(3)
    R[:2, :2] = R2.T

    # 3) 정렬 적용: 회전 후, 바닥을 z=0, 수평 중심을 원점으로
    Vc = V.copy()
    Vc[:, 2] -= floor_z
    Vc = Vc @ R.T
    band_a = Vc[band]
    # 1~99 퍼센타일로 노이즈 제외한 실제 규모
    lo = np.percentile(band_a[:, :2], 1, axis=0)
    hi = np.percentile(band_a[:, :2], 99, axis=0)
    ctr = (lo + hi) / 2
    Vc[:, 0] -= ctr[0]
    Vc[:, 1] -= ctr[1]
    span = hi - lo
    print(f"  정렬 후 규모: 복도축 {span[0]:.1f}m × 폭 {span[1]:.1f}m")

    # 4) 노이즈 제거: 실제 건물 범위 밖(퍼센타일 여유 밖) 정점 마스크로 면 필터
    m.vertices = o3d.utility.Vector3dVector(Vc)
    keep = (
        (Vc[:, 0] > -span[0] / 2 - 2) & (Vc[:, 0] < span[0] / 2 + 2)
        & (Vc[:, 1] > -span[1] / 2 - 2) & (Vc[:, 1] < span[1] / 2 + 2)
        & (Vc[:, 2] > -0.5) & (Vc[:, 2] < 6.0)
    )
    m.remove_vertices_by_mask(~keep)
    m.remove_unreferenced_vertices()
    print(f"  청소 후 정점 {len(np.asarray(m.vertices)):,} / 면 {len(np.asarray(m.triangles)):,}")

    # 5) 저장
    out_ply = HERE / "e5_6f_aligned.ply"
    o3d.io.write_triangle_mesh(str(out_ply), m)
    Vf = np.asarray(m.vertices)
    frame = {
        "source": RAW.name,
        "unit": "meter (가정 — 층고·복도길이가 상식적)",
        "floor_z_original": floor_z,
        "rotation_deg": float(np.degrees(np.arctan2(R2[1, 0], R2[0, 0]))),
        "aligned_bounds": {
            "x": [float(Vf[:, 0].min()), float(Vf[:, 0].max())],
            "y": [float(Vf[:, 1].min()), float(Vf[:, 1].max())],
            "z": [float(Vf[:, 2].min()), float(Vf[:, 2].max())],
        },
        "corridor_len_m": float(span[0]),
        "width_m": float(span[1]),
        "convention": "x=긴 복도축, y=폭, z=높이(바닥 0). 원점=바닥 평면 중심.",
    }
    (HERE / "frame.json").write_text(json.dumps(frame, ensure_ascii=False, indent=2))
    print(f"저장: {out_ply.name}, frame.json")
    return frame


if __name__ == "__main__":
    main()
