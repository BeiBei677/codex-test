"""
增量式 SuperPoint 配准 + OpenCV detail 后端（曝光补偿 / 接缝 / 多带融合）
针对大张数（如 62 张 3k×2k）降低接缝与错位。

使用要点：
1) 仍使用 SuperPoint 做特征与单应性估计（增量多参考）。
2) 后端改为 OpenCV detail：
   - 曝光补偿：GAIN_BLOCKS
   - 接缝：GraphCut (color/color_grad)
   - 融合：MultiBandBlender（blend_strength 可调）
3) 可按需求改 warper 类型（默认平面，可自行改为球面/圆柱）。
"""

import argparse
import gc
import math
import os
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch

from superpoint_frontend import SuperPointFrontend


# ===================== 基础工具 =====================
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def get_luma(bgr: np.ndarray) -> np.ndarray:
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    return ycc[..., 0]


def bilateral_denoise(gray: np.ndarray, d=9, sigmaColor=50, sigmaSpace=50) -> np.ndarray:
    return cv2.bilateralFilter(gray, d, sigmaColor, sigmaSpace)


def homomorphic_filter(gray_u8: np.ndarray, sigma=10.0, gain=1.0) -> np.ndarray:
    g = gray_u8.astype(np.float32) / 255.0
    g_log = np.log1p(g)

    h, w = g_log.shape
    uu, vv = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    du = uu - h / 2.0
    dv = vv - w / 2.0
    D2 = du * du + dv * dv
    Hhp = 1.0 - np.exp(-D2 / (2.0 * (sigma * sigma)))

    G = np.fft.fft2(g_log)
    Gshift = np.fft.fftshift(G)
    Fshift = (1.0 + gain) * Gshift * Hhp
    F = np.fft.ifftshift(Fshift)
    f = np.fft.ifft2(F)

    out = np.expm1(np.real(f))
    out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32)


def minmax_normalize(img_float: np.ndarray) -> np.ndarray:
    mn, mx = float(np.min(img_float)), float(np.max(img_float))
    if mx - mn < 1e-6:
        return np.zeros_like(img_float, dtype=np.float32)
    return ((img_float - mn) / (mx - mn)).astype(np.float32)


def preprocess_luma_pipeline(
    bgr: np.ndarray,
    bilateral_params=(9, 50, 50),
    homo_sigma=10.0,
    homo_gain=1.0,
) -> np.ndarray:
    Y = get_luma(bgr)
    d, sc, ss = bilateral_params
    Y_dn = bilateral_denoise(Y, d=d, sigmaColor=sc, sigmaSpace=ss)
    Y_hm = homomorphic_filter(Y_dn, sigma=homo_sigma, gain=homo_gain)
    Y_norm = minmax_normalize(Y_hm)
    return Y_norm


def resize_for_estimation(bgr: np.ndarray, target_short=1400) -> Tuple[np.ndarray, float]:
    h, w = bgr.shape[:2]
    short = min(h, w)
    scale = float(target_short) / float(short)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    new_w = (new_w // 8) * 8
    new_h = (new_h // 8) * 8
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def scale_matrix(s: float) -> np.ndarray:
    return np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float64)


# ===================== 特征与匹配 =====================
def superpoint_features(
    sp_frontend: SuperPointFrontend,
    gray_float: np.ndarray,
    topk=1000,
    conf_thresh=None,
    nms_dist=None,
):
    if conf_thresh is not None:
        sp_frontend.conf_thresh = conf_thresh
    if nms_dist is not None:
        sp_frontend.nms_dist = nms_dist

    corners, desc, _ = sp_frontend.run(gray_float)
    if desc is None or corners is None or corners.shape[1] == 0:
        return [], None

    if topk is not None and corners.shape[1] > topk:
        idx = np.argsort(corners[2, :])[::-1][:topk]
        corners = corners[:, idx]
        desc = desc[:, idx]

    kps = [
        cv2.KeyPoint(float(corners[0, i]), float(corners[1, i]), 3)
        for i in range(corners.shape[1])
    ]
    return kps, desc.T.astype(np.float32)


def bidirectional_knn(desc1, desc2, ratio=0.75):
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    m12 = bf.knnMatch(desc1, desc2, k=2)
    m21 = bf.knnMatch(desc2, desc1, k=2)

    good = []
    for pair in m12:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            if any(
                (rm[0].queryIdx == m.trainIdx and rm[0].trainIdx == m.queryIdx)
                or (len(rm) > 1 and rm[1].queryIdx == m.trainIdx and rm[1].trainIdx == m.queryIdx)
                for rm in m21
            ):
                good.append(m)
    return good


def estimate_homography(kp1, kp2, matches, reproj_thresh=4.0, use_usac=True):
    if len(matches) < 4:
        return None, None
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    method = cv2.USAC_MAGSAC if use_usac and hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC
    H, inliers = cv2.findHomography(src, dst, method, reproj_thresh)
    return H, inliers


def spatially_uniform_matches(kp_src, kp_dst, matches, img_size, grid=(8, 6), per_cell=30):
    if not matches:
        return []
    w, h = img_size
    gx, gy = grid
    cell_w, cell_h = max(w / gx, 1e-6), max(h / gy, 1e-6)

    matches_sorted = sorted(matches, key=lambda m: m.distance)
    buckets = [[[] for _ in range(gx)] for __ in range(gy)]
    for m in matches_sorted:
        x, y = kp_src[m.queryIdx].pt
        cx = min(gx - 1, max(0, int(x / cell_w)))
        cy = min(gy - 1, max(0, int(y / cell_h)))
        if len(buckets[cy][cx]) < per_cell:
            buckets[cy][cx].append(m)
    uniform = [m for row in buckets for cell in row for m in cell]
    return uniform


def count_occupied_cells(kp_src, matches, img_size, grid=(8, 6), inliers_mask=None):
    if not matches:
        return 0
    w, h = img_size
    gx, gy = grid
    cell_w, cell_h = max(w / gx, 1e-6), max(h / gy, 1e-6)
    occ = set()
    keep_flags = [True] * len(matches) if inliers_mask is None else [bool(v) for v in inliers_mask.ravel().tolist()]
    for m, keep in zip(matches, keep_flags):
        if not keep:
            continue
        x, y = kp_src[m.queryIdx].pt
        cx = min(gx - 1, max(0, int(x / cell_w)))
        cy = min(gy - 1, max(0, int(y / cell_h)))
        occ.add((cy, cx))
    return len(occ)


def draw_matches(img1, img2, kp1, kp2, matches, inliers_mask=None, out_path=None):
    """可选保存匹配可视化。"""
    if not matches:
        return
    if inliers_mask is not None:
        good = [m for m, keep in zip(matches, inliers_mask.ravel().tolist()) if keep]
    else:
        good = matches
    vis = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        good,
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(0, 0, 255),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    if out_path:
        cv2.imwrite(out_path, vis)


def lift_homography_to_full(H_small, s_src, s_dst):
    if H_small is None:
        return None
    S_src = scale_matrix(s_src)
    S_dst = scale_matrix(s_dst)
    H_full = np.linalg.inv(S_dst) @ H_small @ S_src
    return H_full


def sift_fallback(img1_small, img2_small, ratio=0.75, reproj_thresh=2.5):
    sift = cv2.SIFT_create()
    g1 = cv2.cvtColor(img1_small, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2_small, cv2.COLOR_BGR2GRAY)
    k1, d1 = sift.detectAndCompute(g1, None)
    k2, d2 = sift.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 4 or len(k2) < 4:
        return None, None, None, None

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    m = flann.knnMatch(d1, d2, k=2)
    good = [p[0] for p in m if len(p) == 2 and p[0].distance < ratio * p[1].distance]
    if len(good) < 4:
        return None, None, None, None

    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    method = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC
    H, inliers = cv2.findHomography(src, dst, method, reproj_thresh)
    return k1, k2, good, H


# ===================== detail 组件：曝光补偿、接缝、融合 =====================
def run_exposure_compensation(images: List[np.ndarray], masks: List[np.ndarray], corners: List[Tuple[int, int]]):
    """在全分辨率 warped 图上做 GAIN_BLOCKS 曝光补偿。"""
    compensator = cv2.detail.ExposureCompensator_createDefault(cv2.detail.ExposureCompensator_GAIN_BLOCKS)
    compensator.feed(corners=corners, images=[img.astype(np.float32) for img in images], masks=masks)

    compensated = []
    for i, (img, mask, corner) in enumerate(zip(images, masks, corners)):
        img_comp = np.zeros_like(img, dtype=np.float32)
        compensator.apply(index=i, corner=corner, image=img.astype(np.float32), mask=mask, image_compensated=img_comp)
        compensated.append(np.clip(img_comp, 0, 255).astype(np.uint8))
    return compensated


def run_seam_finder(images: List[np.ndarray], corners: List[Tuple[int, int]], masks: List[np.ndarray], mode: str = "gc_colorgrad"):
    """GraphCut seam finder，默认 COLOR_GRAD。"""
    if mode == "gc_color":
        finder = cv2.detail_GraphCutSeamFinder("COST_COLOR")
    else:
        finder = cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
    imgs_float = [img.astype(np.float32) for img in images]
    finder.find(imgs_float, corners, masks)
    return masks


def multiband_blend(images: List[np.ndarray], masks: List[np.ndarray], corners: List[Tuple[int, int]], blend_strength: int = 5):
    """多带融合：根据输出 ROI 自动设置 band 数。"""
    sizes = [(img.shape[1], img.shape[0]) for img in images]
    dst_roi = cv2.detail.resultRoi(corners=corners, sizes=sizes)
    blend_width = np.sqrt(dst_roi[2] * dst_roi[3]) * blend_strength / 100
    if blend_width < 1:
        blender = cv2.detail.Blender_createDefault(cv2.detail.Blender_NO)
    else:
        blender = cv2.detail_MultiBandBlender()
        num_bands = max(1, int((np.log(blend_width) / np.log(2.0) - 1.0)))
        blender.setNumBands(num_bands)
    blender.prepare(dst_roi)

    for img, mask, corner in zip(images, masks, corners):
        blender.feed(cv2.UMat(img.astype(np.int16)), mask, corner)
    result, result_mask = blender.blend(None, None)
    result = cv2.convertScaleAbs(result)
    return result, result_mask


# ===================== 主流程 =====================
def incremental_panorama(
    image_paths: Sequence[str],
    weights_path: str,
    out_dir: str,
    target_short=1800,  # 提高估计分辨率，提升覆盖
    bilateral_params=(9, 50, 50),
    homo_sigma=10.0,
    homo_gain=1.0,
    sp_conf=0.015,
    sp_nms=3,
    ratio=0.7,
    reproj_thresh=2.5,
    use_usac=True,
    topk=4000,
    use_gpu=True,
    save_pair_matches=False,
    max_width=12000,
    max_height=8000,
    max_megapixels=100,
    # 新增：控制曝光/接缝/融合阶段的工作分辨率，防止内存炸裂
    work_megapixels=20,
    min_sp_matches=150,
    min_inliers=120,
    min_inlier_ratio=0.30,
    area_growth_limit=8.0,
    max_center_shift_y_ratio=0.6,
    keep_simple_canvas=True,
    ref_search=8,
    uniform_grid=(12, 10),
    uniform_per_cell=25,
    seam_mode="gc_colorgrad",
    blend_strength=6,
):
    def local_scale_matrix(s: float) -> np.ndarray:
        return np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float64)

    def local_translation_matrix(offset_x: float, offset_y: float) -> np.ndarray:
        return np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float64)

    def safe_normalize_h(H: np.ndarray):
        if H is None:
            return None
        c = H[2, 2]
        return H if abs(c) < 1e-12 else (H / c)

    def is_homography_degenerate(H: np.ndarray) -> bool:
        if H is None or not np.isfinite(H).all() or abs(H[2, 2]) < 1e-12:
            return True
        Hn = safe_normalize_h(H)
        s = np.linalg.svd(Hn[:2, :2], compute_uv=False)
        cond = s[0] / max(s[-1], 1e-12)
        return cond > 1e6

    def bbox_area_after_warp(w, h, H):
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(corners, H)
        x = warped[:, 0, 0]
        y = warped[:, 0, 1]
        return max(0.0, (np.max(x) - np.min(x)) * (np.max(y) - np.min(y)))

    def center_after_warp(w, h, H):
        c = np.float32([[w * 0.5, h * 0.5]]).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(c, H)
        return float(warped[0, 0, 0]), float(warped[0, 0, 1])

    def compute_canvas_size_and_scale(all_corners, max_width, max_height, max_megapixels):
        xs, ys = [], []
        for c in all_corners:
            a = c.reshape(-1, 2)
            a = a[np.isfinite(a).all(axis=1)]
            if a.size == 0:
                continue
            xs.append(a[:, 0])
            ys.append(a[:, 1])
        if not xs or not ys:
            raise RuntimeError("所有角点都无效，无法计算画布尺寸")

        x_min = math.floor(np.min(np.concatenate(xs)))
        y_min = math.floor(np.min(np.concatenate(ys)))
        x_max = math.ceil(np.max(np.concatenate(xs)))
        y_max = math.ceil(np.max(np.concatenate(ys)))

        offset_x = -min(0, x_min)
        offset_y = -min(0, y_min)
        canvas_width = int(x_max - x_min)
        canvas_height = int(y_max - y_min)
        print(f"\n原始画布尺寸: {canvas_width} x {canvas_height}, 偏移: ({offset_x}, {offset_y})")

        max_pixels = int(max_megapixels * 1e6)
        s_out = 1.0
        need_scale = (canvas_width * canvas_height > max_pixels) or (canvas_width > max_width) or (canvas_height > max_height)
        if need_scale:
            s_w = max_width / max(canvas_width, 1)
            s_h = max_height / max(canvas_height, 1)
            s_p = math.sqrt(max_pixels / max(canvas_width * canvas_height, 1))
            s_out = max(min(s_w, s_h, s_p), 1e-3)
            print(f"画布过大，应用全局输出缩放 s_out={s_out:.6f}")
        return canvas_width, canvas_height, offset_x, offset_y, s_out

    ensure_dir(out_dir)
    n = len(image_paths)
    if n < 2:
        raise RuntimeError("至少需要2张图像进行拼接")
    print(f"批量拼接: 共 {n} 张图")

    device_cuda = torch.cuda.is_available() and use_gpu
    sp = SuperPointFrontend(
        weights_path,
        nms_dist=sp_nms,
        conf_thresh=sp_conf,
        nn_thresh=0.7,
        cuda=device_cuda,
    )

    used_indices = [0]
    H_to_ref = {0: np.eye(3, dtype=np.float64)}
    shapes = {}

    img0 = cv2.imread(image_paths[0])
    if img0 is None:
        raise RuntimeError(f"读取图像失败: {image_paths[0]}")
    h0, w0 = img0.shape[:2]
    shapes[0] = (h0, w0)
    small_prev, s_prev = resize_for_estimation(img0, target_short=target_short)
    y_prev = preprocess_luma_pipeline(small_prev, bilateral_params=bilateral_params, homo_sigma=homo_sigma, homo_gain=homo_gain)
    kp_prev, d_prev = superpoint_features(sp, y_prev, topk=topk, conf_thresh=sp_conf, nms_dist=sp_nms)

    small_cache = {0: small_prev}
    scale_cache = {0: s_prev}
    sp_cache = {0: (kp_prev, d_prev)}

    del img0, y_prev
    gc.collect()

    skipped = []
    for i in range(1, n):
        print(f"\n估计配准: 图 {i} → 多参考")

        img_i = cv2.imread(image_paths[i])
        if img_i is None:
            print(f"读取失败，跳过: {image_paths[i]}")
            skipped.append(i)
            continue
        hi, wi = img_i.shape[:2]
        shapes[i] = (hi, wi)

        small_cur, s_cur = resize_for_estimation(img_i, target_short=target_short)
        y_cur = preprocess_luma_pipeline(
            small_cur, bilateral_params=bilateral_params, homo_sigma=homo_sigma, homo_gain=homo_gain
        )
        kp_cur, d_cur = superpoint_features(sp, y_cur, topk=topk, conf_thresh=sp_conf, nms_dist=sp_nms)
        del y_cur, img_i
        gc.collect()

        cand_refs = used_indices[-min(len(used_indices), ref_search) :]
        best = {"score": -1, "ref_idx": None, "H_small": None, "method": None, "matches": None, "inliers": None}

        if d_cur is not None and kp_cur is not None and len(kp_cur) >= 4:
            for ref_idx in reversed(cand_refs):
                kp_ref, d_ref = sp_cache.get(ref_idx, (None, None))
                if d_ref is None or kp_ref is None or len(kp_ref) < 4:
                    continue

                raw_matches = bidirectional_knn(d_cur, d_ref, ratio=ratio)
                uniform_matches = spatially_uniform_matches(
                    kp_cur,
                    kp_ref,
                    raw_matches,
                    img_size=(small_cur.shape[1], small_cur.shape[0]),
                    grid=uniform_grid,
                    per_cell=uniform_per_cell,
                )
                if len(uniform_matches) < min_sp_matches:
                    continue

                H_sm, inliers = estimate_homography(kp_cur, kp_ref, uniform_matches, reproj_thresh=reproj_thresh, use_usac=use_usac)
                if H_sm is None or inliers is None:
                    continue
                inl_cnt = int(np.sum(inliers))
                inl_ratio = inl_cnt / max(len(uniform_matches), 1)
                if inl_cnt < min_inliers or inl_ratio < min_inlier_ratio:
                    continue

                coverage = count_occupied_cells(
                    kp_cur,
                    uniform_matches,
                    img_size=(small_cur.shape[1], small_cur.shape[0]),
                    grid=uniform_grid,
                    inliers_mask=inliers,
                )
                score = inl_cnt + 0.2 * coverage
                if score > best["score"]:
                    best.update(
                        {"score": score, "ref_idx": ref_idx, "H_small": H_sm, "method": "SP", "matches": uniform_matches, "inliers": inliers}
                    )

        if best["H_small"] is None:
            print("SP不达标或失败，使用SIFT兜底...")
            for ref_idx in reversed(cand_refs):
                small_ref = small_cache[ref_idx]
                k1, k2, good, H_sift = sift_fallback(small_cur, small_ref, ratio=ratio, reproj_thresh=reproj_thresh)
                if H_sift is None or good is None:
                    continue
                if len(good) >= min_inliers:
                    best.update({"score": len(good), "ref_idx": ref_idx, "H_small": H_sift, "method": "SIFT", "matches": good, "inliers": None})
                    break

        if best["H_small"] is None or best["ref_idx"] is None:
            print(f"所有参考均失败，跳过图 {i}")
            skipped.append(i)
            continue

        chosen_ref = best["ref_idx"]
        H_full_i_to_ref = lift_homography_to_full(best["H_small"], s_src=s_cur, s_dst=scale_cache[chosen_ref])
        H_full_i_to_ref = safe_normalize_h(H_full_i_to_ref)
        if is_homography_degenerate(H_full_i_to_ref):
            print(f"退化单应性，跳过图 {i}")
            skipped.append(i)
            continue

        try:
            ref_w, ref_h = shapes[chosen_ref][1], shapes[chosen_ref][0]
            cur_w, cur_h = wi, hi
            area_ref = bbox_area_after_warp(ref_w, ref_h, H_to_ref[chosen_ref])
            area_cur = bbox_area_after_warp(cur_w, cur_h, H_to_ref[chosen_ref] @ H_full_i_to_ref)
            growth = (area_cur / area_ref) if area_ref > 0 else 1.0
            cx_ref, cy_ref = center_after_warp(ref_w, ref_h, H_to_ref[chosen_ref])
            cx_cur, cy_cur = center_after_warp(cur_w, cur_h, H_to_ref[chosen_ref] @ H_full_i_to_ref)
            shift_y = abs(cy_cur - cy_ref)
            limit_shift = max_center_shift_y_ratio * ref_h
            print(f"[{best['method']} 选参] 参考={chosen_ref}, 面积增长 x{growth:.2f}, 垂直漂移 {shift_y:.1f} (limit {limit_shift:.1f})")
            if growth > area_growth_limit or shift_y > limit_shift:
                print(f"异常增长/漂移，跳过图 {i}")
                skipped.append(i)
                continue
        except Exception:
            pass

        H_to_ref[i] = safe_normalize_h(H_to_ref[chosen_ref] @ H_full_i_to_ref)
        used_indices.append(i)

        small_cache[i] = small_cur
        scale_cache[i] = s_cur
        sp_cache[i] = (kp_cur, d_cur)

        if save_pair_matches and best["matches"] is not None and best["method"] == "SP":
            try:
                kp_ref, _ = sp_cache[chosen_ref]
                draw_path = os.path.join(out_dir, f"pair_{i}_best_{best['method']}.jpg")
                draw_matches(small_cur, small_cache[chosen_ref], kp_cur, kp_ref, best["matches"], best["inliers"], out_path=draw_path)
            except Exception:
                pass
        gc.collect()

    print(f"\n有效拼接的图像数: {len(used_indices)} / {n}")
    if skipped:
        print(f"跳过的索引: {skipped}")
    if len(used_indices) < 2:
        img_only = cv2.imread(image_paths[used_indices[0]])
        out_path = os.path.join(out_dir, "panorama_poisson.jpg")
        cv2.imwrite(out_path, img_only)
        return img_only, None

    all_corners = []
    for idx in used_indices:
        h, w = shapes[idx]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners, H_to_ref[idx])
        all_corners.append(warped_corners)

    canvas_width_raw, canvas_height_raw, offset_x, offset_y, s_out = compute_canvas_size_and_scale(
        all_corners, max_width, max_height, max_megapixels
    )

    S_out = local_scale_matrix(s_out)
    canvas_width = max(1, int(round(canvas_width_raw * s_out)))
    canvas_height = max(1, int(round(canvas_height_raw * s_out)))
    print(f"最终画布尺寸: {canvas_width} x {canvas_height}, 偏移: ({offset_x}, {offset_y}), s_out={s_out:.6f}")

    # 额外缩放：用于曝光补偿/接缝/融合阶段，降低内存占用
    s_work = min(
        1.0,
        math.sqrt(max(work_megapixels, 1e-3) * 1e6 / max(canvas_width * canvas_height, 1e-9)),
    )
    if s_work < 1.0:
        print(f"为防止内存不足，工作分辨率缩放 s_work={s_work:.6f} (工作约 {work_megapixels} Mpx)")

    T = local_translation_matrix(offset_x, offset_y)

    # ========== Warp 所有有效图 ==========
    warped_imgs = []
    warped_masks = []
    corners = []
    sizes = []
    simple_canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8) if keep_simple_canvas else None

    for k, idx in enumerate(used_indices):
        if (k % 5) == 0:
            print(f"[Warp] 进度 {k}/{len(used_indices)} (idx={idx})")
        img = cv2.imread(image_paths[idx])
        if img is None:
            continue
        H_final = S_out @ T @ H_to_ref[idx]
        warped = cv2.warpPerspective(img, H_final, (canvas_width, canvas_height))
        gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_warped, 5, 255, cv2.THRESH_BINARY)

        if keep_simple_canvas and simple_canvas is not None:
            idxs = np.where(mask > 0)
            simple_canvas[idxs] = warped[idxs]

        # 如需降采样到工作分辨率，统一缩放图和掩膜
        if s_work < 1.0:
            warped = cv2.resize(
                warped,
                (max(1, int(warped.shape[1] * s_work)), max(1, int(warped.shape[0] * s_work))),
                interpolation=cv2.INTER_AREA,
            )
            mask = cv2.resize(
                mask,
                (warped.shape[1], warped.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        warped_imgs.append(warped)
        warped_masks.append(mask)
        corners.append((0, 0))  # 已经放到全局画布
        sizes.append((warped.shape[1], warped.shape[0]))

        del warped, gray_warped, img
        gc.collect()

    # ========== detail 后端：曝光补偿 + 接缝 + 多带 ==========
    print("运行曝光补偿 (GAIN_BLOCKS)...")
    warped_imgs = run_exposure_compensation(warped_imgs, warped_masks, corners)

    print(f"运行接缝查找 ({seam_mode})...")
    warped_masks = run_seam_finder(warped_imgs, corners, warped_masks, mode=seam_mode)

    print(f"运行多带融合 (blend_strength={blend_strength})...")
    blended, blended_mask = multiband_blend(warped_imgs, warped_masks, corners, blend_strength=blend_strength)

    out_blended = os.path.join(out_dir, "panorama_multiband.jpg")
    cv2.imwrite(out_blended, blended)
    if keep_simple_canvas and simple_canvas is not None:
        cv2.imwrite(os.path.join(out_dir, "panorama_simple.jpg"), simple_canvas)

    print("\n结果保存完成：")
    if keep_simple_canvas and simple_canvas is not None:
        print(f"- 简单叠加全景图: {os.path.join(out_dir, 'panorama_simple.jpg')}")
    print(f"- 多带融合全景图: {out_blended}")

    return blended, (simple_canvas if keep_simple_canvas else None)


# ===================== CLI =====================
def collect_image_paths(img_dir, extensions=(".jpg", ".jpeg", ".png", ".bmp"), max_images=None):
    files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if os.path.splitext(f.lower())[1] in extensions]
    files.sort()
    if max_images is not None:
        files = files[:max_images]
    return files


def main():
    parser = argparse.ArgumentParser(description="增量式 SuperPoint + OpenCV detail 融合全景拼接")
    parser.add_argument("--img_dir", type=str, default="images", help="图像目录（默认：images）")
    parser.add_argument("--weights", type=str, default="superpoint_v1.pth", help="SuperPoint 权重路径")
    parser.add_argument("--out", type=str, default="outputs", help="输出目录（默认：outputs）")
    parser.add_argument("--short", type=int, default=1800, help="缩小图短边（特征估计）")
    parser.add_argument("--use_gpu", action="store_true", help="使用 GPU 加速 SuperPoint")
    parser.add_argument("--max_images", type=int, default=62, help="最多处理图像数量")
    parser.add_argument("--save_pair_matches", action="store_true", help="保存匹配可视化")
    parser.add_argument("--sp_conf", type=float, default=0.015, help="SuperPoint 置信度阈值")
    parser.add_argument("--sp_nms", type=int, default=3, help="SuperPoint NMS 距离")
    parser.add_argument("--ratio", type=float, default=0.7, help="匹配 Ratio Test 阈值")
    parser.add_argument("--reproj", type=float, default=2.5, help="重投影误差阈值")
    parser.add_argument("--topk", type=int, default=4000, help="SuperPoint 特征点 TopK")
    parser.add_argument("--blend_strength", type=int, default=6, help="多带融合强度 [0,100]")
    parser.add_argument("--seam_mode", type=str, default="gc_colorgrad", choices=["gc_color", "gc_colorgrad"], help="接缝模式")

    args = parser.parse_args()
    ensure_dir(args.out)
    image_paths = collect_image_paths(args.img_dir, max_images=args.max_images)
    if len(image_paths) < 2:
        raise RuntimeError("至少需要2张图像进行拼接")
    print(f"找到 {len(image_paths)} 张图像，开始拼接...")

    incremental_panorama(
        image_paths=image_paths,
        weights_path=args.weights,
        out_dir=args.out,
        target_short=args.short,
        sp_conf=args.sp_conf,
        sp_nms=args.sp_nms,
        ratio=args.ratio,
        reproj_thresh=args.reproj,
        topk=args.topk,
        use_gpu=args.use_gpu,
        save_pair_matches=args.save_pair_matches,
        blend_strength=args.blend_strength,
        seam_mode=args.seam_mode,
    )


if __name__ == "__main__":
    # 你可以直接运行 main()，也可以按下方示例手动配置：
    # 示例：手动配置路径与参数（保持注释，按需复制到脚本或交互运行）
    # img_dir = "/mnt/d/software/detectron2_project/Test detectron2/images/yongzhou-small"
    # weights_path = "superpoint_v1.pth"
    # out_dir = "/mnt/d/software/detectron2_project/Test detectron2/outputs"
    # target_short = 1400
    # use_gpu = True
    #
    # ensure_dir(out_dir)
    # image_paths = collect_image_paths(img_dir, max_images=62)
    # print(f"找到 {len(image_paths)} 张图像，按文件名排序")
    #
    # incremental_panorama(
    #     image_paths=image_paths,
    #     weights_path=weights_path,
    #     out_dir=out_dir,
    #     target_short=target_short,
    #     use_gpu=use_gpu,
    #     sp_conf=0.015,
    #     sp_nms=3,
    #     ratio=0.7,
    #     reproj_thresh=2.5,
    #     topk=3500,
    #     blend_strength=6,
    #     seam_mode="gc_colorgrad",
    #     max_width=14000,
    #     max_height=10000,
    #     max_megapixels=200,
    #     work_megapixels=20,
    # )
    #
    # print(f"\n拼接完成！结果保存在：{out_dir}")
    # print(f"- 多带融合全景图: {os.path.join(out_dir, 'panorama_multiband.jpg')}")

    main()
