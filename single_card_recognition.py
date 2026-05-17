from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

CARD_WIDTH = 200
CARD_HEIGHT = 300
CORNER_ROI = (0, 0, 29, 150)
MATCH_CANVAS = 64
MATCH_PADDING = 6
TM_METHOD_WEIGHTS = (
    (cv2.TM_CCOEFF_NORMED, 0.45),
    (cv2.TM_CCORR_NORMED, 0.35),
    (cv2.TM_SQDIFF_NORMED, 0.20),
)


"""
4. Rendezi a négyszög sarkait: bal felső, jobb felső, jobb alsó, bal alsó.
Kezeli a forgatott kártyákat és biztosítja a portré tájolást!
"""
def order_points(points: np.ndarray) -> np.ndarray:
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    pts = points[np.argsort(angles)]
    
    dists = [np.linalg.norm(pts[i] - pts[(i+1)%4]) for i in range(4)]
    
    if dists[0] + dists[2] < dists[1] + dists[3]:
        short_edges = [(0, 1), (2, 3)]
    else:
        short_edges = [(1, 2), (3, 0)]
        
    avg_y1 = (pts[short_edges[0][0]][1] + pts[short_edges[0][1]][1]) / 2.0
    avg_y2 = (pts[short_edges[1][0]][1] + pts[short_edges[1][1]][1]) / 2.0
    
    top_edge = short_edges[0] if avg_y1 < avg_y2 else short_edges[1]
    
    p1, p2 = pts[top_edge[0]], pts[top_edge[1]]
    if p1[0] < p2[0]:
        tl, tr = p1, p2
    else:
        tl, tr = p2, p1
        
    bottom_idx = [i for i in range(4) if i not in top_edge]
    p3, p4 = pts[bottom_idx[0]], pts[bottom_idx[1]]
    if p3[0] < p4[0]:
        bl, br = p3, p4
    else:
        bl, br = p4, p3
        
    return np.array([tl, tr, br, bl], dtype=np.float32)


"""
3. Megkeresi a kártyalap kontúrját a képen, visszaadja a sarkok koordinátáit
"""
def find_card_quad(image_gray: np.ndarray) -> np.ndarray:
    h, w = image_gray.shape[:2]
    scale = 800.0 / w
    if scale < 1.0:
        gray_scaled = cv2.resize(image_gray, (0, 0), fx=scale, fy=scale)
    else:
        gray_scaled = image_gray
        scale = 1.0
        
    blurred = cv2.GaussianBlur(gray_scaled, (5, 5), 0)
    
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Nem talált kártya-kontúrt")
        
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    min_area = int(gray_scaled.shape[0] * gray_scaled.shape[1] * 0.05) 
    
    quad = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            break
            
        peri = cv2.arcLength(contour, True)
        for eps in np.linspace(0.01, 0.1, 15):
            approx = cv2.approxPolyDP(contour, eps * peri, True)
            if len(approx) == 4:
                quad = approx.reshape(4, 2).astype(np.float32)
                break
        if quad is not None:
            break
            
    if quad is None:
        edges_fallback = cv2.Canny(blurred, 15, 50)
        edges_fallback = cv2.dilate(edges_fallback, np.ones((3, 3), dtype=np.uint8), iterations=1)
        fallback_contours, _ = cv2.findContours(edges_fallback, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_pts = []
        for c in fallback_contours:
            if cv2.contourArea(c) > 50:
                valid_pts.append(c)
                
        if not valid_pts:
            raise RuntimeError("Nem talált sem élt, sem belső objektumokat a kártyán!")
            
        all_pts = np.vstack(valid_pts)
        rect = cv2.minAreaRect(all_pts)
        
        center, size, angle = rect
        rect = (center, (size[0]*1.15, size[1]*1.15), angle)
        
        quad = cv2.boxPoints(rect).astype(np.float32)
        
    quad = quad / scale
    return order_points(quad)


"""
5. Perspektíva transzformációval kiegyenesíti a kártyát hogy mindig ugyanakkora legyen
"""
def warp_card(image_bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    target = np.array(
        [[0, 0], [CARD_WIDTH - 1, 0], [CARD_WIDTH - 1, CARD_HEIGHT - 1], [0, CARD_HEIGHT - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad, target)
    return cv2.warpPerspective(image_bgr, matrix, (CARD_WIDTH, CARD_HEIGHT))


"""
7. Előkészíti a kivágott szimbólum képet (szürkeárnyalatosítás, elmosás, binarizálás)
"""
def preprocess_symbol(symbol_img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(symbol_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]



"""
9. Normalizálja a szimbólum bináris képét közös méretre
"""
def normalize_binary_symbol(symbol_img: np.ndarray) -> np.ndarray:
    gray = symbol_img.copy()

    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_points = [c for c in contours if cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3] > 10]
    
    if not valid_points:
        return np.zeros((MATCH_CANVAS, MATCH_CANVAS), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(np.vstack(valid_points))
    cropped = binary[y:y + h, x:x + w]

    max_side = max(w, h)
    scale = (MATCH_CANVAS - 2 * MATCH_PADDING) / max(max_side, 1)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((MATCH_CANVAS, MATCH_CANVAS), dtype=np.uint8)
    off_x = (MATCH_CANVAS - new_w) // 2
    off_y = (MATCH_CANVAS - new_h) // 2
    canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized
    return canvas


"""
11. Több template matching metrikát súlyozva egyetlen pontszámmá egyesít.
"""
def template_ensemble_score(query: np.ndarray, template: np.ndarray) -> float:
    score = 0.0
    for method, weight in TM_METHOD_WEIGHTS:
        raw = float(cv2.matchTemplate(query, template, method)[0, 0])
        method_score = 1.0 - raw if method == cv2.TM_SQDIFF_NORMED else raw
        score += weight * method_score
    return float(score)


"""
6. Megkeresi és kivágja a rang és a szín kontúrjait a sarokrégióból
"""
def extract_dynamic_symbols(warped_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    A kártyasarokból dinamikusan kinyeri a rang és suit szimbólumokat.
    """
    def _try_extract(corner_bgr):
        """
        Egy adott ROI képrészleten megpróbál két érvényes szimbólum kontúrt találni.
        """
        corner_binary = preprocess_symbol(corner_bgr)
        
        c_pre, _ = cv2.findContours(corner_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in c_pre:
            bx, by, bw, bh = cv2.boundingRect(c)
            if bh > 100 or bw > 45 or bx > 28:
                cv2.drawContours(corner_binary, [c], -1, 0, -1)
        
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        merged = cv2.morphologyEx(corner_binary, cv2.MORPH_CLOSE, kernel_close)
        
        contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_boxes = []
        for c in contours:
            bx, by, bw, bh = cv2.boundingRect(c)
            if bw >= 5 and bh >= 10:
                valid_boxes.append((bx, by, bw, bh, c))
                
        if len(valid_boxes) < 2:
            return None, None
            
        valid_boxes.sort(key=lambda b: b[1])
        return valid_boxes, corner_binary

    x, y, w, h = CORNER_ROI
    valid_boxes, corner_binary = _try_extract(warped_bgr[y:y+h, x:x+w])
    
    if valid_boxes is None:
        bw_w, bw_h = warped_bgr.shape[1], warped_bgr.shape[0]
        br_x, br_y = bw_w - w - x, bw_h - h - y
        corner_bgr_br = warped_bgr[br_y:br_y+h, br_x:br_x+w]
        corner_bgr_br = cv2.rotate(corner_bgr_br, cv2.ROTATE_180)
        
        valid_boxes, corner_binary = _try_extract(corner_bgr_br)
                
        if valid_boxes is None:
            raise RuntimeError("Nem talált megfelelő szimbólumokat a sarokban")

    rx, ry, rw, rh, _ = valid_boxes[0]
    sx, sy, sw, sh, _ = valid_boxes[1]
    
    pad = 4
    def crop_padded(bx, by, bw, bh):
        """
        A kontúr köré biztonsági margóval vágja ki a szimbólumot.
        """
        y1 = max(0, by - pad)
        y2 = min(corner_binary.shape[0], by + bh + pad)
        x1 = max(0, bx - pad)
        x2 = min(corner_binary.shape[1], bx + bw + pad)
        return corner_binary[y1:y2, x1:x2]
        
    rank_query = crop_padded(rx, ry, rw, rh)
    suit_query = crop_padded(sx, sy, sw, sh)
    
    return rank_query, suit_query


"""
8. Betölti a sablonképeket (rangok, színek)
"""
def load_templates(directory: Path) -> Dict[str, np.ndarray]:
    templates: Dict[str, np.ndarray] = {}
    for path in directory.glob("*.*"):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        templates[path.stem] = normalize_binary_symbol(img)
    return templates


"""
10. Megkeresi hogy a lekérdezett szimbólum melyik sablonhoz hasonlít legjobban
"""
def best_template_match(
    query: np.ndarray,
    templates: Dict[str, np.ndarray],
) -> Tuple[str, float]:
    query_norm = normalize_binary_symbol(query)
    
    def calculate_score(tpl):
        return template_ensemble_score(query_norm, tpl)

    best_name, best_tpl = max(templates.items(), key=lambda item: calculate_score(item[1]), default=("ismeretlen", None))
    return best_name, calculate_score(best_tpl) if best_tpl is not None else -1.0


"""
2. Betölti a képet, megtalálja a kártyát, kivágja a szimbólumokat,
összehasonlítja a sablonokkal, kiírja és megjeleníti az eredményt
"""
def recognize_single_card(
    image_path: Path,
    rank_templates_dir: Path,
    suit_templates_dir: Path,
    debug: bool = False,
) -> Tuple[str, str]:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Nem sikerűlt beolvasni a képet: {image_path}")

    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    quad = find_card_quad(image_gray)
    warped = warp_card(image_bgr, quad)

    rank_query, suit_query = extract_dynamic_symbols(warped)

    rank_templates = load_templates(rank_templates_dir)
    suit_templates = load_templates(suit_templates_dir)

    rank_name, rank_score = best_template_match(rank_query, rank_templates)
    suit_name, suit_score = best_template_match(suit_query, suit_templates)

    """
    matplotlib segítségével vizuális megjelenítés a kapott eredményről
    (Egy kártya ellenőrzése esetén érdemes használni)
    """
    if debug:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 8))
        
        axes[0].imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Eredeti kép")
        axes[0].axis("off")

        
        axes[1].imshow(rank_query, cmap="gray")
        axes[1].set_title(f"Rang (talált: {rank_name})")
        axes[1].axis("off")
        
        axes[2].imshow(suit_query, cmap="gray")
        axes[2].set_title(f"Szín (talált: {suit_name})")
        axes[2].axis("off")
        
        
        plt.tight_layout()
        plt.show()

    return rank_name, suit_name



"""
1. Parancssori argumentumok feldolgozása
"""
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--rank-templates", default=Path("templates/rank"), type=Path)
    parser.add_argument("--suit-templates", default=Path("templates/suit"), type=Path)   
    parser.add_argument("--debug", action="store_true")    
    return parser.parse_args()

"""
Belépési pont: argumentumokat olvas, majd elindítja az egyképes felismerést.
"""
def main() -> None:
    args = parse_args()
    recognize_single_card(
        image_path=args.image,
        rank_templates_dir=args.rank_templates,
        suit_templates_dir=args.suit_templates,
        debug=args.debug
    )


if __name__ == "__main__":
    main()
