from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

CARD_WIDTH = 200
CARD_HEIGHT = 300
RANK_ROI = (8, 8, 64, 84)
SUIT_ROI = (8, 92, 64, 152)


"""
Rendezi a négyszög sarkait bal felső, jobb felső, jobb alsó, bal alsó sorrendbe
"""
def order_points(points: np.ndarray) -> np.ndarray:
    pts = points.astype(np.float32)
    pts = pts[np.argsort(pts[:, 1])]
    top = pts[:2]
    bottom = pts[2:]
    top = top[np.argsort(top[:, 0])]
    tl, tr = top
    bottom = bottom[np.argsort(bottom[:, 0])]
    bl, br = bottom
    return np.array([tl, tr, br, bl], dtype=np.float32)


"""
Megkeresi a kártyalap kontúrját a képen, visszaadja a sarkok koordinátáit
"""
def find_card_quad(image_gray: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
    h, w = image_gray.shape[:2]
    min_area = max(4000, int(h * w * 0.01))

    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Nem talált kártya-kontúrt")
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < min_area:
        raise RuntimeError("Nem talált elég nagy kártya-kontúrt")
    x, y, bw, bh = cv2.boundingRect(contour)
    quad = np.array([
        [x, y],
        [x + bw, y],
        [x + bw, y + bh],
        [x, y + bh]
    ], dtype=np.float32)
    return order_points(quad)


"""
Perspektíva transzformációval kiegyenesíti a kártyát hogy mindig ugyanakkora legyen
"""
def warp_card(image_bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    target = np.array(
        [[0, 0], [CARD_WIDTH - 1, 0], [CARD_WIDTH - 1, CARD_HEIGHT - 1], [0, CARD_HEIGHT - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad, target)
    return cv2.warpPerspective(image_bgr, matrix, (CARD_WIDTH, CARD_HEIGHT))


"""
Előkészíti a kivágott szimbólum képet (szürkeárnyalatosítás, elmosás, binarizálás)
"""
def preprocess_symbol(symbol_img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(symbol_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


"""
Betölti a sablonképeket (rangok, színek)
"""
def load_templates(directory: Path) -> Dict[str, np.ndarray]:
    templates: Dict[str, np.ndarray] = {}
    for path in directory.glob("*.*"):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        templates[path.stem] = img
    return templates


"""
Megkeresi hogy a lekérdezett szimbólum melyik sablonhoz hasonlít legjobban
"""
def best_template_match(query: np.ndarray, templates: Dict[str, np.ndarray]) -> Tuple[str, float]:
    best_name = "ismeretlen"
    best_score = -1.0
    for name, tpl in templates.items():
        resized = cv2.resize(query, (tpl.shape[1], tpl.shape[0]), interpolation=cv2.INTER_AREA)
        score = cv2.matchTemplate(resized, tpl, cv2.TM_CCOEFF_NORMED)[0, 0]
        if score > best_score:
            best_score = float(score)
            best_name = name
    return best_name, best_score


"""
Betölti a képet, megtalálja a kártyát, kivágja a szimbólumokat,
összehasonlítja a sablonokkal, kiírja és megjeleníti az eredményt
"""
def recognize_single_card(
    image_path: Path,
    rank_templates_dir: Path,
    suit_templates_dir: Path,
    debug: bool = False,
) -> None:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Nem sikerűlt beolvasni a képet: {image_path}")

    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    quad = find_card_quad(image_gray)
    warped = warp_card(image_bgr, quad)

    rx1, ry1, rx2, ry2 = RANK_ROI
    sx1, sy1, sx2, sy2 = SUIT_ROI

    rank_query = preprocess_symbol(warped[ry1:ry2, rx1:rx2])
    suit_query = preprocess_symbol(warped[sy1:sy2, sx1:sx2])

    rank_templates = load_templates(rank_templates_dir)
    suit_templates = load_templates(suit_templates_dir)

    rank_name, rank_score = best_template_match(rank_query, rank_templates)
    suit_name, suit_score = best_template_match(suit_query, suit_templates)

    print(f"Felismert lap: {rank_name}_{suit_name}")

    view = image_bgr.copy()
    quad_int = quad.astype(int)
    cv2.polylines(view, [quad_int], True, (0, 255, 0), 2)
    cv2.imshow("Input + Detected Card", view)
    cv2.imshow("Warped Card", warped)
    cv2.imshow("Rank Query", rank_query)
    cv2.imshow("Suit Query", suit_query)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
Parancssori argumentumok feldolgozása
"""
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--rank-templates", default=Path("templates/rank"), type=Path)
    parser.add_argument("--suit-templates", default=Path("templates/suit"), type=Path)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    recognize_single_card(
        image_path=args.image,
        rank_templates_dir=args.rank_templates,
        suit_templates_dir=args.suit_templates
    )


if __name__ == "__main__":
    main()
