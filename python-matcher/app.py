import base64
import io
import math
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image

app = FastAPI(title="Template Matcher", version="0.1.0")

MAX_GLOBAL_SEARCH_DIMENSION = 1600
MIN_TEMPLATE_DIMENSION = 5
MAX_PASS_HITS_MULTIPLIER = 6


def clamp(value: float, min_value: float, max_value: float) -> float:
  return max(min_value, min(max_value, value))


def decode_image_data(data: str) -> np.ndarray:
  if not data:
    raise ValueError("No image data supplied")
  payload = data.strip()
  if payload.startswith("data:image"):
    header, encoded = payload.split(",", 1)
  else:
    encoded = payload
  buffer = base64.b64decode(encoded)
  pil_image = Image.open(io.BytesIO(buffer))
  return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def to_grayscale(image: np.ndarray) -> np.ndarray:
  if len(image.shape) == 2:
    return image
  return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalized_to_pixels(box: "Selection", width: int, height: int):
  x = clamp(box.x, 0.0, 1.0)
  y = clamp(box.y, 0.0, 1.0)
  w = clamp(box.width, 0.0001, 1.0 - x)
  h = clamp(box.height, 0.0001, 1.0 - y)
  left = int(round(x * width))
  top = int(round(y * height))
  right = int(round((x + w) * width))
  bottom = int(round((y + h) * height))
  return left, top, right, bottom


def expand_box(box: "Selection", padding: float) -> "Selection":
  if padding <= 0:
    return box
  pad_w = box.width * padding
  pad_h = box.height * padding
  x = clamp(box.x - pad_w, 0.0, 1.0)
  y = clamp(box.y - pad_h, 0.0, 1.0)
  right = clamp(box.x + box.width + pad_w, 0.0, 1.0)
  bottom = clamp(box.y + box.height + pad_h, 0.0, 1.0)
  return Selection(x=x, y=y, width=right - x, height=bottom - y)


def maybe_downscale_search(
  search_patch: np.ndarray, template_patch: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
  search_height, search_width = search_patch.shape[:2]
  max_dim = max(search_height, search_width)
  if max_dim <= MAX_GLOBAL_SEARCH_DIMENSION:
    return search_patch, template_patch, 1.0
  scale_factor = MAX_GLOBAL_SEARCH_DIMENSION / max_dim
  template_min_dim = max(1, min(template_patch.shape[:2]))
  min_scale_allowed = max(MIN_TEMPLATE_DIMENSION / template_min_dim, 0.05)
  scale_factor = max(scale_factor, min_scale_allowed)
  scale_factor = min(scale_factor, 1.0)
  if scale_factor >= 0.999:
    return search_patch, template_patch, 1.0
  new_search_width = max(8, int(round(search_width * scale_factor)))
  new_search_height = max(8, int(round(search_height * scale_factor)))
  resized_search = cv2.resize(search_patch, (new_search_width, new_search_height), interpolation=cv2.INTER_AREA)
  new_template_width = max(MIN_TEMPLATE_DIMENSION, int(round(template_patch.shape[1] * scale_factor)))
  new_template_height = max(MIN_TEMPLATE_DIMENSION, int(round(template_patch.shape[0] * scale_factor)))
  resized_template = cv2.resize(template_patch, (new_template_width, new_template_height), interpolation=cv2.INTER_AREA)
  return resized_search, resized_template, scale_factor


def rotate_template(template: np.ndarray, angle: float) -> np.ndarray:
  if angle == 0:
    return template
  center = (template.shape[1] / 2, template.shape[0] / 2)
  matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
  abs_cos = abs(matrix[0, 0])
  abs_sin = abs(matrix[0, 1])
  width = int(template.shape[0] * abs_sin + template.shape[1] * abs_cos)
  height = int(template.shape[0] * abs_cos + template.shape[1] * abs_sin)
  matrix[0, 2] += width / 2 - center[0]
  matrix[1, 2] += height / 2 - center[1]
  rotated = cv2.warpAffine(template, matrix, (width, height), flags=cv2.INTER_LINEAR)
  return rotated


def apply_mode_preprocess(template: np.ndarray, mode: str) -> np.ndarray:
  if mode == "linework":
    blurred = cv2.GaussianBlur(template, (3, 3), 0.4)
    edges = cv2.Canny(blurred, 24, 72)
    return edges.astype(np.float32) / 255.0
  if mode == "texture":
    return cv2.GaussianBlur(template, (5, 5), 0.8).astype(np.float32) / 255.0
  # auto
  normalized = cv2.normalize(template, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
  return normalized.astype(np.float32) / 255.0


def non_max_suppression(boxes: List[dict], iou_threshold: float, max_detections: int) -> List[dict]:
  if not boxes:
    return []
  boxes = sorted(boxes, key=lambda item: item["score"], reverse=True)
  picked: List[dict] = []
  while boxes and len(picked) < max_detections:
    current = boxes.pop(0)
    picked.append(current)
    boxes = [candidate for candidate in boxes if intersection_over_union(candidate, current) <= iou_threshold]
  return picked


def intersection_over_union(a: dict, b: dict) -> float:
  x1 = max(a["x"], b["x"])
  y1 = max(a["y"], b["y"])
  x2 = min(a["x"] + a["width"], b["x"] + b["width"])
  y2 = min(a["y"] + a["height"], b["y"] + b["height"])
  inter_area = max(0, x2 - x1) * max(0, y2 - y1)
  if inter_area <= 0:
    return 0.0
  area_a = a["width"] * a["height"]
  area_b = b["width"] * b["height"]
  union = area_a + area_b - inter_area
  return inter_area / union if union > 0 else 0.0


class Selection(BaseModel):
  x: float = Field(..., ge=0.0, le=1.0)
  y: float = Field(..., ge=0.0, le=1.0)
  width: float = Field(..., gt=0.0, le=1.0)
  height: float = Field(..., gt=0.0, le=1.0)


class MatchOptions(BaseModel):
  mode: str = Field("auto", pattern="^(auto|linework|texture)$")
  min_similarity: float = Field(0.6, ge=-1.0, le=1.0)
  max_matches: int = Field(80, ge=1, le=400)
  nms_iou: float = Field(0.35, ge=0.0, le=1.0)
  scales: Optional[List[float]] = None
  rotations: Optional[List[float]] = None
  search_padding: Optional[float] = Field(None, ge=0.0, le=5.0)
  label: Optional[str] = None


class MatchRequest(BaseModel):
  image: str
  coordinates: Selection
  options: Optional[MatchOptions] = None


@app.get("/health")
def health_check():
  return {"status": "ok"}


@app.post("/match-selection")
def match_selection(payload: MatchRequest):
  try:
    image_bgr = decode_image_data(payload.image)
  except Exception as exc:  # noqa: BLE001
    raise HTTPException(status_code=400, detail=f"Failed to decode image: {exc}") from exc

  height, width = image_bgr.shape[:2]
  if width == 0 or height == 0:
    raise HTTPException(status_code=400, detail="Invalid image dimensions")

  selection = payload.coordinates
  options = payload.options or MatchOptions()
  template_label = options.label or "template-match"
  min_similarity = clamp(options.min_similarity, -1.0, 1.0)
  search_box = Selection(x=0, y=0, width=1, height=1) if options.search_padding is None else expand_box(selection, options.search_padding)
  sel_left, sel_top, sel_right, sel_bottom = normalized_to_pixels(selection, width, height)
  search_left, search_top, search_right, search_bottom = normalized_to_pixels(search_box, width, height)

  image_gray = to_grayscale(image_bgr)
  search_patch = image_gray[search_top:search_bottom, search_left:search_right]
  template_patch = image_gray[sel_top:sel_bottom, sel_left:sel_right]
  if template_patch.shape[0] < 5 or template_patch.shape[1] < 5:
    raise HTTPException(status_code=400, detail="Selection too small for template matching")

  search_patch_proc, template_patch_proc, downscale_factor = maybe_downscale_search(search_patch, template_patch)
  downscale_factor = max(downscale_factor, 1e-3)
  base_template = apply_mode_preprocess(template_patch_proc, options.mode)
  search_surface = search_patch_proc.astype(np.float32) / 255.0
  search_height, search_width = search_surface.shape[:2]
  default_scales = options.scales or [0.65, 0.85, 1.0, 1.2, 1.45]
  default_rotations = options.rotations or [0.0]

  candidates: List[dict] = []
  stats: List[dict] = []

  for scale in default_scales:
    scaled_template = cv2.resize(base_template, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    if scaled_template.shape[0] < 5 or scaled_template.shape[1] < 5:
      continue
    if scaled_template.shape[0] >= search_height or scaled_template.shape[1] >= search_width:
      continue
    for rotation in default_rotations:
      variant = rotate_template(scaled_template, rotation) if rotation else scaled_template
      if variant.shape[0] >= search_height or variant.shape[1] >= search_width:
        continue
      pass_name = f"scale={scale:.2f}_rot={rotation:.0f}"
      start = time.perf_counter()
      result = cv2.matchTemplate(search_surface, variant, cv2.TM_CCOEFF_NORMED)
      duration_ms = (time.perf_counter() - start) * 1000
      best_similarity = float(result.max()) if result.size else None
      kept_points = 0
      if result.size and best_similarity is not None:
        locations = np.where(result >= min_similarity)
        hits = list(zip(locations[1], locations[0]))  # x, y order
        max_hits = max(options.max_matches * MAX_PASS_HITS_MULTIPLIER, 400)
        if len(hits) > max_hits:
          # keep highest-scoring hits only
          scored_hits = sorted(((float(result[y, x]), (x, y)) for x, y in hits), key=lambda item: item[0], reverse=True)
          hits = [item[1] for item in scored_hits[:max_hits]]
        kept_points = len(hits)
        for x, y in hits:
          score = float(result[y, x])
          box_width = variant.shape[1]
          box_height = variant.shape[0]
          abs_x = search_left + x / downscale_factor
          abs_y = search_top + y / downscale_factor
          candidates.append(
            {
              "x": abs_x,
              "y": abs_y,
              "width": box_width / downscale_factor,
              "height": box_height / downscale_factor,
              "score": score,
              "pass": pass_name,
            }
          )
      stats.append(
        {
          "pass": pass_name,
          "candidates": int(result.size),
          "kept": kept_points,
          "durationMs": duration_ms,
          "bestSimilarity": best_similarity,
        }
      )

  if not candidates:
    return {
      "matches": [],
      "stats": stats,
      "bestSimilarity": None,
      "selection": selection,
      "searchRegion": search_box,
    }

  filtered = non_max_suppression(candidates, options.nms_iou, options.max_matches)
  matches = []
  for candidate in filtered:
    matches.append(
      {
        "label": template_label,
        "confidence": candidate["score"],
        "boundingBox": {
          "x": clamp(candidate["x"] / width, 0.0, 1.0),
          "y": clamp(candidate["y"] / height, 0.0, 1.0),
          "width": clamp(candidate["width"] / width, 0.0, 1.0),
          "height": clamp(candidate["height"] / height, 0.0, 1.0),
        },
        "templateSimilarity": candidate["score"],
        "templatePass": candidate["pass"],
      }
    )

  matches.sort(key=lambda item: item["templateSimilarity"], reverse=True)
  best_similarity = matches[0]["templateSimilarity"] if matches else None

  return {
    "matches": matches,
    "stats": stats,
    "bestSimilarity": best_similarity,
    "selection": selection,
    "searchRegion": search_box,
  }
