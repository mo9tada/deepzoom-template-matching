"""Tests for the template-matching service.

The fixtures draw a synthetic "blueprint": a blank sheet with the same symbol
stamped at known positions. Selecting one symbol must find all of them.
"""
import base64
import io
import os
import sys

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import (  # noqa: E402
  Selection,
  app,
  decode_image_data,
  expand_box,
  intersection_over_union,
  non_max_suppression,
  normalized_to_pixels,
)

client = TestClient(app)

SHEET_W, SHEET_H = 800, 600
SYMBOL = 40
SYMBOL_POSITIONS = [(100, 100), (400, 120), (250, 380), (620, 450)]


def draw_symbol(img: np.ndarray, x: int, y: int) -> None:
  """A door-like symbol: a square with a diagonal and a quarter arc."""
  cv2.rectangle(img, (x, y), (x + SYMBOL, y + SYMBOL), 0, 2)
  cv2.line(img, (x, y + SYMBOL), (x + SYMBOL, y), 0, 2)
  cv2.ellipse(img, (x, y + SYMBOL), (SYMBOL, SYMBOL), 0, 270, 360, 0, 2)


def make_sheet(mode: str = "RGB") -> Image.Image:
  gray = np.full((SHEET_H, SHEET_W), 255, dtype=np.uint8)
  for x, y in SYMBOL_POSITIONS:
    draw_symbol(gray, x, y)
  # a distractor that must not match
  cv2.circle(gray, (680, 120), 30, 0, 2)
  return Image.fromarray(gray).convert(mode)


def to_data_url(image: Image.Image) -> str:
  buf = io.BytesIO()
  image.save(buf, format="PNG")
  return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def selection_for(x: int, y: int) -> dict:
  pad = 4
  return {
    "x": (x - pad) / SHEET_W,
    "y": (y - pad) / SHEET_H,
    "width": (SYMBOL + 2 * pad) / SHEET_W,
    "height": (SYMBOL + 2 * pad) / SHEET_H,
  }


# --- pure helpers -------------------------------------------------------------

def test_iou_identical_and_disjoint():
  a = {"x": 0, "y": 0, "width": 10, "height": 10}
  assert intersection_over_union(a, a) == pytest.approx(1.0)
  assert intersection_over_union(a, {"x": 20, "y": 20, "width": 5, "height": 5}) == 0.0


def test_nms_keeps_best_of_overlapping_boxes():
  boxes = [
    {"x": 0, "y": 0, "width": 10, "height": 10, "score": 0.9},
    {"x": 1, "y": 1, "width": 10, "height": 10, "score": 0.8},
    {"x": 50, "y": 50, "width": 10, "height": 10, "score": 0.7},
  ]
  picked = non_max_suppression(boxes, iou_threshold=0.3, max_detections=10)
  assert [b["score"] for b in picked] == [0.9, 0.7]


def test_normalized_to_pixels_clamps_to_image():
  box = Selection(x=0.9, y=0.9, width=0.5, height=0.5)
  left, top, right, bottom = normalized_to_pixels(box, 100, 200)
  assert (left, top) == (90, 180)
  assert right <= 100 and bottom <= 200


def test_expand_box_stays_inside_unit_square():
  grown = expand_box(Selection(x=0.0, y=0.0, width=0.2, height=0.2), padding=1.0)
  assert grown.x == 0.0 and grown.y == 0.0
  assert grown.x + grown.width <= 1.0


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L", "P"])
def test_decode_handles_common_png_modes(mode):
  decoded = decode_image_data(to_data_url(make_sheet(mode)))
  assert decoded.shape == (SHEET_H, SHEET_W, 3)


# --- API ----------------------------------------------------------------------

def test_health():
  assert client.get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize("mode", ["auto", "linework"])
def test_finds_every_copy_of_the_selected_symbol(mode):
  x, y = SYMBOL_POSITIONS[0]
  response = client.post(
    "/match-selection",
    json={
      "image": to_data_url(make_sheet()),
      "coordinates": selection_for(x, y),
      "options": {"mode": mode, "min_similarity": 0.5, "scales": [1.0]},
    },
  )
  assert response.status_code == 200
  matches = response.json()["matches"]

  centers = [
    (
      (m["boundingBox"]["x"] + m["boundingBox"]["width"] / 2) * SHEET_W,
      (m["boundingBox"]["y"] + m["boundingBox"]["height"] / 2) * SHEET_H,
    )
    for m in matches
  ]
  for sx, sy in SYMBOL_POSITIONS:
    expected = (sx + SYMBOL / 2, sy + SYMBOL / 2)
    assert any(abs(cx - expected[0]) < 10 and abs(cy - expected[1]) < 10 for cx, cy in centers), (
      f"symbol at {(sx, sy)} not found"
    )
  # the distractor circle must not be reported
  assert not any(abs(cx - 680) < 25 and abs(cy - 120) < 25 for cx, cy in centers)


def test_rejects_invalid_image():
  response = client.post(
    "/match-selection",
    json={"image": "not-base64!!", "coordinates": selection_for(100, 100)},
  )
  assert response.status_code == 400


def test_rejects_tiny_selection():
  response = client.post(
    "/match-selection",
    json={
      "image": to_data_url(make_sheet()),
      "coordinates": {"x": 0.1, "y": 0.1, "width": 0.001, "height": 0.001},
    },
  )
  assert response.status_code == 400
