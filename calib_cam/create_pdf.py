"""
Generate ArUco (4x4_50) markers to a A4 PDF.
- Border width = final_size_cm / 8 (as requested).
- No repeated markers (IDs start at 0, stop before 50).
- Draws a dashed cut-guide exactly on the outer edge (no margin).
"""

import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
import os

# --- User input: groups = (quantity, final_size_cm) ---
groups = [
    (1, 12),
    (1, 10),
]

# --- Config ---
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
PDF_OUT = "aruco_markers_fapesp.pdf"

# Print quality (dpi). Increase for very small markers; 600 is good for small prints.
DPI = 600
px_per_cm = DPI / 2.54  # pixels per centimeter

# PDF layout
c = canvas.Canvas(PDF_OUT, pagesize=A4)
page_w, page_h = A4
page_margin = 1.0 * cm
x = page_margin
y = page_h - page_margin
spacing = 0.5 * cm  # horizontal gap
row_spacing = 5.5 * cm  # vertical gap

marker_id = 0

for qty, final_size_cm in groups:
    for _ in range(qty):
        if marker_id >= 50:
            raise RuntimeError("DICT_4X4_50 supports only 50 ids (0..49). Too many markers requested.")

        # Compute border and inner marker sizes (cm -> px)
        border_cm = final_size_cm / 8.0
        inner_cm = max(0.0, final_size_cm - 2.0 * border_cm)

        # convert to pixels
        border_px = max(1, int(round(border_cm * px_per_cm)))
        inner_px = max(4, int(round(inner_cm * px_per_cm)))

        # Generate inner marker
        try:
            marker_inner = cv2.aruco.drawMarker(DICT, marker_id, inner_px)
        except Exception:
            marker_inner = cv2.aruco.generateImageMarker(DICT, marker_id, inner_px)

        # Add white border = 1 module each side
        marker_with_border = cv2.copyMakeBorder(
            marker_inner,
            border_px, border_px, border_px, border_px,
            borderType=cv2.BORDER_CONSTANT,
            value=255
        )

        # Save temporary PNG
        fname = f"marker_{marker_id:02d}.png"
        cv2.imwrite(fname, marker_with_border)

        # Placement size in PDF
        size_pt = final_size_cm * cm

        # Wrap to next line if doesn't fit horizontally
        if x + size_pt > page_w - page_margin:
            x = page_margin
            y -= (size_pt + row_spacing)

        # New page if doesn't fit vertically
        if y - size_pt < page_margin:
            c.showPage()
            x = page_margin
            y = page_h - page_margin

        # Draw marker image
        c.drawImage(fname, x, y - size_pt, width=size_pt, height=size_pt,
                    preserveAspectRatio=False, mask='auto')

        # Draw dashed cut-guide exactly on the border
        c.setLineWidth(0.5)
        c.setStrokeColorRGB(0.2, 0.2, 0.2)
        c.setDash([3, 3], 0)
        c.rect(x, y - size_pt, size_pt, size_pt, stroke=1, fill=0)
        c.setDash()  # reset

        # Advance x
        x += size_pt + spacing
        marker_id += 1

        # Clean up temp PNG
        try:
            os.remove(fname)
        except OSError:
            pass

# finalize PDF
c.save()
print(f"PDF generated: {PDF_OUT}")
