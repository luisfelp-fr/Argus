# -*- coding: utf-8 -*-
"""
Gera o ícone do app: um olho verde (tema Argus, o gigante de muitos olhos).
Salva assets/eye_icon.png (256x256, fundo transparente) com anti-aliasing.

Uso:  python assets/generate_icon.py
"""

import os

import numpy as np
from PIL import Image

SS = 1024          # resolução interna (supersampling p/ bordas suaves)
OUT = 256          # tamanho final
t = 26             # espessura do contorno (escala interna)

cx = cy = SS / 2

# cores (RGBA) — verde dominante, tom Argus
SCLERA    = (244, 247, 245, 255)
IRIS      = (26, 160, 90, 255)     # verde vivo
IRIS_RING = (17, 110, 66, 255)     # anel mais escuro (motivo concêntrico)
PUPIL     = (10, 28, 20, 255)
HILIGHT   = (255, 255, 255, 235)
OUTLINE   = (15, 74, 52, 255)      # verde escuro

# geometria do amêndoa (interseção de dois círculos)
W = 360.0          # meia-largura do olho
h0 = 215.0         # meia-altura no centro
d = (W * W - h0 * h0) / (2 * h0)   # deslocamento dos centros
R = d + h0                          # raio dos círculos

yy, xx = np.mgrid[0:SS, 0:SS].astype(float)
dist_c = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)


def almond(radius):
    c1 = (xx - cx) ** 2 + (yy - (cy - d)) ** 2 <= radius ** 2
    c2 = (xx - cx) ** 2 + (yy - (cy + d)) ** 2 <= radius ** 2
    return c1 & c2


eye = almond(R)
eye_in = almond(R - t)
outline = eye & ~eye_in

ri = 210.0          # raio da íris
iris = (dist_c <= ri) & eye_in
iris_ring = (dist_c <= ri) & (dist_c >= ri - 22) & eye_in
pupil = dist_c <= 95.0
hi = np.sqrt((xx - (cx - 55)) ** 2 + (yy - (cy - 70)) ** 2) <= 38.0

img = np.zeros((SS, SS, 4), dtype=np.uint8)
img[eye_in] = SCLERA
img[iris] = IRIS
img[iris_ring] = IRIS_RING
img[pupil & iris] = PUPIL
img[hi] = HILIGHT
img[outline] = OUTLINE

pil = Image.fromarray(img, "RGBA").resize((OUT, OUT), Image.LANCZOS)
out_path = os.path.join(os.path.dirname(__file__), "eye_icon.png")
pil.save(out_path)
print("OK ->", out_path)
