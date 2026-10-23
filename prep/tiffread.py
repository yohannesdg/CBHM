"""Windowed GeoTIFF reading without the 'imagecodecs' package.

tifffile parses the tags and segment offsets. The strips or tiles that overlap the requested window
are decoded here: uncompressed, Deflate (zlib, from the standard library) or LZW (a pure-Python
decoder). Only the window is decoded, so a basin cut from the HydroSHEDS Africa DEM takes seconds,
not the whole 150 MB raster.
"""
from __future__ import annotations

import zlib

import numpy as np


def lzw_decode(data: bytes) -> bytes:
    """TIFF-flavour LZW (MSB-first codes, 9-12 bits, early change, Clear=256, EOI=257)."""
    data = bytes(data) + b"\x00\x00\x00"
    nbits_total = (len(data) - 3) * 8
    out = bytearray()
    table = [bytes([i]) for i in range(256)] + [b"", b""]
    append = table.append
    nbits, bitpos, prev = 9, 0, None
    from_bytes = int.from_bytes
    while bitpos + nbits <= nbits_total:
        byte = bitpos >> 3
        code = (from_bytes(data[byte:byte + 3], "big") >> (24 - (bitpos & 7) - nbits)) & ((1 << nbits) - 1)
        bitpos += nbits
        if code == 257:
            break
        if code == 256:
            del table[258:]
            nbits, prev = 9, None
            continue
        if prev is None:
            entry = table[code]
        elif code < len(table):
            entry = table[code]
            append(prev + entry[:1])
        else:
            entry = prev + prev[:1]
            append(entry)
        out += entry
        prev = entry
        n = len(table)
        if n >= 511 and nbits == 9 or n >= 1023 and nbits == 10 or n >= 2047 and nbits == 11:
            nbits += 1
    return bytes(out)


def _decode(raw: bytes, compression: int) -> bytes:
    if compression == 1:
        return raw
    if compression in (8, 32946):
        return zlib.decompress(raw)
    if compression == 5:
        return lzw_decode(raw)
    if compression == 32773:  # PackBits
        out, i = bytearray(), 0
        while i < len(raw):
            n = raw[i] if raw[i] < 128 else raw[i] - 256
            i += 1
            if n >= 0:
                out += raw[i:i + n + 1]; i += n + 1
            elif n != -128:
                out += bytes([raw[i]]) * (1 - n); i += 1
        return bytes(out)
    raise ValueError(f"TIFF compression {compression} is not supported without imagecodecs")


def read_window(path, r0, r1, c0, c1):
    """Rows r0:r1 and columns c0:c1 of the first band of page 0, in the file's dtype."""
    import tifffile
    with tifffile.TiffFile(path) as tf, open(path, "rb") as fh:
        page = tf.pages[0]
        dtype = np.dtype(page.dtype).newbyteorder(tf.byteorder)
        comp = int(page.compression)
        pred = int(getattr(page, "predictor", 1) or 1)
        spp = int(page.samplesperpixel)
        if pred not in (1, 2):
            raise ValueError(f"TIFF predictor {pred} is not supported")
        H, W = page.imagelength, page.imagewidth
        out = np.empty((r1 - r0, c1 - c0), dtype=dtype.newbyteorder("="))
        offs, cnts = page.dataoffsets, page.databytecounts

        def seg(i, shape):
            fh.seek(offs[i]); raw = fh.read(cnts[i])
            a = np.frombuffer(_decode(raw, comp), dtype=dtype)
            need = shape[0] * shape[1] * spp
            if a.size < need:
                a = np.concatenate([a, np.zeros(need - a.size, dtype)])
            a = a[:need].reshape(shape[0], shape[1], spp)[..., 0].astype(dtype.newbyteorder("="))
            if pred == 2:
                a = np.cumsum(a, axis=1, dtype=a.dtype)
            return a

        if page.is_tiled:
            th, tw = page.tilelength, page.tilewidth
            tiles_across = -(-W // tw)
            for ty in range(r0 // th, (r1 - 1) // th + 1):
                for tx in range(c0 // tw, (c1 - 1) // tw + 1):
                    t = seg(ty * tiles_across + tx, (th, tw))
                    ya, yb = max(r0, ty * th), min(r1, (ty + 1) * th, H)
                    xa, xb = max(c0, tx * tw), min(c1, (tx + 1) * tw, W)
                    out[ya - r0:yb - r0, xa - c0:xb - c0] = t[ya - ty * th:yb - ty * th, xa - tx * tw:xb - tx * tw]
        else:
            rps = min(page.rowsperstrip or H, H)
            for si in range(r0 // rps, (r1 - 1) // rps + 1):
                rows = min(rps, H - si * rps)
                s = seg(si, (rows, W))
                ya, yb = max(r0, si * rps), min(r1, si * rps + rows)
                out[ya - r0:yb - r0] = s[ya - si * rps:yb - si * rps, c0:c1]
        return out
