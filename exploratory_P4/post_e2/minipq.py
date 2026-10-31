"""Fallback only (used when pyarrow is unavailable). Minimal pure-Python parquet reader (flat columns; snappy/uncompressed; PLAIN and dictionary encodings).

Enough for pyarrow-written pandas frames. List/nested columns are skipped.
"""
import struct


def snappy_decompress(b):
    pos, n, shift = 0, 0, 0
    while True:
        c = b[pos]; pos += 1
        n |= (c & 0x7F) << shift; shift += 7
        if not c & 0x80:
            break
    out = bytearray()
    while pos < len(b):
        tag = b[pos]; pos += 1
        t = tag & 3
        if t == 0:
            ln = tag >> 2
            if ln >= 60:
                nb = ln - 59
                ln = int.from_bytes(b[pos:pos + nb], 'little'); pos += nb
            ln += 1
            out += b[pos:pos + ln]; pos += ln
            continue
        if t == 1:
            ln = ((tag >> 2) & 7) + 4
            off = ((tag >> 5) << 8) | b[pos]; pos += 1
        elif t == 2:
            ln = (tag >> 2) + 1
            off = int.from_bytes(b[pos:pos + 2], 'little'); pos += 2
        else:
            ln = (tag >> 2) + 1
            off = int.from_bytes(b[pos:pos + 4], 'little'); pos += 4
        start = len(out) - off
        for k in range(ln):
            out.append(out[start + k])
    assert len(out) == n, (len(out), n)
    return bytes(out)


class Thrift:
    def __init__(self, b, pos=0):
        self.b, self.pos = b, pos

    def byte(self):
        v = self.b[self.pos]; self.pos += 1; return v

    def varint(self):
        n, shift = 0, 0
        while True:
            c = self.byte()
            n |= (c & 0x7F) << shift; shift += 7
            if not c & 0x80:
                return n

    def zz(self):
        n = self.varint()
        return (n >> 1) ^ -(n & 1)

    def value(self, t):
        if t == 1:
            return True
        if t == 2:
            return False
        if t == 3:
            v = self.byte(); return v - 256 if v > 127 else v
        if t in (4, 5, 6):
            return self.zz()
        if t == 7:
            v = struct.unpack('<d', self.b[self.pos:self.pos + 8])[0]; self.pos += 8; return v
        if t == 8:
            n = self.varint(); v = self.b[self.pos:self.pos + n]; self.pos += n; return v
        if t in (9, 10):
            h = self.byte(); size, et = h >> 4, h & 15
            if size == 15:
                size = self.varint()
            if et in (1, 2):
                return [self.byte() == 1 for _ in range(size)]
            return [self.value(et) for _ in range(size)]
        if t == 11:
            size = self.varint()
            if not size:
                return {}
            h = self.byte(); kt, vt = h >> 4, h & 15
            return {self.value(kt): self.value(vt) for _ in range(size)}
        if t == 12:
            return self.struct()
        raise ValueError(t)

    def struct(self):
        d, last = {}, 0
        while True:
            h = self.byte()
            if h == 0:
                return d
            t, delta = h & 15, h >> 4
            fid = last + delta if delta else self.zz()
            last = fid
            d[fid] = self.value(t)


def rle_hybrid(b, pos, end, bw, count):
    out = []
    bytew = (bw + 7) // 8
    while len(out) < count and pos < end:
        t = Thrift(b, pos); h = t.varint(); pos = t.pos
        if h & 1:
            groups = h >> 1
            nbytes = groups * bw
            chunk = int.from_bytes(b[pos:pos + nbytes], 'little'); pos += nbytes
            mask = (1 << bw) - 1 if bw else 0
            for k in range(groups * 8):
                out.append((chunk >> (k * bw)) & mask if bw else 0)
        else:
            run = h >> 1
            v = int.from_bytes(b[pos:pos + bytew], 'little') if bytew else 0; pos += bytew
            out += [v] * run
    return out[:count], pos


def plain(b, pos, ptype, n):
    if ptype == 0:
        bits = int.from_bytes(b[pos:pos + (n + 7) // 8], 'little')
        return [bool((bits >> k) & 1) for k in range(n)]
    if ptype == 1:
        return list(struct.unpack_from(f'<{n}i', b, pos))
    if ptype == 2:
        return list(struct.unpack_from(f'<{n}q', b, pos))
    if ptype == 4:
        return list(struct.unpack_from(f'<{n}f', b, pos))
    if ptype == 5:
        return list(struct.unpack_from(f'<{n}d', b, pos))
    if ptype == 6:
        out = []
        for _ in range(n):
            ln = struct.unpack_from('<i', b, pos)[0]; pos += 4
            out.append(b[pos:pos + ln].decode('utf-8')); pos += ln
        return out
    raise ValueError(ptype)


def read(path):
    b = open(path, 'rb').read()
    assert b[:4] == b'PAR1' and b[-4:] == b'PAR1'
    n = struct.unpack('<i', b[-8:-4])[0]
    meta = Thrift(b, len(b) - 8 - n).struct()
    schema = meta[2]
    # flat leaves with their repetition (0 required, 1 optional, 2 repeated) and whether nested
    leaves, stack = {}, []
    i = 1
    def walk(i, path, nested, maxdef):
        el = schema[i]
        name = el[4].decode()
        rep = el.get(3, 0)
        md = maxdef + (1 if rep in (1, 2) else 0)
        nch = el.get(5, 0)
        p = path + [name]
        if nch:
            j = i + 1
            for _ in range(nch):
                j = walk(j, p, True, md)
            return j
        leaves['.'.join(p)] = dict(ptype=el.get(1), maxdef=md, nested=nested or rep == 2)
        return i + 1
    j = 1
    for _ in range(schema[0].get(5, 0)):
        j = walk(j, [], False, 0)
    cols = {}
    for rg in meta[4]:
        for cc in rg[1]:
            md = cc[3]
            name = '.'.join(x.decode() for x in md[3])
            info = leaves[name]
            if info['nested']:
                continue
            codec, nvals = md[4], md[5]
            pos = md.get(11, md[9])
            if 11 in md and md[11] is not None:
                pos = min(md[11], md[9])
            end = pos + md[7]
            dictionary, values = None, []
            while pos < end and len(values) < nvals:
                t = Thrift(b, pos); ph = t.struct(); pos = t.pos
                raw = b[pos:pos + ph[3]]; pos += ph[3]
                ptype = ph[1]
                if ptype == 3:   # data page v2: levels uncompressed
                    h = ph[8]
                    dl, rl = h[5], h[6]
                    lv = raw[:rl + dl]
                    body = raw[rl + dl:]
                    if h.get(7, True) and codec == 1:
                        body = snappy_decompress(body)
                    data = lv + body
                    nv = h[1]
                    defs = rle_hybrid(data, rl, rl + dl, 1, nv)[0] if info['maxdef'] else [1] * nv
                    dpos = rl + dl
                    enc = h[4]
                else:
                    data = snappy_decompress(raw) if codec == 1 else raw
                    assert codec in (0, 1), codec
                if ptype == 2:
                    dictionary = plain(data, 0, info['ptype'], ph[7][1])
                    continue
                if ptype == 0:
                    h = ph[5]; nv = h[1]; enc = h[2]; dpos = 0
                    if info['maxdef']:
                        ln = struct.unpack_from('<i', data, 0)[0]
                        defs = rle_hybrid(data, 4, 4 + ln, 1, nv)[0]
                        dpos = 4 + ln
                    else:
                        defs = [1] * nv
                nn = sum(1 for d in defs if d == info['maxdef'])
                if enc in (2, 8):
                    bw = data[dpos]
                    idx = rle_hybrid(data, dpos + 1, len(data), bw, nn)[0]
                    vals = [dictionary[k] for k in idx]
                elif enc == 0:
                    vals = plain(data, dpos, info['ptype'], nn)
                elif enc == 3 and info['ptype'] == 0:
                    ln = struct.unpack_from('<i', data, dpos)[0]
                    vals = [bool(v) for v in rle_hybrid(data, dpos + 4, dpos + 4 + ln, 1, nn)[0]]
                else:
                    raise ValueError(('encoding', enc, name))
                it = iter(vals)
                values += [next(it) if d == info['maxdef'] else None for d in defs]
            cols.setdefault(name, []).extend(values)
    return cols
