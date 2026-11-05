"""CPU checks of generated candidate constants and uint32 tape boundaries; no physics stepping."""
import unittest
from types import SimpleNamespace

import numpy as np

from nablaengine import torch_ffi
from nablaengine.g1_engine import host, step, tape
from check_pair_capacity import generated_model


class ContactEncoding(unittest.TestCase):
    def test_geom_and_metadata_boundaries(self):
        for g1, g2 in ((0, 541), (541, 0), (65535, 65535)):
            word = step.pair_geom_word(g1, g2)
            self.assertEqual((word & 65535, word >> 16), (g1, g2))
        word = step.pair_word(255, 15, 1, 1)
        self.assertEqual((word & 255, (word >> 8) & 15, (word >> 12) & 1, (word >> 13) & 1),
                         (255, 15, 1, 1))
        for geoms in ((65536, 0), (0, 65536), (-1, 0)):
            with self.assertRaises(AssertionError):
                step.pair_geom_word(*geoms)

    def test_generated_constants_and_loader(self):
        model, feet, selection, _ = generated_model("capacity15231")
        self.assertEqual((model.ngeom, selection["collision_geoms"], model.npair), (577, 542, 15231))
        packed = step.pair_constants(model, feet)
        arrays, offset = {}, 0
        layout = step.const_layout(model)
        for symbol, kind, count in layout:
            width = count * (2 if kind == "u" else 1)
            values = packed[offset:offset + width]
            arrays[symbol] = (values[::2].astype(np.uint32) | values[1::2].astype(np.uint32) << 16
                              if kind == "u" else values)
            offset += width
        self.assertEqual(offset, len(packed))
        geoms = step.collision_geoms(model)
        index = {g: i for i, g in enumerate(geoms)}
        for pair in (0, 15230, *feet):
            g1, g2 = int(model.pair_geom1[pair]), int(model.pair_geom2[pair])
            if step.GEOM_TYPE[int(model.geom_type[g1])] > step.GEOM_TYPE[int(model.geom_type[g2])]:
                g1, g2 = g2, g1
            word = int(arrays["c_pgeom"][pair])
            self.assertEqual((word & 65535, word >> 16), (index[g1], index[g2]))
            meta = int(arrays["c_pword"][pair])
            self.assertEqual((meta >> 8) & 15, model.pair_dim[pair])
            self.assertEqual((meta >> 12) & 1, int(pair in feet))
        header = step.step_header(model)
        self.assertIn("#define NGEOMC 542", header)
        self.assertIn("#define MAXCON 128", header)
        self.assertIn("#define NEFC 576", header)
        loader = host.loader([(symbol, kind, count, 1) for symbol, kind, count in layout], {})
        self.assertIn("cudaMemcpyToSymbol(c_pgeom,", loader)
        self.assertIn("cudaMemcpyToSymbol(c_pword,", loader)

    def test_tape_pair_and_slot_boundaries(self):
        model = SimpleNamespace(nq=36, nv=35, npair=65536)
        entries = [tape.pack_contact(pair, slot) for pair in (0, 15230, 65535) for slot in range(8)]
        raw = np.zeros(tape.rec_words(model), np.float32)
        words = raw.view(np.uint32)
        layout = tape.layout(model)
        words[layout["hdr"][0]] = len(entries)
        words[layout["reserved"][0]] = tape.FORMAT_MARKER
        start, length = layout["con"]
        self.assertEqual(length, 128)
        words[start:start+length] = tape.CON_NONE
        words[start:start+len(entries)] = entries
        decoded = tape.decode(raw, model)
        np.testing.assert_array_equal(decoded["pair"][:len(entries)], np.repeat([0, 15230, 65535], 8))
        np.testing.assert_array_equal(decoded["slot"][:len(entries)], np.tile(np.arange(8), 3))
        np.testing.assert_array_equal(decoded["pair"][len(entries):], -1)
        self.assertTrue(all(entry != tape.CON_NONE for entry in entries))
        for pair, slot in ((65536, 0), (-1, 0), (0, 8)):
            with self.assertRaises(ValueError):
                tape.pack_contact(pair, slot)
        words[layout["reserved"][0]] = 0x4E54020B
        with self.assertRaisesRegex(ValueError, "incompatible"):
            tape.decode(raw, model)

    def test_full_contact_and_row_header(self):
        model = SimpleNamespace(nq=36, nv=35, npair=65536)
        raw = np.zeros(tape.rec_words(model), np.float32)
        self.assertEqual(tape.rec_words(model), 1600)
        words, layout = raw.view(np.uint32), tape.layout(model)
        words[layout["reserved"][0]] = tape.FORMAT_MARKER
        words[layout["hdr"][0]] = 128 | 576 << 8 | 29 << 18 | 255 << 24
        start, length = layout["con"]
        words[start:start+length] = tape.pack_contact(65535, 7)
        decoded = tape.decode(raw, model)
        self.assertEqual(tuple(int(decoded[key]) for key in ("ncon", "nefc", "nlim", "niter")),
                         (128, 576, 29, 255))
        np.testing.assert_array_equal(decoded["pair"], np.full(128, 65535))
        np.testing.assert_array_equal(decoded["slot"], np.full(128, 7))
        with self.assertRaises(ValueError):
            step.validate_maxcon(129)

    def test_artifact_schema(self):
        self.assertEqual(torch_ffi.ARTIFACT_VERSION, 4)
        self.assertEqual(torch_ffi.TAPE_ENCODING,
                         dict(contact_bits=32, pair_bits=tape.PAIR_BITS, slot_capacity=tape.SLOT_CAPACITY, header_bits=[8, 10, 6, 8],
                              format_marker=tape.FORMAT_MARKER))


if __name__ == "__main__":
    unittest.main()
