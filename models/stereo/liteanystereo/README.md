# LiteAnyStereo pretrained checkpoint inventory

Downloaded 2026-09-27.  All eight available pretrained checkpoints are kept in
this directory.  They are model weights, not source code; do not commit them.

## LAS2 — direct official release

These files came directly from the authors' [LiteAnyStereoV2 Hugging Face
repository](https://huggingface.co/tomtomtommi/LiteAnyStereoV2), which lists
the S, M, L, and H release models.  The model card states an MIT license.

| Model | File | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| LAS2-S | `LAS2_S.pth` | 24,380,949 | `dda5ad5495710466f7e5c5a2dff395223893ef68a17d57d88e7ea7dfbb9d08e1` |
| LAS2-M | `LAS2_M.pth` | 41,925,933 | `c41b61efce9b36e6b8b38b580421cd20cc9e84a47f0f45361767281938f69696` |
| LAS2-L | `LAS2_L.pth` | 109,270,797 | `12f86f758b0aa6cd59c11d40f41e7d954121d7626e4719eff71d9dd22da36539` |
| LAS2-H | `LAS2_H.pth` | 46,698,761 | `758585a25c3a332711f92a28ad1437e08080fb714ad1146de7cf2c01ce8479f4` |

## LAS1 — public checkpoint mirror

The authors' [official repository](https://github.com/TomTomTommi/LiteAnyStereo)
documents LAS1 and links its downloads through an author-provided Google Drive
folder, but does not publish LAS1 on Hugging Face.  These four files came from
the public [Miayan/stereo-matching-weights](https://huggingface.co/Miayan/stereo-matching-weights)
mirror.  Their hashes match the mirror's published LFS object identifiers.

| Training stage | File | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| Final LAS1 | `LiteAnyStereo.pth` | 30,775,410 | `ee0c3a0dc1d4b49cbd67edf00079b9993c0fa21f6c19a0eb812fa32f7ec1b9b1` |
| MIX stage 1 | `LiteAnyStereo_MIX_Stage1.pth` | 30,776,094 | `da8d113b16d13b5b27ea93660d41e7dfa3e4f5cc2335b3ec545e79e2d5124afb` |
| MIX stage 2 | `LiteAnyStereo_MIX_Stage2.pth` | 30,775,410 | `f26d4b3733e5dc17b8b493566bf174e53735a8cb40794fe35687eb4fb17c4500` |
| SceneFlow stage 2 | `LiteAnyStereo_SF_Stage2.pth` | 30,775,410 | `5e4c7663047ef60ace14fc2d65c1bc70ab9c19ac8b4b041c7a44e13f878f0892` |

LAS1 and LAS2 use different code paths.  Use the authors' current repository
and pair LAS2-S/M/L/H with the matching `--model_size`; do not load LAS1
weights into LAS2 code or vice versa.
