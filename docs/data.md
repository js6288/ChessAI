# Dataset acquisition and preparation

## Default source

The default optional source is the Chinese Chess Practical Dataset (CCPD). The
acquisition module pins commit
`368a47a947773dd8692c026e286dd19b6277b993`, verifies the checkout and license,
and writes `source-manifest.json` with hashes, the Git tree, selected scope, and
materialized PGN count. Downloaded files are ignored by Git.

```powershell
uv run chessai data fetch --destination data/raw/ccpd
uv run chessai data prepare --source data/raw/ccpd --destination data/processed/ccpd
uv run chessai data validate data/processed/ccpd
```

For a bounded parser smoke test, append `--limit 100` to `prepare`. The limit is
deterministic after path sorting and must never be presented as the full corpus.
`--workers 1` is the reproducible baseline; a cloud host can use
`--workers 16` or `--workers 32`. Ordered process results keep byte-identical
splits and manifests regardless of worker count (apart from the recorded worker
setting and creation timestamp).

## Acceptance pipeline

Each game is decoded by trying Big5, UTF-8, and GB18030, parsed, converted to
ICCS, and replayed through the reference rules engine. Records with decoding,
metadata, result, legality, reconstruction, or terminal-consistency failures
are rejected with a reason. Accepted games are deduplicated by normalized
initial FEN, full ICCS move list, and result. A stable whole-game hash assigns
90/5/5 train/validation/test splits so positions from one game cannot leak
between splits.

The prepared directory contains JSONL splits, a reproducible filter report, and
`file-manifest.jsonl`. That per-input ledger records path, byte size, SHA-256,
detected encoding, accepted/duplicate/rejected status, game ID/split or exact
rejection reason. The top-level manifest hashes this ledger and every split.
Tactical positions are not mixed into supervised complete-game training.

## Current full-corpus validation evidence

The pinned master-game scope was fully materialized and independently validated
on 2026-09-02:

- 53,685 PGN files read;
- 27,667 unique complete games accepted;
- 25,445 normalized duplicates removed;
- 573 records rejected by the documented strict pipeline;
- 24,878 / 1,418 / 1,371 games in train/validation/test;
- file-ledger SHA-256:
  `df9cd351b1c6280022a33cc912d41f8a1eb8da633c27980105923b5c818f0164`;
- train SHA-256:
  `dda272822b285b5ba884659b7ddb6dff27d9bebbbda3a87c813a245687a8e6e9`;
- validation SHA-256:
  `b217220814aab4d2ff3c303cf6407d635dc25c4053d25638e35bc62ec6497b24`;
- test SHA-256:
  `2046fae9fece6113c65327e8e4a1cfc2614bb4ef1cfa47dfe1a79650855372b0`.

All accepted records in this source revision decoded as Big5. The manifest,
split files, and raw checkout remain ignored rather than entering ordinary Git
history. Re-run `chessai data validate data/processed/ccpd` after transfer; the
hashes above are the handoff contract, not a substitute for validation on the
destination disk.

The 573 strict rejections reconcile to 449 notation-or-legality replay
failures, 77 records with no recognized moves, 34 missing/unknown results, and
13 declared-result versus physical-terminal conflicts. Exact file paths,
source hashes, encodings, and unabridged per-file reasons remain in
`file-manifest.jsonl`; these category counts are diagnostic data-quality facts,
not permission to silently repair upstream records.

## License failure behavior

If the pinned revision, upstream license, expected source directories, or file
integrity cannot be verified, stop. Pure self-play is the supported fallback;
do not scrape another game site automatically.
