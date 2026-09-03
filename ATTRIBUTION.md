# Third-party attribution and separation policy

ChessAI source code is distributed under the MIT License in `LICENSE`.

## Chinese Chess Practical Dataset (CCPD)

The optional data pipeline downloads the Chinese Chess Practical Dataset from
<https://github.com/Yvonne761/Chinese-Chess-Practical-Dataset> at the exact
commit recorded in `src/chessai/data/source.py`. The upstream project identifies
its data as Creative Commons Attribution 4.0. Its upstream `LICENSE`, README,
commit, file hashes, acquisition time, and processed-data manifest are retained
beside the downloaded data. Raw and processed records are excluded from Git.

If those provenance checks fail, `chessai data fetch` or `prepare` stops. The
project does not silently substitute records from a site with unclear terms.

## Pikafish

Pikafish is not copied, linked, or redistributed by this repository. Evaluation
uses a separately installed executable through its UCI protocol. Users remain
responsible for complying with the license of the exact Pikafish build they
configure.

## Fonts

The web client currently relies on system Chinese serif and sans-serif fallbacks
and therefore does not redistribute a font file. If a release later embeds a
font subset, its source, subset command, and license must be added here.
