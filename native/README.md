# Optional C++20 rules backend

The Python engine is the correctness oracle. This pybind11 module implements
FEN parsing, legal-move generation, king-safety filtering, move application,
position keys, check detection, and perft in C++20 for the self-play hot path.
Repetition/long-check/long-chase adjudication remains history-aware in Python;
the native module never silently invents missing history from a bare FEN.

Build on Ubuntu/WSL2 or a cloud image:

```bash
uv sync --extra native
PYBIND11_DIR="$(uv run python -m pybind11 --cmakedir)"
cmake -S native -B native/build -G Ninja \
  -Dpybind11_DIR="$PYBIND11_DIR" \
  -DPython_EXECUTABLE="$(uv run python -c 'import sys; print(sys.executable)')"
cmake --build native/build --config Release
uv run python -c "import _chessai_native as n; print(n.perft('rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1', 3))"
```

The expected initial perft values are 44, 1,920, and 79,666 for depths 1–3.
