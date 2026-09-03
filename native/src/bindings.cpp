#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace chessai {

constexpr int kFiles = 9;
constexpr int kRanks = 10;
constexpr int kSquares = kFiles * kRanks;
constexpr std::string_view kPieces = "KABNRCPkabnrcp";
constexpr std::string_view kRuleVersion = "wxf-2018-computer-v1";

enum class Color { Red, Black };

Color opposite(Color color) { return color == Color::Red ? Color::Black : Color::Red; }

struct Square {
  int file{};
  int rank{};

  [[nodiscard]] int index() const { return rank * kFiles + file; }
};

struct Move {
  Square from;
  Square to;
};

[[nodiscard]] bool inside(int file, int rank) {
  return file >= 0 && file < kFiles && rank >= 0 && rank < kRanks;
}

[[nodiscard]] bool valid_piece(char piece) {
  return kPieces.find(piece) != std::string_view::npos;
}

[[nodiscard]] Color piece_color(char piece) {
  return std::isupper(static_cast<unsigned char>(piece)) ? Color::Red : Color::Black;
}

[[nodiscard]] bool belongs_to(char piece, Color color) {
  return piece != '.' && piece_color(piece) == color;
}

[[nodiscard]] bool inside_palace(Square square, Color color) {
  if (square.file < 3 || square.file > 5) return false;
  return color == Color::Red ? square.rank >= 0 && square.rank <= 2
                             : square.rank >= 7 && square.rank <= 9;
}

[[nodiscard]] std::string square_text(Square square) {
  std::string result;
  result.push_back(static_cast<char>('a' + square.file));
  result.push_back(static_cast<char>('0' + square.rank));
  return result;
}

[[nodiscard]] std::string move_text(Move move) {
  return square_text(move.from) + square_text(move.to);
}

[[nodiscard]] Move parse_move(std::string_view text) {
  if (text.size() != 4 || text[0] < 'a' || text[0] > 'i' || text[2] < 'a' ||
      text[2] > 'i' || text[1] < '0' || text[1] > '9' || text[3] < '0' ||
      text[3] > '9') {
    throw std::invalid_argument("invalid ICCS move: " + std::string(text));
  }
  Move move{{text[0] - 'a', text[1] - '0'}, {text[2] - 'a', text[3] - '0'}};
  if (move.from.index() == move.to.index()) {
    throw std::invalid_argument("a move must change squares");
  }
  return move;
}

struct Position {
  std::array<char, kSquares> board{};
  Color side_to_move{Color::Red};
  int halfmove_clock{0};
  int fullmove_number{1};

  static Position from_compact(std::string_view pieces, Color side) {
    if (pieces.size() != kSquares) {
      throw std::invalid_argument("compact board must contain 90 squares");
    }
    Position position;
    position.side_to_move = side;
    for (int index = 0; index < kSquares; ++index) {
      const char piece = pieces[static_cast<std::size_t>(index)];
      if (piece != '.' && !valid_piece(piece)) {
        throw std::invalid_argument("invalid compact board piece");
      }
      position.board[static_cast<std::size_t>(index)] = piece;
    }
    return position;
  }

  static Position from_fen(std::string_view fen) {
    std::istringstream input{std::string(fen)};
    std::vector<std::string> parts;
    for (std::string part; input >> part;) parts.push_back(part);
    if (parts.size() < 2) throw std::invalid_argument("FEN needs board and side-to-move");

    Position position;
    position.board.fill('.');
    std::vector<std::string> rows;
    std::stringstream rows_stream(parts[0]);
    for (std::string row; std::getline(rows_stream, row, '/');) rows.push_back(row);
    if (rows.size() != kRanks) throw std::invalid_argument("Xiangqi FEN needs 10 ranks");

    for (int fen_row = 0; fen_row < kRanks; ++fen_row) {
      const int rank = 9 - fen_row;
      int file = 0;
      for (const char token : rows[fen_row]) {
        if (token >= '1' && token <= '9') {
          file += token - '0';
        } else if (valid_piece(token)) {
          if (file >= kFiles) throw std::invalid_argument("too many squares in FEN rank");
          position.board[rank * kFiles + file] = token;
          ++file;
        } else {
          throw std::invalid_argument("invalid FEN token");
        }
      }
      if (file != kFiles) throw std::invalid_argument("FEN rank does not expand to 9 files");
    }

    if (parts[1] == "w") {
      position.side_to_move = Color::Red;
    } else if (parts[1] == "b") {
      position.side_to_move = Color::Black;
    } else {
      throw std::invalid_argument("invalid FEN side-to-move");
    }
    if (parts.size() >= 5) position.halfmove_clock = std::stoi(parts[4]);
    if (parts.size() >= 6) position.fullmove_number = std::stoi(parts[5]);
    if (position.halfmove_clock < 0 || position.fullmove_number < 1) {
      throw std::invalid_argument("invalid FEN clocks");
    }
    return position;
  }

  [[nodiscard]] std::string to_fen() const {
    std::vector<std::string> rows;
    rows.reserve(kRanks);
    for (int rank = 9; rank >= 0; --rank) {
      std::string row;
      int empty = 0;
      for (int file = 0; file < kFiles; ++file) {
        const char piece = board[rank * kFiles + file];
        if (piece == '.') {
          ++empty;
        } else {
          if (empty) {
            row += std::to_string(empty);
            empty = 0;
          }
          row.push_back(piece);
        }
      }
      if (empty) row += std::to_string(empty);
      rows.push_back(row);
    }
    std::ostringstream output;
    for (std::size_t index = 0; index < rows.size(); ++index) {
      if (index) output << '/';
      output << rows[index];
    }
    output << (side_to_move == Color::Red ? " w - - " : " b - - ") << halfmove_clock
           << ' ' << fullmove_number;
    return output.str();
  }

  [[nodiscard]] std::string position_key() const {
    return std::string(board.begin(), board.end()) +
           (side_to_move == Color::Red ? "w" : "b");
  }

  [[nodiscard]] std::optional<Square> general(Color color) const {
    const char target = color == Color::Red ? 'K' : 'k';
    for (int index = 0; index < kSquares; ++index) {
      if (board[index] == target) return Square{index % kFiles, index / kFiles};
    }
    return std::nullopt;
  }

  [[nodiscard]] int line_blockers(Square origin, Square target) const {
    if (origin.file != target.file && origin.rank != target.rank) return -1;
    const int df = origin.file == target.file ? 0 : (target.file > origin.file ? 1 : -1);
    const int dr = origin.rank == target.rank ? 0 : (target.rank > origin.rank ? 1 : -1);
    int file = origin.file + df;
    int rank = origin.rank + dr;
    int blockers = 0;
    while (file != target.file || rank != target.rank) {
      if (board[rank * kFiles + file] != '.') ++blockers;
      file += df;
      rank += dr;
    }
    return blockers;
  }

  [[nodiscard]] bool piece_attacks(Square origin, Square target, char piece) const {
    const Color color = piece_color(piece);
    const char kind = static_cast<char>(std::toupper(static_cast<unsigned char>(piece)));
    const int df = target.file - origin.file;
    const int dr = target.rank - origin.rank;
    if (kind == 'R') return line_blockers(origin, target) == 0;
    if (kind == 'C') return line_blockers(origin, target) == 1;
    if (kind == 'N') {
      if (!((std::abs(df) == 1 && std::abs(dr) == 2) ||
            (std::abs(df) == 2 && std::abs(dr) == 1))) {
        return false;
      }
      const Square leg = std::abs(dr) == 2
                             ? Square{origin.file, origin.rank + (dr > 0 ? 1 : -1)}
                             : Square{origin.file + (df > 0 ? 1 : -1), origin.rank};
      return board[leg.index()] == '.';
    }
    if (kind == 'B') {
      if (std::abs(df) != 2 || std::abs(dr) != 2) return false;
      if (color == Color::Red && target.rank > 4) return false;
      if (color == Color::Black && target.rank < 5) return false;
      const Square eye{origin.file + df / 2, origin.rank + dr / 2};
      return board[eye.index()] == '.';
    }
    if (kind == 'A') {
      return std::abs(df) == 1 && std::abs(dr) == 1 && inside_palace(target, color);
    }
    if (kind == 'K') {
      if (std::abs(df) + std::abs(dr) == 1 && inside_palace(target, color)) return true;
      const char enemy_general = color == Color::Red ? 'k' : 'K';
      return df == 0 && board[target.index()] == enemy_general &&
             line_blockers(origin, target) == 0;
    }
    if (kind == 'P') {
      const int forward = color == Color::Red ? 1 : -1;
      if (df == 0 && dr == forward) return true;
      const bool crossed = color == Color::Red ? origin.rank >= 5 : origin.rank <= 4;
      return crossed && dr == 0 && std::abs(df) == 1;
    }
    throw std::logic_error("unknown piece");
  }

  [[nodiscard]] bool square_attacked(Square target, Color by_color) const {
    for (int index = 0; index < kSquares; ++index) {
      const char piece = board[index];
      if (piece == '.' || !belongs_to(piece, by_color)) continue;
      const Square origin{index % kFiles, index / kFiles};
      if (piece_attacks(origin, target, piece)) return true;
    }
    return false;
  }

  [[nodiscard]] bool in_check(Color color) const {
    const auto square = general(color);
    return !square.has_value() || square_attacked(*square, opposite(color));
  }

  void add_step_moves(std::vector<Move>& moves, Square origin, char piece,
                      const std::vector<std::pair<int, int>>& offsets) const {
    const Color color = piece_color(piece);
    for (const auto& [df, dr] : offsets) {
      const int file = origin.file + df;
      const int rank = origin.rank + dr;
      if (!inside(file, rank)) continue;
      const Square target{file, rank};
      const char occupant = board[target.index()];
      if (occupant != '.' && belongs_to(occupant, color)) continue;
      if (piece_attacks(origin, target, piece)) moves.push_back({origin, target});
    }
  }

  [[nodiscard]] std::vector<Move> pseudo_legal(Color mover) const {
    std::vector<Move> moves;
    for (int index = 0; index < kSquares; ++index) {
      const char piece = board[index];
      if (piece == '.' || !belongs_to(piece, mover)) continue;
      const Square origin{index % kFiles, index / kFiles};
      const Color color = piece_color(piece);
      const char kind = static_cast<char>(std::toupper(static_cast<unsigned char>(piece)));
      if (kind == 'R' || kind == 'C') {
        for (const auto& [df, dr] : std::array<std::pair<int, int>, 4>{
                 std::pair{1, 0}, std::pair{-1, 0}, std::pair{0, 1}, std::pair{0, -1}}) {
          int file = origin.file + df;
          int rank = origin.rank + dr;
          bool screened = false;
          while (inside(file, rank)) {
            const Square target{file, rank};
            const char occupant = board[target.index()];
            if (kind == 'R') {
              if (occupant == '.') {
                moves.push_back({origin, target});
              } else {
                if (!belongs_to(occupant, color)) moves.push_back({origin, target});
                break;
              }
            } else if (!screened) {
              if (occupant == '.') {
                moves.push_back({origin, target});
              } else {
                screened = true;
              }
            } else if (occupant != '.') {
              if (!belongs_to(occupant, color)) moves.push_back({origin, target});
              break;
            }
            file += df;
            rank += dr;
          }
          }
        continue;
      }

      std::vector<std::pair<int, int>> offsets;
      if (kind == 'N') {
        offsets = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
      } else if (kind == 'B') {
        offsets = {{2, 2}, {2, -2}, {-2, -2}, {-2, 2}};
      } else if (kind == 'A') {
        offsets = {{1, 1}, {1, -1}, {-1, -1}, {-1, 1}};
      } else if (kind == 'K') {
        offsets = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
      } else if (kind == 'P') {
        const int forward = color == Color::Red ? 1 : -1;
        const bool crossed = color == Color::Red ? origin.rank >= 5 : origin.rank <= 4;
        offsets = crossed ? std::vector<std::pair<int, int>>{{0, forward}, {1, 0}, {-1, 0}}
                          : std::vector<std::pair<int, int>>{{0, forward}};
      } else {
        throw std::logic_error("unknown piece");
      }
      add_step_moves(moves, origin, piece, offsets);

      if (kind == 'K') {
        const auto enemy = general(opposite(color));
        if (enemy.has_value() && enemy->file == origin.file && line_blockers(origin, *enemy) == 0) {
          moves.push_back({origin, *enemy});
        }
      }
    }
    return moves;
  }

  [[nodiscard]] Position apply_unchecked(Move move) const {
    Position child = *this;
    const char piece = child.board[move.from.index()];
    const char captured = child.board[move.to.index()];
    if (piece == '.') throw std::invalid_argument("move origin is empty");
    child.board[move.to.index()] = piece;
    child.board[move.from.index()] = '.';
    child.side_to_move = opposite(side_to_move);
    child.halfmove_clock = captured == '.' ? halfmove_clock + 1 : 0;
    child.fullmove_number = fullmove_number + (side_to_move == Color::Black ? 1 : 0);
    return child;
  }

  [[nodiscard]] std::vector<Move> legal_moves() const {
    // This is the ordinary move-generation primitive. History-dependent and
    // experiment-clock adjudication is owned by Python before it calls here.
    if (!general(Color::Red).has_value() || !general(Color::Black).has_value()) {
      return {};
    }
    std::vector<Move> moves;
    for (const Move move : pseudo_legal(side_to_move)) {
      const Position child = apply_unchecked(move);
      if (child.general(side_to_move).has_value() && !child.in_check(side_to_move)) {
        moves.push_back(move);
      }
    }
    return moves;
  }

  [[nodiscard]] Position apply_legal(Move move) const {
    const std::string requested = move_text(move);
    const auto moves = legal_moves();
    const auto found = std::find_if(moves.begin(), moves.end(), [&](Move candidate) {
      return move_text(candidate) == requested;
    });
    if (found == moves.end()) throw std::invalid_argument("illegal move " + requested);
    return apply_unchecked(move);
  }
};

[[nodiscard]] std::vector<std::string> legal_move_strings(std::string_view fen) {
  const Position position = Position::from_fen(fen);
  std::vector<std::string> result;
  for (const Move move : position.legal_moves()) result.push_back(move_text(move));
  return result;
}

[[nodiscard]] std::vector<std::uint16_t> legal_move_codes(std::string_view fen) {
  const Position position = Position::from_fen(fen);
  std::vector<std::uint16_t> result;
  for (const Move move : position.legal_moves()) {
    result.push_back(static_cast<std::uint16_t>(move.from.index() * kSquares + move.to.index()));
  }
  return result;
}

[[nodiscard]] std::uint64_t perft_position(const Position& position, int depth) {
  if (depth == 0) return 1;
  std::uint64_t nodes = 0;
  for (const Move move : position.legal_moves()) {
    nodes += perft_position(position.apply_unchecked(move), depth - 1);
  }
  return nodes;
}

}  // namespace chessai

PYBIND11_MODULE(_chessai_native, module) {
  module.doc() = "C++20 Xiangqi move-generation accelerator for ChessAI";
  module.attr("RULE_VERSION") = std::string(chessai::kRuleVersion);
  module.def("legal_moves", &chessai::legal_move_strings, py::arg("fen"),
             py::call_guard<py::gil_scoped_release>());
  module.def("legal_move_codes", &chessai::legal_move_codes, py::arg("fen"),
             py::call_guard<py::gil_scoped_release>());
  module.def("apply_move", [](std::string_view fen, std::string_view move) {
    return chessai::Position::from_fen(fen).apply_legal(chessai::parse_move(move)).to_fen();
  }, py::arg("fen"), py::arg("move"), py::call_guard<py::gil_scoped_release>());
  module.def("is_in_check", [](std::string_view fen, std::optional<std::string> color) {
    const chessai::Position position = chessai::Position::from_fen(fen);
    chessai::Color checked = position.side_to_move;
    if (color.has_value()) {
      if (*color == "red") checked = chessai::Color::Red;
      else if (*color == "black") checked = chessai::Color::Black;
      else throw std::invalid_argument("color must be red or black");
    }
    return position.in_check(checked);
  }, py::arg("fen"), py::arg("color") = py::none(),
  py::call_guard<py::gil_scoped_release>());
  module.def("is_in_check_board", [](std::string_view board, std::string_view color) {
    chessai::Color checked;
    if (color == "red") checked = chessai::Color::Red;
    else if (color == "black") checked = chessai::Color::Black;
    else throw std::invalid_argument("color must be red or black");
    return chessai::Position::from_compact(board, checked).in_check(checked);
  }, py::arg("board"), py::arg("color"), py::call_guard<py::gil_scoped_release>());
  module.def("position_key", [](std::string_view fen) {
    return chessai::Position::from_fen(fen).position_key();
  }, py::arg("fen"), py::call_guard<py::gil_scoped_release>());
  module.def("perft", [](std::string_view fen, int depth) {
    if (depth < 0 || depth > 8) throw std::invalid_argument("perft depth must be in [0, 8]");
    return chessai::perft_position(chessai::Position::from_fen(fen), depth);
  }, py::arg("fen"), py::arg("depth"), py::call_guard<py::gil_scoped_release>());
}
