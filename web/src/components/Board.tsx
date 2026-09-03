import { useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import type { Color, Piece } from "../types/api";

const FILES = "abcdefghi";
const RANKS = Array.from({ length: 10 }, (_, index) => index);
const MARGIN_X = 60;
const MARGIN_Y = 60;
const CELL = 80;

interface BoardProps {
  pieces: Piece[];
  legalMoves: string[];
  lastMove: string | null;
  selected: string | null;
  flipped: boolean;
  interactiveColor: Color;
  disabled?: boolean;
  inCheck?: boolean;
  onSquare: (square: string) => void;
  onMove: (move: string) => void;
}

function coordinates(square: string, flipped: boolean): { x: number; y: number } {
  const file = FILES.indexOf(square[0]);
  const rank = Number(square[1]);
  const displayFile = flipped ? 8 - file : file;
  const displayRank = flipped ? rank : 9 - rank;
  return { x: MARGIN_X + displayFile * CELL, y: MARGIN_Y + displayRank * CELL };
}

function squareAt(displayFile: number, displayRank: number, flipped: boolean): string {
  const file = flipped ? 8 - displayFile : displayFile;
  const rank = flipped ? displayRank : 9 - displayRank;
  return `${FILES[file]}${rank}`;
}

function squareDescription(square: string): string {
  return `${square[0].toUpperCase()} 线，第 ${Number(square[1]) + 1} 横线`;
}

export function Board({
  pieces,
  legalMoves,
  lastMove,
  selected,
  flipped,
  interactiveColor,
  disabled = false,
  inCheck = false,
  onSquare,
  onMove,
}: BoardProps) {
  const [dragFrom, setDragFrom] = useState<string | null>(null);
  const movedByDrag = useRef(false);
  const legalDestinations = new Set(
    selected ? legalMoves.filter((move) => move.startsWith(selected)).map((move) => move.slice(2)) : [],
  );
  const lastSquares = new Set(lastMove ? [lastMove.slice(0, 2), lastMove.slice(2)] : []);
  const pieceBySquare = new Map(pieces.map((piece) => [piece.square, piece]));

  const finishDrag = (to: string) => {
    if (dragFrom && dragFrom !== to && legalMoves.includes(`${dragFrom}${to}`)) {
      movedByDrag.current = true;
      onMove(`${dragFrom}${to}`);
    }
    setDragFrom(null);
  };

  const activate = (square: string) => {
    if (movedByDrag.current) {
      movedByDrag.current = false;
      return;
    }
    onSquare(square);
  };

  const onKey = (event: KeyboardEvent<SVGElement>, square: string) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      activate(square);
    }
  };

  const startDrag = (event: PointerEvent<SVGGElement>, piece: Piece) => {
    if (disabled || piece.color !== interactiveColor) return;
    event.preventDefault();
    setDragFrom(piece.square);
  };

  return (
    <div className="board-shell" data-flipped={flipped}>
      <div className="board-caption" aria-live="polite">
        <span>{flipped ? "黑方视角" : "红方视角"}</span>
        <span className="board-caption-rule" />
        <span>{inCheck ? "将军" : disabled ? "静候对手" : "轮到你落子"}</span>
      </div>
      <svg
        className="xiangqi-board"
        viewBox="0 0 760 840"
        role="group"
        aria-label={`中国象棋棋盘，${flipped ? "黑方" : "红方"}在下`}
        onPointerLeave={() => setDragFrom(null)}
      >
        <defs>
          <filter id="piece-shadow" x="-30%" y="-30%" width="160%" height="170%">
            <feDropShadow dx="0" dy="4" stdDeviation="4" floodColor="#2a2117" floodOpacity=".23" />
          </filter>
          <radialGradient id="piece-paper" cx="36%" cy="28%" r="72%">
            <stop offset="0" stopColor="#fffdf5" />
            <stop offset="1" stopColor="#ded2ba" />
          </radialGradient>
          <pattern id="paper-grain" width="48" height="48" patternUnits="userSpaceOnUse">
            <circle cx="7" cy="15" r=".7" fill="#554b3d" opacity=".12" />
            <circle cx="32" cy="6" r=".5" fill="#554b3d" opacity=".1" />
            <path d="M2 35 Q18 31 44 36" fill="none" stroke="#554b3d" strokeWidth=".45" opacity=".09" />
          </pattern>
        </defs>

        <rect x="18" y="16" width="724" height="808" rx="4" className="board-paper" />
        <rect x="26" y="24" width="708" height="792" fill="url(#paper-grain)" opacity=".8" />
        <rect x="44" y="44" width="672" height="752" className="board-frame" />

        {RANKS.map((displayRank) => (
          <line
            key={`rank-${displayRank}`}
            x1={MARGIN_X}
            y1={MARGIN_Y + displayRank * CELL}
            x2={MARGIN_X + 8 * CELL}
            y2={MARGIN_Y + displayRank * CELL}
            className="board-line"
          />
        ))}
        {Array.from({ length: 9 }, (_, displayFile) => {
          const x = MARGIN_X + displayFile * CELL;
          if (displayFile === 0 || displayFile === 8) {
            return <line key={`file-${displayFile}`} x1={x} y1={60} x2={x} y2={780} className="board-line" />;
          }
          return (
            <g key={`file-${displayFile}`}>
              <line x1={x} y1={60} x2={x} y2={380} className="board-line" />
              <line x1={x} y1={460} x2={x} y2={780} className="board-line" />
            </g>
          );
        })}

        <g className="palace-lines">
          <path d="M300 60 L460 220 M460 60 L300 220" />
          <path d="M300 620 L460 780 M460 620 L300 780" />
        </g>

        <g className="river-mark" aria-hidden="true">
          <text x="222" y="434" textAnchor="middle">楚 河</text>
          <circle cx="380" cy="420" r="3" />
          <text x="538" y="434" textAnchor="middle">汉 界</text>
        </g>

        {Array.from({ length: 9 }, (_, displayFile) =>
          RANKS.map((displayRank) => {
            const square = squareAt(displayFile, displayRank, flipped);
            const { x, y } = coordinates(square, flipped);
            const isLegal = legalDestinations.has(square);
            const isCapture = isLegal && pieceBySquare.has(square);
            return (
              <g key={`target-${square}`}>
                {lastSquares.has(square) && <rect x={x - 36} y={y - 36} width="72" height="72" rx="8" className="last-square" />}
                {isLegal && (isCapture ? (
                  <circle cx={x} cy={y} r="39" className="capture-target" />
                ) : (
                  <circle cx={x} cy={y} r="9" className="move-target" />
                ))}
                <circle
                  cx={x}
                  cy={y}
                  r="37"
                  className="square-hit"
                  data-testid={`square-${square}`}
                  role={isLegal ? "button" : undefined}
                  tabIndex={isLegal && !disabled ? 0 : -1}
                  aria-label={isLegal ? `走到 ${squareDescription(square)}` : undefined}
                  onClick={() => !disabled && activate(square)}
                  onKeyDown={(event) => isLegal && onKey(event, square)}
                  onPointerUp={() => !disabled && finishDrag(square)}
                />
              </g>
            );
          }),
        )}

        {pieces.map((piece) => {
          const { x, y } = coordinates(piece.square, flipped);
          const active = selected === piece.square || dragFrom === piece.square;
          const interactive = !disabled
            && piece.color === interactiveColor
            && legalMoves.some((move) => move.startsWith(piece.square));
          return (
            <g
              key={piece.square}
              className={`piece piece-${piece.color}${active ? " is-selected" : ""}`}
              transform={`translate(${x} ${y})`}
              role="button"
              tabIndex={interactive ? 0 : -1}
              aria-disabled={!interactive}
              aria-label={`${piece.color === "red" ? "红方" : "黑方"}${piece.glyph}，${squareDescription(piece.square)}`}
              data-testid={`piece-${piece.square}`}
              onKeyDown={(event) => onKey(event, piece.square)}
              onClick={() => !disabled && activate(piece.square)}
              onPointerDown={(event) => startDrag(event, piece)}
              onPointerUp={() => !disabled && finishDrag(piece.square)}
            >
              <g
                className="piece-face"
                data-testid={`piece-face-${piece.square}`}
                transform={active ? "translate(0 -3)" : "translate(0 0)"}
              >
                <circle r="35" className="piece-disc" />
                <circle r="29" className="piece-ring" />
                <text y="11" textAnchor="middle" className="piece-glyph">{piece.glyph}</text>
              </g>
            </g>
          );
        })}

        <g className="coordinate-labels" aria-hidden="true">
          {Array.from({ length: 9 }, (_, displayFile) => {
            const square = squareAt(displayFile, 9, flipped);
            return <text key={displayFile} x={60 + displayFile * 80} y="816" textAnchor="middle">{square[0]}</text>;
          })}
        </g>
      </svg>
    </div>
  );
}
