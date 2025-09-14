"use client";
import React from "react";
import Tile from "./Tile";

interface GameBoardProps {
  board: number[][];
  isGameOver: boolean;
  onRetry: () => void;
  Overlay: React.ReactNode;
}

const GameBoard: React.FC<GameBoardProps> = ({
  board,
  isGameOver,
  Overlay,
}) => {
  return (
    <div className="relative bg-gray-800 rounded-lg p-2 grid grid-cols-4 gap-2 md:gap-4 w-full max-w-md aspect-square">
      {board.map((row, rowIndex) =>
        row.map((cell, colIndex) => (
          <div
            key={`${rowIndex}-${colIndex}`}
            className="w-full h-full bg-gray-700 rounded-md flex items-center justify-center"
          >
            <Tile value={cell} />
          </div>
        ))
      )}
      {isGameOver && Overlay}
    </div>
  );
};

export default GameBoard;
