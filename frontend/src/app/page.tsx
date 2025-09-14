"use client";
import React, { useState, useEffect, useCallback } from "react";
import Header from "./components/header";
import GameBoard from "./components/gameBoard";
import GameOverOverlay from "./components/gameOverLay";
import { getEmptyBoard, addRandomTile, moveBoard, isGameOver } from "./game";

const App: React.FC = () => {
  const [board, setBoard] = useState<number[][]>(getEmptyBoard());
  const [score, setScore] = useState<number>(0);
  const [bestScore, setBestScore] = useState<number>(0);
  const [gameOver, setGameOver] = useState<boolean>(false);

  const initializeGame = useCallback(() => {
    let newBoard = getEmptyBoard();
    newBoard = addRandomTile(newBoard);
    newBoard = addRandomTile(newBoard);
    setBoard(newBoard);
    setScore(0);
    setGameOver(false);
  }, []);

  useEffect(() => {
    const savedBestScore = localStorage.getItem("bestScore");
    if (savedBestScore) setBestScore(parseInt(savedBestScore));
    initializeGame();
  }, [initializeGame]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (gameOver) return;
      let direction = "";
      switch (event.key) {
        case "ArrowUp":
          direction = "up";
          break;
        case "ArrowDown":
          direction = "down";
          break;
        case "ArrowLeft":
          direction = "left";
          break;
        case "ArrowRight":
          direction = "right";
          break;
        default:
          return;
      }

      const [newBoard, scoreGained] = moveBoard(board, direction);
      if (JSON.stringify(newBoard) !== JSON.stringify(board)) {
        const updatedBoard = addRandomTile(newBoard);
        setBoard(updatedBoard);
        const newScore = score + scoreGained;
        setScore(newScore);
        if (newScore > bestScore) {
          setBestScore(newScore);
          localStorage.setItem("bestScore", newScore.toString());
        }
        if (isGameOver(updatedBoard)) setGameOver(true);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [board, score, bestScore, gameOver]);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 px-4">
      <Header score={score} bestScore={bestScore} />
      <GameBoard
        board={board}
        isGameOver={gameOver}
        onRetry={initializeGame}
        Overlay={<GameOverOverlay onRetry={initializeGame} />}
      />
      <button
        onClick={initializeGame}
        className="mt-4 md:mt-6 px-4 md:px-6 py-2 md:py-3 bg-green-500 text-white rounded-md hover:bg-green-600"
      >
        New Game
      </button>
    </div>
  );
};

export default App;
