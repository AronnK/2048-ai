"use client";
import React from "react";

interface GameOverOverlayProps {
  onRetry: () => void;
}

const GameOverOverlay: React.FC<GameOverOverlayProps> = ({ onRetry }) => {
  return (
    <div className="absolute inset-0 bg-black bg-opacity-50 flex flex-col items-center justify-center rounded-lg">
      <div className="text-white text-3xl md:text-4xl font-bold mb-4">
        Game Over
      </div>
      <button
        onClick={onRetry}
        className="bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600"
      >
        Retry
      </button>
    </div>
  );
};

export default GameOverOverlay;
