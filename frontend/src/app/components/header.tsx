"use client";
import React from "react";

interface HeaderProps {
  score: number;
  bestScore: number;
}

const Header: React.FC<HeaderProps> = ({ score, bestScore }) => {
  return (
    <div className="flex justify-between items-center mb-4 md:mb-6">
      <h1 className="text-3xl md:text-4xl font-bold text-white">2048</h1>
      <div className="flex space-x-2 md:space-x-4">
        <div className="bg-gray-700 text-white rounded-md px-3 md:px-4 py-2 md:py-3 text-center">
          <div className="text-xs md:text-sm">SCORE</div>
          <div className="text-lg md:text-xl font-bold">{score}</div>
        </div>
        <div className="bg-gray-700 text-white rounded-md px-3 md:px-4 py-2 md:py-3 text-center">
          <div className="text-xs md:text-sm">BEST</div>
          <div className="text-lg md:text-xl font-bold">{bestScore}</div>
        </div>
      </div>
    </div>
  );
};

export default Header;
