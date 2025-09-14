"use client";
import React from "react";

type TileValue = number;

const getTileClasses = (value: TileValue): string => {
  const baseClasses =
    "flex items-center justify-center font-bold rounded-md text-3xl md:text-4xl transition-all duration-300";
  let colorClasses = "";

  switch (value) {
    case 0:
      colorClasses = "bg-gray-700";
      break;
    case 2:
      colorClasses = "bg-gray-200 text-gray-800";
      break;
    case 4:
      colorClasses = "bg-gray-300 text-gray-800";
      break;
    case 8:
      colorClasses = "bg-orange-300 text-white";
      break;
    case 16:
      colorClasses = "bg-orange-400 text-white";
      break;
    case 32:
      colorClasses = "bg-orange-500 text-white";
      break;
    case 64:
      colorClasses = "bg-orange-600 text-white";
      break;
    case 128:
      colorClasses = "bg-yellow-400 text-white";
      break;
    case 256:
      colorClasses = "bg-yellow-500 text-white";
      break;
    case 512:
      colorClasses = "bg-yellow-600 text-white";
      break;
    case 1024:
      colorClasses = "bg-green-500 text-white";
      break;
    case 2048:
      colorClasses = "bg-green-600 text-white";
      break;
    default:
      colorClasses = "bg-red-500 text-white";
      break;
  }

  return `${baseClasses} ${colorClasses}`;
};

interface TileProps {
  value: TileValue;
}

const Tile: React.FC<TileProps> = ({ value }) => {
  const classes = getTileClasses(value);
  const scale = value !== 0 ? "scale-100" : "scale-0";

  return (
    <div className={`${classes} w-full h-full transform ${scale}`}>
      {value !== 0 ? value : ""}
    </div>
  );
};

export default Tile;
