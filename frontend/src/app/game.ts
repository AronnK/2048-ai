const BOARD_SIZE = 4;

export const getEmptyBoard = (): number[][] =>
  Array.from({ length: BOARD_SIZE }, () => Array(BOARD_SIZE).fill(0));

const getRandomEmptyCell = (board: number[][]): [number, number] | null => {
  const emptyCells: [number, number][] = [];
  board.forEach((row, i) =>
    row.forEach((cell, j) => {
      if (cell === 0) emptyCells.push([i, j]);
    })
  );
  return emptyCells.length > 0
    ? emptyCells[Math.floor(Math.random() * emptyCells.length)]
    : null;
};

export const addRandomTile = (board: number[][]): number[][] => {
  const newBoard = board.map((row) => [...row]);
  const cell = getRandomEmptyCell(newBoard);
  if (cell) {
    const [i, j] = cell;
    newBoard[i][j] = Math.random() < 0.9 ? 2 : 4;
  }
  return newBoard;
};

const transpose = (board: number[][]): number[][] =>
  board[0].map((_, colIndex) => board.map((row) => row[colIndex]));

const reverse = (board: number[][]): number[][] =>
  board.map((row) => [...row].reverse());

const slideAndCombine = (row: number[]): [number[], number] => {
  const newRow = row.filter((val) => val !== 0);
  let scoreGained = 0;
  for (let i = 0; i < newRow.length - 1; i++) {
    if (newRow[i] === newRow[i + 1]) {
      newRow[i] *= 2;
      scoreGained += newRow[i];
      newRow.splice(i + 1, 1);
    }
  }
  while (newRow.length < BOARD_SIZE) {
    newRow.push(0);
  }
  return [newRow, scoreGained];
};

export const moveBoard = (
  board: number[][],
  direction: string
): [number[][], number] => {
  let rotatedBoard: number[][];
  switch (direction) {
    case "up": rotatedBoard = transpose(board); break;
    case "down": rotatedBoard = reverse(transpose(board)); break;
    case "left": rotatedBoard = board; break;
    case "right": rotatedBoard = reverse(board); break;
    default: rotatedBoard = board;
  }

  let scoreGained = 0;
  const newBoard = rotatedBoard.map((row) => {
    const [newRow, gained] = slideAndCombine(row);
    scoreGained += gained;
    return newRow;
  });

  let restoredBoard: number[][];
  switch (direction) {
    case "up": restoredBoard = transpose(newBoard); break;
    case "down": restoredBoard = transpose(reverse(newBoard)); break;
    case "left": restoredBoard = newBoard; break;
    case "right": restoredBoard = reverse(newBoard); break;
    default: restoredBoard = newBoard;
  }

  return [restoredBoard, scoreGained];
};

export const isGameOver = (board: number[][]): boolean => {
  if (getRandomEmptyCell(board)) return false;
  for (let i = 0; i < BOARD_SIZE; i++) {
    for (let j = 0; j < BOARD_SIZE; j++) {
      const current = board[i][j];
      if (
        (i < BOARD_SIZE - 1 && current === board[i + 1][j]) ||
        (j < BOARD_SIZE - 1 && current === board[i][j + 1])
      ) {
        return false;
      }
    }
  }
  return true;
};
