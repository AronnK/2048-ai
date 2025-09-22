import * as ort from "onnxruntime-web";

export async function getBestMove(board: number[][], validMoves: boolean[]): Promise<number> {
  try {
    const session = await ort.InferenceSession.create("/model.onnx");

    const logBoard = board.map(row => 
      row.map(val => (val > 0 ? Math.log2(val) : 0))
    );
    
    const flattenedBoard = new Float32Array(logBoard.flat());
    const tensor = new ort.Tensor("float32", flattenedBoard, [1, 1, 4, 4]);

    const feeds = { "input": tensor };
    const results = await session.run(feeds);
    
    const qValues = results.output.data as Float32Array;

    let bestAction = -1;
    let maxQValue = -Infinity;

    for (let i = 0; i < qValues.length; i++) {
      if (validMoves[i] && qValues[i] > maxQValue) {
        maxQValue = qValues[i];
        bestAction = i;
      }
    }
    
    return bestAction;

  } catch (e) {
    console.error(`Failed to run AI inference: ${e}`);
    const validIndices = validMoves.map((v, i) => v ? i : -1).filter(i => i !== -1);
    return validIndices[Math.floor(Math.random() * validIndices.length)];
  }
}
