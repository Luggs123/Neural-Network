package Matrices;

/**
 * An interface for creating functions to be applied based on: current entry value, entry position, and matrix size
 */
public abstract class MatrixFunction {
	
	public abstract double apply(int entryRow, int entryCol, int matrixRows, int matrixCols, double currentVal);
}
