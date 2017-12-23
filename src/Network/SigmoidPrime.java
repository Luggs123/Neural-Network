package Network;

import Matrices.MatrixFunction;

public class SigmoidPrime extends MatrixFunction {

	public SigmoidPrime() {
		
	}
	
	@Override
	public double apply(int entryRow, int entryCol, int matrixRows, int matrixCols, double currentVal) {
		Sigmoid s = new Sigmoid();
		return s.apply(entryRow, entryCol, matrixRows, matrixCols, currentVal) * (1 - s.apply(entryRow, entryCol, matrixRows, matrixCols, currentVal));
	}
	
}
