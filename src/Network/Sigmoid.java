package Network;

import Matrices.MatrixFunction;

public class Sigmoid extends MatrixFunction {

	public Sigmoid() {
		
	}
	
	@Override
	public double apply(int entryRow, int entryCol, int matrixRows, int matrixCols, double currentVal) {
		return 1 / (1 + Math.pow(Math.E, -currentVal));
	}

}