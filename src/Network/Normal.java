package Network;

import org.apache.commons.math3.distribution.NormalDistribution;

import Matrices.MatrixFunction;

public class Normal extends MatrixFunction {

	public Normal() {
		
	}
	
	@Override
	public double apply(int entryRow, int entryCol, int matrixRows, int matrixCols, double currentVal) {
		NormalDistribution normal = new NormalDistribution();
		
		return normal.sample();
	}

}
