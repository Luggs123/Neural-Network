import cern.colt.function.DoubleDoubleFunction;
import cern.colt.function.DoubleFunction;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.linalg.*;
import cern.jet.random.Normal;
import cern.jet.random.engine.DRand;
import java.math.*;
import java.util.ArrayList;

public class Network {
	//Storage of info: layers, and matrices of values, biases, and weights.
	int numLayers;
	int[] sizes;
	ArrayList<DenseDoubleMatrix2D> values;
	ArrayList<DenseDoubleMatrix2D> biases;
	ArrayList<DenseDoubleMatrix2D> weights;
	
	//Storage and random generation for the Normal Distribution.
	Normal normalDist = new Normal(0, 1, new DRand());
	DoubleFunction randDouble = normalDist;
	
	public Network(int[] sizes) {
		//Initializes the matrices and values, generating random biases and weights.
		this.sizes = sizes;
		this.numLayers = this.sizes.length;
		
		for (int i = 0; i < this.numLayers; i++) {
			this.values.add(new DenseDoubleMatrix2D(sizes[i], 1));
		}
		
		for (int i = 0; i < this.numLayers - 1; i++) {
			this.biases.add(new DenseDoubleMatrix2D(sizes[i+1], 1));
			this.biases.get(i).assign(randDouble);
			
			this.weights.add(new DenseDoubleMatrix2D(sizes[i], sizes[i + 1]));
			this.weights.get(i).assign(randDouble);
		}
	}
	
	private void feedForward() {
		for (int i = 0; i < this.numLayers - 2; i++) {
			DenseDoubleMatrix2D currentValues = this.values.get(i);
			DenseDoubleMatrix2D currentWeights = this.weights.get(i);
			DenseDoubleMatrix2D currentBiases = this.biases.get(i);
			DenseDoubleMatrix2D targetValues = this.values.get(1 + 1);
			
			//Multiplies current weight and values to store in target values. Normalizes, then assigns to next value matrix.
			targetValues.assign(currentWeights.zMult(currentValues, targetValues));
			targetValues = addMatrices(targetValues, currentBiases);
			sigmoid(targetValues);
			this.values.set(i + 1, targetValues);
		}
	}
	
	//Since apparently Colt doesn't have a function for adding matrices.
	static DenseDoubleMatrix2D addMatrices(DenseDoubleMatrix2D mat1, DenseDoubleMatrix2D mat2) {
		DenseDoubleMatrix2D temp = new DenseDoubleMatrix2D(Math.max(mat1.rows(), mat2.rows()), Math.max(mat1.columns(), mat2.columns()));
		temp.assign(mat1);
		temp.assign(mat2, new Add());
		return temp;
	}
	
	//Applies the sigmoid function by elements to a matrix.
	static void sigmoid(DoubleMatrix2D matrix) {
		matrix.assign(new Sigmoid());
	}
}

class Sigmoid implements DoubleFunction {

	@Override
	public double apply(double arg0) {
		return 1/(1+Math.pow(Math.E, -arg0));
	}

}

class Add implements DoubleDoubleFunction {
	@Override
	public double apply(double arg0, double arg1) {
		return arg0 + arg1;
	}
}
