package Network;
import cern.colt.function.DoubleDoubleFunction;
import cern.colt.function.DoubleFunction;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.linalg.*;
import cern.jet.random.Normal;
import cern.jet.random.engine.DRand;
import java.math.*;
import java.util.ArrayList;
import Matrices.*;

public class Network {
	//Storage of info: layers, and matrices of values, biases, and weights.
	int numLayers;
	int[] sizes;
	ArrayList<DenseDoubleMatrix2D> values;
	ArrayList<DenseDoubleMatrix2D> zValues;
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
			this.zValues.add(new DenseDoubleMatrix2D(sizes[i], 1));
		}
		
		for (int i = 0; i < this.numLayers - 1; i++) {
			this.biases.add(new DenseDoubleMatrix2D(sizes[i+1], 1));
			this.biases.get(i).assign(randDouble);
			
			this.weights.add(new DenseDoubleMatrix2D(sizes[i], sizes[i + 1]));
			this.weights.get(i).assign(randDouble);
		}
	}
	
	private void feedForward() {
		//Iterates forward through layers, finding the new activations until the outputs are reached
		for (int i = 0; i < this.numLayers - 2; i++) {
			DenseDoubleMatrix2D currentValues = this.values.get(i);
			DenseDoubleMatrix2D currentWeights = this.weights.get(i);
			DenseDoubleMatrix2D currentBiases = this.biases.get(i);
			DenseDoubleMatrix2D targetValues = this.values.get(i + 1);
			
			//Multiplies current weight and values to store in target values. Normalizes, then assigns to next value matrix.
			targetValues.assign(currentWeights.zMult(currentValues, targetValues));
			targetValues = addMatrices(targetValues, currentBiases);
			this.zValues.set(i + 1, targetValues);
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
	
	static void hadamardProduct(DoubleMatrix2D mat1, DoubleMatrix2D mat2) {
		mat1.assign(mat2, new Multiply());
	}
	
	//TODO: introduce read training data to Network
	//Initializes the gradient matrices, calls backpropagation, and subtracts the gradient from the original weighrs and biases
	private void updateMiniBatch(ArrayList<Pair<double[], Integer>> miniBatch, double learningRate) {
		ArrayList<DenseDoubleMatrix2D> nablaW = new ArrayList<>(weights);
		ArrayList<DenseDoubleMatrix2D> nablaB = new ArrayList<>(biases);
		
		for (int i = 0; i < nablaW.size(); i++) {
			nablaW.get(i).assign(0);
		}
		
		for (int i = 0; i < nablaB.size(); i++) {
			nablaB.get(i).assign(0);
		}
		
		//This loop finds and adds the gradients to their respective matrices
		for (Pair<double[], Integer> tuple : miniBatch) {
			Pair<ArrayList<DenseDoubleMatrix2D>, ArrayList<DenseDoubleMatrix2D>> deltas = this.backpropagate(tuple.getFirst(), tuple.getSecond());
			
			for (int i = 0; i < this.numLayers - 1; i++) {
				nablaB.get(i).assign(deltas.getSecond().get(i), new Add());
				nablaW.get(i).assign(deltas.getFirst().get(i), new Add());
			}
		}
		
		for (int i = 0; i < this.numLayers - 1; i++) {
			nablaW.get(i).assign(new ScalarMultiply(- learningRate / miniBatch.size()));
			this.weights.get(i).assign(nablaW.get(i), new Add());
			
			nablaB.get(i).assign(new ScalarMultiply(- learningRate / miniBatch.size()));
			this.biases.get(i).assign(nablaB.get(i), new Add());
		}
	}
	
	//Backpropagation that determines the adjustments to the weights and biases according to each training example in the mini-batch
	private Pair<ArrayList<DenseDoubleMatrix2D>, ArrayList<DenseDoubleMatrix2D>> backpropagate(double[] x, int y) {
		ArrayList<DenseDoubleMatrix2D> nablaW = new ArrayList<>(weights);
		ArrayList<DenseDoubleMatrix2D> nablaB = new ArrayList<>(biases);
		
		for (int i = 0; i < nablaW.size(); i++) {
			nablaW.get(i).assign(0);
		}
		
		for (int i = 0; i < nablaB.size(); i++) {
			nablaB.get(i).assign(0);
		}
	
		DenseDoubleMatrix2D image = new DenseDoubleMatrix2D(x.length, 1);
		for (int i = 0; i < x.length; i++) {
			image.setQuick(i, 0, x[i]);
		}
		
		values.set(0, image);
		
		this.feedForward();
		
		//TODO: perform backwards passes
	}
	
	private DenseDoubleMatrix2D costDerivative(int expectation) {
		DenseDoubleMatrix2D output = this.values.get(values.size() - 1);
		DenseDoubleMatrix2D expectationMatrix = generateExpectationMatrix(expectation, this.values.get(this.numLayers - 1).size());
		expectationMatrix.assign(new ScalarMultiply(-1));
		output.assign(expectationMatrix, new Add());
		
		return output;
	}
	
	private DenseDoubleMatrix2D generateExpectationMatrix(int expectation, int size) {
		DenseDoubleMatrix2D expectationMatrix = new DenseDoubleMatrix2D(size, 1);
		expectationMatrix.setQuick(expectation, 1, 1);
		return expectationMatrix;
	}
}

class Sigmoid implements DoubleFunction {

	@Override
	public double apply(double arg0) {
		return 1/(1+Math.pow(Math.E, -arg0));
	}

}

class SigmoidPrime implements DoubleFunction {

	@Override
	public double apply(double arg0) {
		return Math.pow(Math.E, arg0)/Math.pow((1 + Math.pow(Math.E, arg0)), 2);
	}
	
}

class Add implements DoubleDoubleFunction {
	@Override
	public double apply(double arg0, double arg1) {
		return arg0 + arg1;
	}
}

class Multiply implements DoubleDoubleFunction {
	@Override
	public double apply(double arg0, double arg1) {
		return arg0 * arg1;
	}
}

class ScalarMultiply implements DoubleFunction {
	
	private double scalar;

	public ScalarMultiply(double scalar) {
		this.scalar = scalar;
	}
	
	@Override
	public double apply(double arg0) {
		return scalar * arg0;
	}
	
}