package Network;
import cern.colt.function.DoubleDoubleFunction;
import cern.colt.function.DoubleFunction;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.linalg.*;
import java.math.*;
import java.util.ArrayList;
import org.apache.commons.math3.distribution.NormalDistribution;
import Matrices.*;

public class Network {
	//Storage of info: layers, and matrices of values, biases, and weights.
	int numLayers;
	int[] sizes;
	ArrayList<Matrix> values;
	ArrayList<Matrix> biases;
	ArrayList<Matrix> weights;
	
	public Network(int[] sizes) {
		//Initializes the matrices and values, generating random biases and weights.
		this.sizes = sizes;
		this.numLayers = this.sizes.length;
		
		for (int i = 0; i < this.numLayers; i++) {
			this.values.add(new Matrix(sizes[i], 1));
		}
		
		for (int i = 0; i < this.numLayers - 1; i++) {
			this.biases.add(new Matrix(sizes[i+1], 1));
			this.biases.set(i, Matrix.applyFunction(new Normal(), this.biases.get(i)));
			
			this.weights.add(new Matrix(sizes[i], sizes[i + 1]));
			this.weights.set(i, Matrix.applyFunction(new Normal(), this.weights.get(i)));
		}
	}
	
	private void feedForward() throws MatrixMultiplicationException, MatrixAdditionException {
		//Iterates forward through layers, finding the new activations until the outputs are reached
		for (int i = 0; i < this.numLayers - 2; i++) {
			Matrix currentValues = this.values.get(i);
			Matrix currentWeights = this.weights.get(i);
			Matrix currentBiases = this.biases.get(i);
			Matrix nextValues = new Matrix(this.values.get(i + 1), false);
			
			//Multiplies current weight and values to store in target values. Normalizes, then assigns to next value matrix.
			nextValues = Matrix.matrixMultiply(currentWeights, currentValues);
			nextValues = Matrix.matrixAdd(nextValues, currentBiases);
			nextValues = Matrix.applyFunction(new Sigmoid(), nextValues);
			this.values.set(i + 1, nextValues);
		}
	}
	
	//TODO: introduce read training data to Network
	//Initializes the gradient matrices, calls backpropagation, and subtracts the gradient from the original weights and biases
	private void updateMiniBatch(ArrayList<Pair<double[], double[]>> miniBatch, double learningRate) 
			throws MatrixAdditionException, MatrixMultiplicationException, NetworkInputException, NetworkExpectationException {
		
		ArrayList<Matrix> nablaW = new ArrayList<>(weights);
		ArrayList<Matrix> nablaB = new ArrayList<>(biases);
		
		for (int i = 0; i < nablaW.size(); i++) {
			nablaW.set(i, new Matrix(nablaW.get(i), false));
		}
		
		for (int i = 0; i < nablaB.size(); i++) {
			nablaB.set(i, new Matrix(nablaB.get(i), false));
		}
		
		//This loop finds and adds the gradients to their respective matrices
		for (Pair<double[], double[]> tuple : miniBatch) {
			Pair<ArrayList<Matrix>, ArrayList<Matrix>> deltas = this.backpropagate(tuple.getFirst(), tuple.getSecond());
			
			for (int i = 0; i < this.numLayers - 1; i++) {
				nablaW.set(i, Matrix.matrixAdd(nablaW.get(i), deltas.getFirst().get(i)));
				nablaB.set(i, Matrix.matrixAdd(nablaB.get(i), deltas.getSecond().get(i)));
			}
		}
		
		for (int i = 0; i < this.numLayers - 1; i++) {
			nablaW.set(i, Matrix.scalarMultiply(- learningRate / miniBatch.size(), nablaW.get(i)));
			this.weights.set(i, Matrix.matrixAdd(this.weights.get(i), nablaW.get(i)));
			
			nablaB.set(i, Matrix.scalarMultiply(- learningRate / miniBatch.size(), nablaB.get(i)));
			this.biases.set(i, Matrix.matrixAdd(this.biases.get(i), nablaB.get(i)));
		}
	}
	
	//Backpropagation that determines the adjustments to the weights and biases according to each training example in the mini-batch
	private Pair<ArrayList<Matrix>, ArrayList<Matrix>> backpropagate(double[] data, double[] expectation) 
			throws MatrixMultiplicationException, MatrixAdditionException, NetworkInputException, NetworkExpectationException {
		
		if (data.length != this.values.get(0).getRows()) {
			throw new NetworkInputException("Input matrix wrong size");
		} else if (expectation.length != this.values.get(this.values.size() - 1).getRows()) {
			throw new NetworkExpectationException("Expectation Matrix wrong size");
		}
		
		ArrayList<Matrix> nablaW = new ArrayList<>(weights);
		ArrayList<Matrix> nablaB = new ArrayList<>(biases);
		
		for (int i = 0; i < nablaW.size(); i++) {
			nablaW.set(i, new Matrix(nablaW.get(i), false));
		}
		
		for (int i = 0; i < nablaB.size(); i++) {
			nablaB.set(i, new Matrix(nablaB.get(i), false));
		}
	
		Matrix image = new Matrix(data, true);
		
		this.values.set(0, image);
		
		this.feedForward();
		
		//TODO: perform backwards passes
	}
	
	private Matrix costDerivative(int expectation) {
		//FIXME: Change from colt to Matrix class
		Matrix output = this.values.get(values.size() - 1);
		Matrix expectationMatrix = generateExpectationMatrix(expectation, this.values.get(this.numLayers - 1).size());
		expectationMatrix.assign(new ScalarMultiply(-1));
		output.assign(expectationMatrix, new Add());
		
		return output;
	}
}