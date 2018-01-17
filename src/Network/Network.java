package Network;
import java.io.*;
import java.math.*;
import java.util.ArrayList;
import org.apache.commons.math3.distribution.NormalDistribution;
import Matrices.*;

public class Network implements Serializable {
	// Storage of info: layers, and matrices of activations, biases, and weights.
	int numLayers;
	int[] sizes;
	ArrayList<Matrix> activations;
	ArrayList<Matrix> zValues;
	ArrayList<Matrix> biases;
	ArrayList<Matrix> weights;

	public Network(int[] sizes) {
		// Initializes the matrices and activations, generating random biases and weights.
		this.sizes = sizes;
		this.numLayers = this.sizes.length;

		for (int i = 0; i < this.numLayers; i++) {
			this.activations.add(new Matrix(sizes[i], 1));
			this.zValues.add(new Matrix(sizes[i], 1)); // First index of zValues left unused
		}

		for (int i = 0; i < this.numLayers - 1; i++) {
			this.biases.add(new Matrix(sizes[i + 1], 1));
			this.biases.set(i, Matrix.applyFunction(new Normal(), this.biases.get(i)));

			this.weights.add(new Matrix(sizes[i + 1], sizes[i]));
			this.weights.set(i, Matrix.applyFunction(new Normal(), this.weights.get(i)));
		}
	}

	private void feedForward() throws MatrixMultiplicationException, MatrixAdditionException {
		// Iterates forward through layers, finding the new activations until the outputs are reached
		for (int i = 0; i < this.numLayers - 2; i++) {
			Matrix currentActivations = this.activations.get(i);
			Matrix currentWeights = this.weights.get(i);
			Matrix currentBiases = this.biases.get(i);
			Matrix nextZValues = new Matrix(this.zValues.get(i + 1), false);
			Matrix nextActivations = new Matrix(this.activations.get(i + 1), false);

			// Multiplies current weight and activations to store in target activations. Normalizes, then assigns to next value matrix.
			nextZValues = Matrix.matrixMultiply(currentWeights, currentActivations);
			nextZValues = Matrix.matrixAdd(nextZValues, currentBiases);
			nextActivations = Matrix.applyFunction(new Sigmoid(), nextZValues);
			this.zValues.set(i + 1, nextZValues);
			this.activations.set(i + 1, nextActivations);
		}
	}

	// Initializes the gradient matrices, calls backpropagation, and subtracts the gradient from the original weights and biases
	private void updateMiniBatch(ArrayList<Pair<double[], double[]>> miniBatch, double learningRate) 
			throws MatrixAdditionException, MatrixMultiplicationException, NetworkInputException, NetworkExpectationException {

		// ArrayLists of gradient matrices
		ArrayList<Matrix> nablaW = new ArrayList<>(weights);
		ArrayList<Matrix> nablaB = new ArrayList<>(biases);

		for (int i = 0; i < nablaW.size(); i++) {
			nablaW.set(i, new Matrix(nablaW.get(i), false));
		}

		for (int i = 0; i < nablaB.size(); i++) {
			nablaB.set(i, new Matrix(nablaB.get(i), false));
		}

		// This loop finds and adds the gradients to their respective matrices. Most of the work is done in backpropagate()
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

	// Backpropagation that determines the adjustments to the weights and biases according to each training example in the mini-batch
	private Pair<ArrayList<Matrix>, ArrayList<Matrix>> backpropagate(double[] data, double[] expectation) 
			throws MatrixMultiplicationException, MatrixAdditionException, NetworkInputException, NetworkExpectationException {

		// Checks if the data has the correct size input
		if (data.length != this.activations.get(0).getRows()) {
			throw new NetworkInputException("Input matrix wrong size");
		} else if (expectation.length != this.activations.get(this.activations.size() - 1).getRows()) {
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

		this.activations.set(0, image);

		this.feedForward();

		// Calculates the gradient for all weights and biases using backpropagation
		double activationDerivative = 0;
		for (int i = numLayers - 1; i >= 0; i--) {
			if (i == numLayers - 1) {
				for (int j = 0; j < activations.get(i).getRows(); j++) {
					activationDerivative += 2 * (activations.get(i).getEntry(j, 0) - expectation[j]) * 
							Matrix.applyFunction(new SigmoidPrime(), zValues.get(i)).getEntry(j, 0);
				}
			} else {
				double layerActivationDerivative = 0;

				// TODO: Calculate factor of derivative for generalized case

				activationDerivative *= layerActivationDerivative;
			}

			for (int row = 0; row < this.weights.get(i).getRows(); row++) {
				for (int col = 0; col < this.weights.get(i).getCols(); col++) {
					nablaW.get(i).setEntry(row, col, activationDerivative * activations.get(i).getEntry(col, 0));
				}

				nablaB.get(i).setEntry(row, 0, activationDerivative);
			}
		}
	}

	private Matrix costDerivative(int expectation) {
		// FIXME: Change from colt to Matrix class
		Matrix output = this.activations.get(activations.size() - 1);
		Matrix expectationMatrix = generateExpectationMatrix(expectation, this.activations.get(this.numLayers - 1).size());
		expectationMatrix.assign(new ScalarMultiply(-1));
		output.assign(expectationMatrix, new Add());

		return output;
	}

	public final static void writeObject(Network x) throws IOException {
		try {
			FileOutputStream fileOut =
					new FileOutputStream("/tmp/network.ser");
			ObjectOutputStream out = new ObjectOutputStream(fileOut);
			out.writeObject(x);
			out.close();
			fileOut.close();
			System.out.printf("Serialized data is saved in /tmp/network.ser");
		} catch (IOException i) {
			i.printStackTrace();
		}
	}

	public final static Network readObject() throws IOException, ClassNotFoundException {
		Network x = null;
		try {
			FileInputStream fileIn = new FileInputStream("/tmp/network.ser");
			ObjectInputStream in = new ObjectInputStream(fileIn);
			x = (Network) in.readObject();
			in.close();
			fileIn.close();
		} catch (IOException i) {
			i.printStackTrace();
			return null;
		} catch (ClassNotFoundException c) {
			System.out.println("Employee class not found");
			c.printStackTrace();
			return null;
		}
		
		return x;
	}
}