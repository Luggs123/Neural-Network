package Driver;

import java.util.ArrayList;

import Network.*;

public class NetworkDriver {
	public static void main(String args[]) {
		//TODO: Write driver class
		int[] networkSize = {728, 16, 16, 10};
		Network neuralNet = new Network(networkSize);
		
		final String LABEL_FILE = "/Users/Michael/Documents/GitHub/Neural Network/assets/t10k-labels.idx1-ubyte";
		final String IMAGE_FILE = "/Users/Michael/Documents/GitHub/Neural Network/assets/t10k-images.idx3-ubyte";
		
		ArrayList<Pair<double[], double[]>> data = MnistReader.getData(IMAGE_FILE, LABEL_FILE);
	}
}
