package Matrices;

/**
 * Class to store entries and function as a Matrix
 * @author Michael Luger
 */
public class Matrix {
	/**
	 * The data stored in the matrix
	 */
	double[][] entries;
	
	/**
	 * Rows in the matrix
	 */
	int rows;
	
	/**
	 * Columns in the matrix
	 */
	int cols;


	/**
	 * Zero matrix constructor
	 * @param rows Rows in the matrix
	 * @param cols Columns in the matrix
	 */
	public Matrix(int rows, int cols) {
		entries = new double[rows][cols];
		this.rows = rows;
		this.cols = cols;

		for (int i = 0; i < this.rows; i++) {
			for (int j = 0; j < this.cols; j++) {
				this.setEntry(i, j, 0);
			}
		}
	}

	/**
	 * Clone constructor with optional entry copying
	 * @param m Matrix to copy
	 * @param copyEntries Option to add entries from previous matrix
	 */
	public Matrix(Matrix m, boolean copyEntries) {
		this(m.rows, m.cols);

		if (copyEntries) {
			for (int i = 0; i < this.rows; i++) {
				for (int j = 0; j < this.cols; j++) {
					this.setEntry(i, j, m.getEntry(i, j));
				}
			}
		}
	}

	/**
	 * Creates a matrix from two-dimensional array
	 * @param m Array to copy
	 */
	public Matrix(double[][] m) {
		int rows = m.length;
		int cols = 0;

		for (double[] array : m) {
			cols = Math.max(cols, array.length);
		}

		entries = new double[rows][cols];
		this.rows = rows;
		this.cols = cols;

		for (int i = 0; i < this.rows; i++) {
			for (int j = 0; j < this.cols; j++) {
				if (j > m[i].length - 1) {
					this.setEntry(i, j, 0);
				} else {
					this.setEntry(i, j, m[i][j]);
				}
			}
		}
	}

	/**
	 * Creates a row or column vector matrix
	 * @param m Array to convert into vector
	 * @param col Whether to turn it into a row or column vector. If false, makes row vector. If true, makes column vector.
	 */
	public Matrix(double[] m, boolean col) {
		if (col) {
			this.rows = m.length;
			this.cols = 1;

			this.entries = new double[rows][cols];

			for (int i = 0; i < this.rows; i++) {
				this.setEntry(i, 0, m[i]);
			}
		} else {
			this.rows = 1;
			this.cols = m.length;

			this.entries = new double[rows][cols];

			for (int i = 0; i < this.rows; i++) {
				this.setEntry(0, i, m[i]);
			}
		}
	}

	
	/**
	 * Gets the content of the specified entry
	 * @param row Specified row
	 * @param col Specified column
	 * @throws ArrayIndexOutOfBoundsException If the specified entry does not exist
	 */
	public double getEntry(int row, int col) throws ArrayIndexOutOfBoundsException {
		return this.entries[row][col];
	}

	/**
	 * Sets the content of the specified entry to a new value
	 * @param row Specified row
	 * @param col Specified column
	 * @param val New value of entry
	 * @throws ArrayIndexOutOfBoundsException If the specified entry does not exist
	 */
	public void setEntry(int row, int col, double val) throws ArrayIndexOutOfBoundsException {
		this.entries[row][col] = val;
	}

	/**
	 * Gets the result of multiplication of two matrices
	 * @param m1 Matrix multiplied on the left
	 * @param m2 Matrix multiplied on the right
	 * @return The result of the matrix multiplication
	 * @throws MatrixMultiplicationException If the left matrix's columns are unequal to the right matrix's rows
	 */
	public static Matrix matrixMultiply(Matrix m1, Matrix m2) throws MatrixMultiplicationException {
		if (m1.cols != m2.rows) {
			throw new MatrixMultiplicationException("Failed to multiply matrices");
		}

		Matrix result = new Matrix(m1.rows, m2.cols);

		for (int i = 0; i < result.rows; i++) {
			for (int j = 0; j < result.cols; j++) {
				for (int k = 0; k < m1.cols; k++) {
					result.setEntry(i, j, result.getEntry(i,  j) + m1.entries[i][k] * m2.entries[k][j]);
				}
			}
		}

		return result;
	}

	/**
	 * Multiplies a matrix by a scalar
	 * @param c Scalar constant to multiply
	 * @param m Matrix to be multiplied
	 * @return The result of the scalar multiplication
	 */
	public static Matrix scalarMultiply(double c, Matrix m) {
		Matrix result = new Matrix(m, false);

		for (int i = 0; i < m.rows; i++) {
			for (int j = 0; j < m.cols; j++) {
				result.setEntry(i, j, m.getEntry(i, j) * c);
			}
		}

		return result;
	}
	
	/**
	 * Adds two matrices
	 * @param m1 The first matrix to add
	 * @param m2 The second matrix to add
	 * @return The result of the matrix addition
	 * @throws MatrixAdditionException If the sizes of the two matrices are unequal.
	 */
	public static Matrix matrixAdd(Matrix m1, Matrix m2) throws MatrixAdditionException {
		if (m1.rows != m2.rows || m1.cols != m2.cols) {
			throw new MatrixAdditionException("Failed to add matrices");
		}
		
		Matrix result = new Matrix(m1, false);
		
		for (int i = 0; i < result.rows; i++) {
			for (int j = 0; j < result.cols; j++) {
				result.setEntry(i, j, m1.getEntry(i, j) + m2.getEntry(i, j));
			}
		}
		
		return result;
	}
	
	/**
	 * Applies an abstract function to every element in the matrix
	 * @param function The function, extending the abstract class MatrixFunction to be applied to the matrix
	 * @param m The matrix on which the function will be applied
	 * @return The resulting matrix from the given function.
	 */
	public static <A extends MatrixFunction> Matrix applyFunction(A function, Matrix m) {
		Matrix result = new Matrix(m, false);
		
		for (int i = 0; i < m.rows; i++) {
			for (int j = 0; j < m.cols; j++) {
				result.setEntry(i, j, function.apply(i, j, m.rows, m.cols, m.getEntry(i, j)));
			}
		}
		
		return result;
	}
}
