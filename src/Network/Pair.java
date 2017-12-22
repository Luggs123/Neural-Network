package Network;

public class Pair<A, B> {
	private A first;
	private B second;
	
	public Pair(A obj1, B obj2) {
		this.first = obj1;
		this.second = obj2;
	}

	public A getFirst() {
		return first;
	}

	public B getSecond() {
		return second;
	}
}
