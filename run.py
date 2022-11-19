# reproducing results:

from src import remove_words, build_graph, train, visualize, visualize_words
import sys

def main():
    if len(sys.argv) != 2:
	    sys.exit("Use: python run.py <dataset>")

    datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr', 'clickbait']
    dataset = sys.argv[1]

    if dataset not in datasets:
        sys.exit("wrong dataset name")

    # Step 1: prepare dataset
    remove_words.prepare(dataset)

    # Step 2: build network graph
    build_graph.build(dataset)

    # Step 3: train text gcn
    train.train(dataset)

    # Step 4: visualize results
    visualize.plot(dataset)
    visualize_words.plot(dataset)

if __name__ == "__main__":
    main()