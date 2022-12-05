# reproducing results:

from src import remove_words, build_graph, train, visualize
import sys

def main():
    if len(sys.argv) != 2:
	    sys.exit("Use: python run.py <dataset>")

    datasets = ['clickbait', 'test']
    dataset = sys.argv[1]

    if dataset not in datasets:
        sys.exit("wrong dataset name")

    # Step 1: clean dataset
    remove_words.clean(dataset)

    # Step 2: build network graph
    build_graph.build(dataset)

    # Step 3: train text gcn
    train.train(dataset)

    # Step 4: visualize results
    visualize.plot(dataset)
    visualize.plot_words(dataset)

if __name__ == "__main__":
    main()
