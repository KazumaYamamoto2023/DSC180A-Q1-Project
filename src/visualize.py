from matplotlib.backends.backend_pdf import PdfPages
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np

def plot_loss(dataset, epoch_lst, train_losses, train_accs, val_losses, val_accs):
    pdf = PdfPages('data/visualizations/' + dataset + '_train_val_curve.pdf')
    plt.plot(epoch_lst, train_losses, label = "training loss")
    plt.plot(epoch_lst, train_accs, label = "training accuracy")
    plt.plot(epoch_lst, val_losses, label = "validation loss")
    plt.plot(epoch_lst, val_accs, label = "validation accuracy")

    plt.legend(ncol=2,  )
    plt.legend(ncol=5,loc='upper center',bbox_to_anchor=(0.48, -0.08),fontsize=11)
    plt.xlabel("Epoch")
    plt.tight_layout()
    pdf.savefig()
    plt.show()
    pdf.close()

def plot(dataset):
    f = open('data/output/' + dataset + '_shuffle.txt', 'r')
    lines = f.readlines()
    f.close()

    f = open('data/output/' + dataset + '_doc_vectors.txt', 'r')
    embedding_lines = f.readlines()
    f.close()

    target_names = set()
    labels = []
    docs = []
    for i in range(len(lines)):
        line = lines[i].strip()
        temp = line.split('\t')
        if temp[1].find('test') != -1:
            labels.append(temp[2])
            emb_str = embedding_lines[i].strip().split()
            values_str_list = emb_str[1:]
            values = [float(x) for x in values_str_list]
            docs.append(values)
            target_names.add(temp[2])

    target_names = list(target_names)

    label = np.array(labels)

    fea = TSNE(n_components=2).fit_transform(docs)
    pdf = PdfPages('data/visualizations/' + dataset + '_gcn_doc_test_1st_layer.pdf')
    cls = np.unique(label)

    # cls=range(10)
    fea_num = [fea[label == i] for i in cls]
    for i, f in enumerate(fea_num):
        if cls[i] in range(10):
            plt.scatter(f[:, 0], f[:, 1], label=cls[i], marker='+')
        else:
            plt.scatter(f[:, 0], f[:, 1], label=cls[i])
    plt.legend(ncol=2,  )
    plt.legend(ncol=5,loc='upper center',bbox_to_anchor=(0.48, -0.08),fontsize=11)
    # plt.ylim([-20,35])
    # plt.title(md_file)
    plt.tight_layout()
    plt.savefig('classification.png')
    pdf.savefig()
    plt.show()
    pdf.close()

def plot_words(dataset):
    f = open('data/output/' + dataset + '_word_vectors.txt', 'r')
    embedding_lines = f.readlines()
    f.close()

    target_names = set()
    labels = []
    docs = []
    for i in range(len(embedding_lines)):
        line = embedding_lines[i].strip()
        temp = line.split('\t')
        emb_str = embedding_lines[i].strip().split()
        values_str_list = emb_str[1:]
        values = [float(x) for x in values_str_list]
        label = np.argmax(values)
        docs.append(values)
        target_names.add(label)
        labels.append(label)

    target_names = list(target_names)

    label = np.array(labels)

    fea = TSNE(n_components=2).fit_transform(docs)
    pdf = PdfPages('data/visualizations/' + dataset + '_word_1st_layer.pdf')
    cls = np.unique(label)

    # cls=range(10)
    fea_num = [fea[label == i] for i in cls]
    for i, f in enumerate(fea_num):
        if cls[i] in range(10):
            plt.scatter(f[:, 0], f[:, 1], label=cls[i], marker='+')
        else:
            plt.scatter(f[:, 0], f[:, 1], label=cls[i])
    plt.legend(ncol=2,  )
    plt.legend(ncol=5,loc='upper center',bbox_to_anchor=(0.48, -0.08),fontsize=11)
    # plt.ylim([-20,35])
    # plt.title(md_file)
    plt.tight_layout()
    plt.savefig('words.png')
    pdf.savefig()
    plt.show()
    pdf.close()