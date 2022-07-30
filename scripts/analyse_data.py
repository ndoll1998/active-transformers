import datasets
import numpy as np
import matplotlib.pyplot as plt

from itertools import chain

if __name__ == '__main__':

    dataset = 'conll2003' # 'swedish_ner_corpus' #'src/data/conll2003.py'
    label_column = 'ner_tags'

    # load dataset
    data = datasets.load_dataset(dataset, split='train')
    label_names = data.info.features[label_column].feature.names

    # filter dataset
    # data = data.filter(lambda ex: len(ex['tokens']) > 15)


    # get tokens and target labels
    tokens = data['tokens']
    labels = data[label_column]
    
    # process
    seq_lengths = list(map(len, tokens))
    unique_seq_lengths, seq_counts = np.unique(seq_lengths, return_counts=True)
    
    flat_labels = list(chain.from_iterable(labels))
    unique_labels, label_counts = np.unique(flat_labels, return_counts=True)
    unqiue_labels = [label_names[i] for i in unique_labels]
    
    # closer look at specific samples
    seq_lengths = np.asarray(seq_lengths)
    idx, = np.where(seq_lengths <= 4)
    if len(idx) > 0:
        idx = np.random.choice(idx, 5)
        for i in np.random.choice(idx, 10):
            print("Tokens:  ", tokens[i])
            print("NER-Tags:", labels[i])
            print("-" * 50)

    # plot
    fig, (ax_seq_len, ax_label_dist) = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle(data.info.builder_name)
    # plot sequence length histogram
    ax_seq_len.bar(unique_seq_lengths, seq_counts, width=0.8)
    ax_seq_len.set(
        title="Sequence Lengths",
        xlabel="Lengths",
        ylabel="Counts"
    )
    # plot label distribution
    ax_label_dist.bar(unique_labels, label_counts, width=0.8, bottom=1)
    ax_label_dist.set(
        title="Label Distribution",
        xlabel="Label",
        ylabel="Counts",
        yscale="log"
    )
    # rotate label ticks
    plt.draw()
    ax_label_dist.set_xticklabels(unique_labels, rotation=45)
    # save figure
    fig.savefig("%s.png" % data.info.builder_name)    

