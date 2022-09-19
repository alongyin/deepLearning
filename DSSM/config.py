
def load_vacab(file_path):
    word_dict = {}
    with open(file_path,encoding="utf8") as f:
        for idx,word in enumerate(f.readlines()):
            word = word.strip()
            word_dict[word] = idx
        return word_dict

class Config(object):
    def __init__(self):
        self.vocab_map = load_vacab(self.vocab_path)
        self.nwords = len(self.vocab_map)


    unk = '[UNK]'
    pad = '[PAD]'    
    vocab_path = "./data/vocab.txt"

    max_seq_len = 40
    hidden_size_rnn = 100
    use_stack_rnn = False
    learning_rate = 0.001
    decay_step = 2000
    lr_decay = 0.95
    num_epoch = 300
    epoch_no_imprv = 5
    optimizer = "lazyadam"
    sumaries_dir = "./results/Summaries"
    gpu = 0
    word_dim = 100
    batch_size = 64
    keep_porb = 0.5
    dropout = 1 - keep_porb

    # checkpoint_dir
    checkpoint_dir = './results/checkpoint'


if __name__ == "__main__":
    conf = Config()
    print(len(conf.vocab_map))
    pass