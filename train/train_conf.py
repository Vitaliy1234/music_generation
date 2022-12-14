class MusicTrainerConf:
    # TODO: describe all the parameters of init
    def __init__(self,
                 tokenizer_path="",
                 dataset_train_files=None,
                 dataset_validate_files=None,
                 pad_length=768,
                 n_head=8,
                 n_layer=6,
                 n_embd=512,
                 n_positions=1024,
                 n_ctx=1024,
                 n_epoch=10,
                 batch_size=8):
        self.tokenizer_path = tokenizer_path
        self.n_head = n_head
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_positions = n_positions
        self.n_ctx = n_ctx
        self.dataset_train_files = dataset_train_files
        self.pad_length = pad_length
        self.dataset_validate_files = dataset_validate_files
        self.n_epoch = n_epoch
        self.batch_size = batch_size
