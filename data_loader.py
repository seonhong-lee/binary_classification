import torchtext
import torchtext import data

class DataLoader(object):
    
    def __init__(
        self,
        train_file_name,
        batch_size=64,
        valid_ratio=.2,
        device=-1,
        max_vocab=999999,
        min_freq=1,
        use_eos=False,
        shuffle=True
    ):
        super().__init__()
        
        self.label = data.Field(
            sequential=False,
            use_vocab=True,
            unk_token=None
        )
        self.text = data.Field(
            use_vocab=True,
            batch_first=True,
            include_lengths=False,
            eos_token="<EOS>" if use_eos else None
        )