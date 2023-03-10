import re


class BarTokenizer:
    """
    Class for tokenizing text midi encoded by MMMTokenizer into bars
    """
    def __init__(self, text_midi):
        self.text_midi = text_midi
        self.bar_start = "BAR_START"
        self.bar_end = "BAR_END"

    def tokenize_text_midi(self):
        reg_bars = re.compile(r'BAR_START.+?BAR_END')
        bars = reg_bars.findall(self.text_midi)
        bars = [';'.join(bar.split(' ')) for bar in bars]
        return bars
