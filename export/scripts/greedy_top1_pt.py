# python greedy_top1_pt.py ../models/hi/greedy_decoder.pt

import torch
from pathlib import Path
import sys
save_path = sys.argv[1]

class GreedyDecoder(torch.nn.Module):
    def __init__(self):
        super(GreedyDecoder, self).__init__()
      
    def forward(self, batch_log_ctc_probs):
        return torch.argmax(batch_log_ctc_probs, dim=-1).int()


m = torch.jit.script(GreedyDecoder())
decoder_path = Path(save_path) # need a greedy_decoder
# Save to file
torch.jit.save(m, decoder_path)
