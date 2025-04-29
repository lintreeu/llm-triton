from __future__ import annotations
import sentencepiece as spm

class SentencePieceTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
    def encode(self,text,add_bos=True,add_eos=False):
        ids=self.sp.encode(text,out_type=int)
        if add_bos: ids=[self.sp.bos_id()]+ids
        if add_eos: ids=ids+[self.sp.eos_id()]
        return ids
    def decode(self,ids):
        ids=[i for i in ids if i not in (self.sp.bos_id(), self.sp.eos_id(), self.sp.pad_id())]
        return self.sp.decode(ids)
    @property
    def bos_id(self): return self.sp.bos_id()
    @property
    def eos_id(self): return self.sp.eos_id()
    @property
    def pad_id(self): return self.sp.pad_id()
