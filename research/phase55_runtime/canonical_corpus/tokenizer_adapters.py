"""Adapters used by the common Phase 3 evaluation harness."""
from __future__ import annotations
import json

class SentencePieceAdapter:
    kind="native_sentencepiece"
    def __init__(self, model):
        import sentencepiece as spm
        self.processor=spm.SentencePieceProcessor(model_file=str(model)); self.model=str(model)
    def encode(self,text): return self.processor.encode(text,out_type=int)
    def decode(self,ids): return self.processor.decode(ids)
    def pieces(self,ids): return [self.processor.id_to_piece(i) for i in ids]
    def is_byte(self,token):
        try:
            return bool(self.processor.is_byte(token))
        except (TypeError, ValueError):
            if isinstance(token, int):
                token = self.processor.id_to_piece(token)
            return bool(self.processor.is_byte(token))
    def is_unk(self,token): return token==self.processor.unk_id() or token=="<unk>"
    def unk_id(self): return self.processor.unk_id()
    def vocabulary(self): return [{"piece":self.processor.id_to_piece(i),"score":self.processor.get_score(i)} for i in range(self.processor.get_piece_size())]

class ReversibleSentencePieceAdapter(SentencePieceAdapter):
    """Escape wrapper kept separate from native SP results."""
    kind="reversible_sentencepiece_wrapper"
    ESC="\ue000"
    def _escape(self,text):
        return (text.replace(self.ESC,self.ESC+self.ESC).replace("\r\n",self.ESC+"R").replace("\r",self.ESC+"r").replace("\n",self.ESC+"n").replace("\u2028",self.ESC+"L").replace("\u2029",self.ESC+"P").replace("▁",self.ESC+"s").replace("\x00",self.ESC+"0"))
    def _unescape(self,text):
        out=[]; i=0
        while i<len(text):
            if text[i]!=self.ESC: out.append(text[i]); i+=1; continue
            if i+1>=len(text): raise ValueError("dangling sentinel")
            c=text[i+1]; out.append(self.ESC if c==self.ESC else {"R":"\r\n","r":"\r","n":"\n","L":"\u2028","P":"\u2029","s":"▁","0":"\x00"}.get(c,self.ESC+c)); i+=2
        return "".join(out)
    @classmethod
    def escape_text(cls, text):
        return cls._escape(object.__new__(cls), text)
    def encode(self,text): return self.processor.encode(self._escape(text),out_type=int)
    def decode(self,ids): return self._unescape(self.processor.decode(ids))

class HFTokenizerAdapter:
    kind="huggingface_reference"
    def __init__(self, tokenizer): self.tokenizer=tokenizer
    def encode(self,text):
        x=self.tokenizer.encode(text,add_special_tokens=False); return list(getattr(x,"ids",x))
    def decode(self,ids): return self.tokenizer.decode(list(ids),skip_special_tokens=False)
    def pieces(self,ids): return self.tokenizer.id_to_token if False else [self.tokenizer.id_to_token(i) for i in ids]
    def is_byte(self, token):
        piece = self.tokenizer.id_to_token(token) if isinstance(token, int) else str(token)
        return piece.startswith("<0x") and piece.endswith(">")
    def is_unk(self, token):
        unk = self.unk_id()
        return unk is not None and token == unk
    def unk_id(self):
        token = self.tokenizer.token_to_id("[UNK]")
        return token if token is not None else self.tokenizer.token_to_id("<unk>")
    def vocabulary(self): return [{"piece":p,"score":None} for p in self.tokenizer.get_vocab()]

def load_hf(path):
    from tokenizers import Tokenizer
    return HFTokenizerAdapter(Tokenizer.from_file(str(path)))
