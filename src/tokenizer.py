import numpy as np
from typing import List, Tuple, Union, Set

class TrieNode:
    def __init__(self, char: str = None):
        self.char = char
        self.children = {}
        self.is_end = False
        self.value = None

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: bytes, value: Union[bytes, int]):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode(char)
            node = node.children[char]
        node.is_end = True
        node.value = value

    def find_longest_prefix(self, word: bytes) -> Tuple[int, Union[bytes, int]]:
        node = self.root
        last_match_idx = -1
        last_match_value = None

        for idx, char in enumerate(word):
            if char in node.children:
                node = node.children[char]
                if node.is_end:
                    last_match_idx = idx
                    last_match_value = node.value
            else:
                break

        return last_match_idx + 1, last_match_value

class RWKVTokenizer:
    def __init__(self, file_name: str):
        self.idx2token = {}
        self.token2idx = {}
        self.trie = Trie()

        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            self.idx2token[idx] = x
            self.token2idx[x] = idx
            self.trie.insert(x, idx)

        # Add special token for end of text
        self.idx2token[0] = b'<|endoftext|>'
        self.token2idx[b'<|endoftext|>'] = 0

    def encode(self, text: Union[str, List[str]]) -> List[List[int]]:
        if isinstance(text, str):
            text = [text]
        
        encoded = []
        for s in text:
            tokens = []
            s_bytes = s.encode("utf-8")
            idx = 0
            while idx < len(s_bytes):
                length, token_idx = self.trie.find_longest_prefix(s_bytes[idx:])
                if token_idx is None:
                    # If no match found, treat the byte as an unknown token
                    tokens.append(0)  # You might want to use a specific unknown token ID
                    idx += 1
                else:
                    tokens.append(token_idx)
                    idx += length
            encoded.append(tokens)
        return encoded

    def decode(self, tokens: List[List[int]]) -> List[str]:
        decoded = []
        for batch in tokens:
            text = b''.join(self.idx2token.get(i, b'') for i in batch)
            decoded.append(text.decode('utf-8', errors='replace'))
        return decoded

    def encode_numpy(self, text: Union[str, List[str]]) -> np.ndarray:
        encoded = self.encode(text)
        max_len = max(len(seq) for seq in encoded)
        padded = [seq + [0] * (max_len - len(seq)) for seq in encoded]
        return np.array(padded, dtype=np.int32)

    def decode_numpy(self, tokens: np.ndarray) -> List[str]:
        return self.decode(tokens.tolist())

if __name__ == "__main__":
    tokenizer = RWKVTokenizer("/home/sarangangster/Desktop/rwkv_jax/rwkv_vocab_v20230424.txt")
    
    text = ["Hello, world!", "This is a test."]
    encoded = tokenizer.encode(text)
    print("Encoded:", encoded)

    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)

    encoded_np = tokenizer.encode_numpy(text)
    print("Encoded (NumPy):", encoded_np)
    
    decoded_np = tokenizer.decode_numpy(encoded_np)
    print("Decoded (NumPy):", decoded_np)