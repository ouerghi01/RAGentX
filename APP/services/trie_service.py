import ctypes
import os
class WordList(ctypes.Structure):
    _fields_ = [("words", ctypes.POINTER(ctypes.c_char_p)), ("size", ctypes.c_int), ("capacity", ctypes.c_int)]

class TrieService:
    def __init__(self,tokens=[], filename="trie_data.txt"):
        
        self.lib = ctypes.CDLL("./clib.so")
        self.lib.allocate_node.restype = ctypes.POINTER(ctypes.c_void_p)
        self.lib.Trie_Insert.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)), ctypes.c_char_p]
        self.lib.InsertMany.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)), ctypes.POINTER(ctypes.c_char_p), ctypes.c_int]
        self.lib.InsertMany.restype = None
        self.lib.saveTrie.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        self.lib.loadTrieFromFile.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)), ctypes.c_char_p]
        self.lib.freeWordList.argtypes = [ctypes.POINTER(WordList)]
        self.lib.freeWordList.restype = None
        self.root = self.lib.allocate_node()
        self.filename = filename
        if(tokens):
            self.insert(tokens)
        # Load trie data if exists
        if os.path.exists(self.filename):
            print(f"Loading Trie from {self.filename}")
            self.lib.loadTrieFromFile(ctypes.byref(self.root), self.filename.encode("utf-8"))
       
    def insert(self, tokens: list[str]):
        self.lib.InsertMany(ctypes.byref(self.root), tokens, len(tokens))
        self.save()


    def autocomplete(self, prefix: str):
        self.lib.printAutoSuggestions.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        self.lib.printAutoSuggestions.restype = ctypes.POINTER(WordList)
        result = self.lib.printAutoSuggestions(self.root, prefix.encode("utf-8"))
        if not result or result.contents.size == 0:
            return []
        if result and result.contents.size > 0:
            suggestions = [result.contents.words[i].decode("utf-8") for i in range(result.contents.size)]
            print("Suggestions:", suggestions)

            self.lib.freeWordList(result)
        else:
            print("No suggestions found for prefix:", prefix.decode("utf-8"))
        return suggestions

    def save(self):
        print(f"Saving Trie to {self.filename}")
        self.lib.saveTrie(self.root, self.filename.encode("utf-8"))
