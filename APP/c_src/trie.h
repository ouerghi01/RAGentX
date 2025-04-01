#ifndef TRIE_H
#define TRIE_H

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Node {
    struct Node *Children[26];
    char value;
    bool end;
} Node;
typedef struct {
    char** words;  // Array of strings
    int size;      // Number of stored words
    int capacity;  // Maximum allocated capacity
} WordList;

__attribute__((visibility("default"))) Node * allocate_node(void);
__attribute__((visibility("default"))) void Trie_Insert(Node** root, char* key);
__attribute__((visibility("default"))) bool TrieFind(Node * root , char * key);
__attribute__((visibility("default"))) void generate_dot(Node *root, const char *filename);
__attribute__((visibility("default"))) void InsertMany(Node** root, char** keys, int num_keys);
__attribute__((visibility("default"))) void freeWordList(WordList* list);
__attribute__((visibility("default"))) WordList* printAutoSuggestions(Node* root, const char* query);
__attribute__((visibility("default"))) void suggestionsRec(Node* root, char* currPrefix, WordList* results);
__attribute__((visibility("default"))) void saveTrieToFile(Node* root, FILE* file, char* prefix);
__attribute__((visibility("default"))) void loadTrieFromFile(Node** root, const char* filename);
#ifdef __cplusplus
}
#endif

#endif  // TRIE_H
