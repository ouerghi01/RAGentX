
#include "trie.h"
int capacity=1024;
#define MAX_WORD_LENGTH 100  // Maximum word size
#define INITIAL_CAPACITY 10   // Initial size for results array

WordList* createWordList() {
    WordList* list = (WordList*)malloc(sizeof(WordList));
    list->size = 0;
    list->capacity = INITIAL_CAPACITY;
    list->words = (char**)malloc(list->capacity * sizeof(char*));
    return list;
}
void addWordToList(WordList* list, const char* word) {
    if (list->size >= list->capacity) {
        list->capacity *= 2;
        list->words = (char**)realloc(list->words, list->capacity * sizeof(char*));
    }
    list->words[list->size] = strdup(word);  // Duplicate the string and store it
    list->size++;
}

Node * allocate_node(void) {
    Node *root = (Node*) malloc(sizeof(Node));
    if (root == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return (Node *)NULL;
    }
    for(int i = 0; i < 26; i++){
        root->Children[i] = (Node*)NULL;
    }
    root->end = false;
    
    return root;
}
void Trie_Insert(Node** root, char* key) {
    if (root == NULL) {
        printf("ERROR: Trie_Insert called with NULL root pointer!\n");
        return;
    }
    if (*root == NULL) {
        printf("ERROR: Trie_Insert called with uninitialized root!\n");
        return;
    }
    if (key == NULL) {
        printf("ERROR: Trie_Insert called with NULL key!\n");
        return;
    }

    printf("Inserting: %s\n", key);

    if (strlen(key) == 0) {
        (*root)->end = true;
        return;
    }

    Node* ptr = *root;
    int index = (int)(*key) - (int)'a';
    
    if (index < 0 || index >= 26) {
        printf("ERROR: Invalid character index: %d ('%c')\n", index, *key);
        return;
    }

    if (ptr->Children[index] == NULL) {
        ptr->Children[index] = allocate_node();
        if (ptr->Children[index] == NULL) {
            printf("ERROR: Failed to allocate memory for new Trie node!\n");
            return;
        }
        ptr->Children[index]->value = *key;
    }

    ptr = ptr->Children[index];
    Trie_Insert(&ptr, key + 1);
}


void InsertMany(Node** root, char** keys, int num_keys) {
    for (int i = 0; i < num_keys; i++) {
        Trie_Insert(root, keys[i]);  // Pass the full key (string) instead of individual characters
    }
    
}
bool isLastnode(Node *node) {
    for(int i = 0 ; i<26;i++){
        if(node->Children[i]) return false ;
    }
    return true;
}
// Recursive function to collect words from the Trie
void suggestionsRec(Node* root, char* currPrefix, WordList* results) {
    if (root->end) {
        addWordToList(results, currPrefix);  // Store the word
    }

    for (int i = 0; i < 26; i++) {
        if (root->Children[i]) {
            char child = 'a' + i;
            char newPrefix[MAX_WORD_LENGTH];
            snprintf(newPrefix, sizeof(newPrefix), "%s%c", currPrefix, child);
            suggestionsRec(root->Children[i], newPrefix, results);
        }
    }
}

WordList* printAutoSuggestions(Node* root, const char* query) {
    Node* pCrawl = root;
    
    for (int i = 0; i < strlen(query); i++) {
        int ind = query[i] - 'a';
        if (!pCrawl->Children[ind]) {
            return createWordList();  // Prefix not found in Trie
        }
        pCrawl = pCrawl->Children[ind];
    }

    WordList* results = createWordList();
    suggestionsRec(pCrawl, query, results);

    return results;
}

// write project to parse pseudo code into c code 
bool    TrieFind(Node * root , char * key){
    if (root == NULL) return  false ;
    Node * ptr = root ;
    for (int  i = 0 ; i< strlen(key);i++) {
        char k = (unsigned char) key[i];
        if (ptr->Children[k] ==NULL)  return  false;
        ptr = ptr->Children[(unsigned char)key[i]];
        
    }
    return true ;
    
}

void print_trie_dot(Node *root, FILE *file) {
    static int node_id = 0;

    if (root == NULL) return;

    int current_id = node_id++;

    // Label the node
    fprintf(file, "  node%d [label=\"%c\"];\n", current_id, root->value ? root->value : ' ');

    for (int i = 0; i < 26; i++) {
        if (root->Children[i] != NULL) {
            int child_id = node_id;
            fprintf(file, "  node%d -> node%d;\n", current_id, child_id);
            print_trie_dot(root->Children[i], file);
        }
    }
}
void generate_dot(Node *root, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("Error opening file");
        return;
    }
    fprintf(file, "digraph Trie {\n");
    fprintf(file, "  node [shape=circle];\n");
    print_trie_dot(root, file);
    fprintf(file, "}\n");

    fclose(file);
}
// Free memory allocated for WordList
void freeWordList(WordList* list) {
    for (int i = 0; i < list->size; i++) {
        free(list->words[i]);
    }
    free(list->words);
    free(list);
}
// Save Trie to a file
void saveTrieToFile(Node* root, FILE* file, char* prefix) {
    if (root->end) {
        fprintf(file, "%s\n", prefix);  // Save words
    }

    for (int i = 0; i < 26; i++) {
        if (root->Children[i]) {
            char newPrefix[MAX_WORD_LENGTH];
            snprintf(newPrefix, sizeof(newPrefix), "%s%c", prefix, 'a' + i);
            saveTrieToFile(root->Children[i], file, newPrefix);
        }
    }
}
void saveTrie(Node* root, const char* filename) {
    FILE* file = fopen(filename, "w");
    if (!file) return;

    char prefix[MAX_WORD_LENGTH] = "";
    saveTrieToFile(root, file, prefix);
    fclose(file);
}
// Load Trie from a file
void loadTrieFromFile(Node** root, const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) return;

    char word[MAX_WORD_LENGTH];
    while (fscanf(file, "%s", word) != EOF) {
        Trie_Insert(root, word);
    }
    fclose(file);
}
