CC = gcc
CFLAGS = -Wall -Wextra -fPIC -shared
INCLUDE = -I./c_src
SRC_FILES = c_src/trie.c
OBJ_FILES = c_src/trie.o

clib.so: $(OBJ_FILES)
	$(CC) $(CFLAGS) -o $@ $^

src/%.o: src/%.c
	$(CC) -c $(CFLAGS) $(INCLUDE) -o $@ $<

clean:
	rm -f $(OBJ_FILES) clib.so
