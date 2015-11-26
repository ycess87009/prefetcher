CFLAGS = -msse2 -std=gnu99 -O0 -Wall

EXEC = naive_transpose sse_transpose sse_prefetch_transpose asm_sse_transpose

all: $(EXEC)

naive_transpose:main.c
	$(CC) $(CFLAGS) -D NAIVE $^ -o $@ 

sse_transpose:main.c
	$(CC) $(CFLAGS) -D SSE  $^ -o $@

sse_prefetch_transpose:main.c
	$(CC) $(CFLAGS) -D SSE_PREFETCH $^ -o $@

asm_sse_transpose:main.c
	$(CC) $(CFLAGS) -D ASM_SSE $^ -o $@

clean:
	$(RM) $(EXEC)
