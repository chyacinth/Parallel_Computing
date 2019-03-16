all:scan run

CC=gcc

scan:scan.o genscan.o
	@gcc scan.c genscan.c -o scan

run:scan
	@./scan 10

clean:
	@/bin/rm -rf *.o
