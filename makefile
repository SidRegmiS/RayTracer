default: main

main:
	gcc -g -Wall main.c -o ray3 -lm

clean:
	rm -f ray3