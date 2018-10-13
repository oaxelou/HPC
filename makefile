# Makefile. If you change it, remember than in makefiles multiple spaces
# ARE NOT EQUIVALENT to tabs. The line after a rule starts with a tab!

#Add any executable you want to be created here.
EXECUTABLES	= sobel_orig

#This is the compiler to use
CC = icc

#These are the flags passed to the compiler. Change accordingly
CFLAGS = -Wall -O0

#These are the flags passed to the linker. Nothing in our case
LDFLAGS = -lm

### MY FLAG: name of time output file
OUTFILE = 4_function_inlining_O0.txt

# make all will create all executables
all: $(EXECUTABLES)

# This is the rule to create any executable from the corresponding .c
# file with the same name.
%: %.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

# make clean will remove all executables, jpg files and the
# output of previous executions.
clean:
	rm -f $(EXECUTABLES) *.jpg output_sobel.grey

# make image will create the output_sobel.jpg from the output_sobel.grey.
# Remember to change this rule if you change the name of the output file.
image: output_sobel.grey
	convert -depth 8 -size 4096x4096 GRAY:output_sobel.grey output_sobel.jpg

data_calculator:
	gcc -Wall -g data_calculator.c -o data_calculator -lm

test: sobel_orig data_calculator
	./sobel_orig > $(OUTFILE);
	for number in 1 2 3 4 5 6 7 8 9 10 11 ; do \
		./sobel_orig >> $(OUTFILE); \
	done
	./data_calculator < $(OUTFILE) >> $(OUTFILE)
