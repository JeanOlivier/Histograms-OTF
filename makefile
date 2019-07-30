# histograms.dll/.so depends on all the .o files (in /obj) which, in turn, 
# depend on their correspondind .c file (in /src) which, inturn depend on their 
# corresponding .h (in /includes)
# For this simple project all %.c have a correspondin %.h and generate a corresponding %.o which are all linked
# in histograms.dll/.so 

# subdirectories
#SDIR = src
IDIR = includes
ODIR = obj
LDIR = lib
SDIR = src

# Lits of .c and corresponding .o and .h
SRC = $(wildcard $(SDIR)/*.c)
OBJ = $(patsubst $(SDIR)/%.c,$(ODIR)/%.o,$(SRC))
HEAD= $(patsubst $(SDIR)/%.c,$(IDIR)/%.h,$(SRC))

# Toolchain, using mingw on windows
CC = $(OS:Windows_NT=x86_64-w64-mingw32-)gcc
PKG_CFG = $(OS:Windows_NT=x86_64-w64-mingw32-)pkg-config

# flags
CFLAGS = -Ofast -march=native -Wall -I$(IDIR)
#CFLAGS = -Ofast -march=native -Wall -funroll-loops
#CFLAGS = -O3 -march=native  -Wall
#CFLAGS = -Ofast -march=native -fno-math-errno  -Wall
#CFLAGS = -Ofast -march=native -ffast-math  -Wall
OMPFLAGS = -fopenmp -fopenmp
SHRFLAGS = -fPIC -shared

# libraries
LDLIBS = -lmpfr

# Change the target extension depending on OS (histograms.dll or histograms.so)
TARGET = histograms
SHREXT = $(if $(filter $(OS),Windows_NT),.dll,.so)
SHRTGT = $(TARGET)$(SHREXT)

$(SHRTGT): $(OBJ)
	@ echo " "
	@ echo "---------Compile library $(SHRTGT)---------"
	$(CC) $(SHRFLAGS) -o $(SHRTGT) $^ $(CFLAGS) $(OMPFLAGS) $(LDLIBS)

$(ODIR)/%.o : $(SDIR)/%.c $(IDIR)/%.h
	@ echo " "
	@ echo "---------Compile object $@ from $<--------"
	$(CC) -c -Wall -o $@ $<  $(CFLAGS) $(CFLAGS) $(OMPFLAGS) $(LDLIBS)
	
# force: 
#	 $(CC) $(SRC) -o $(SHRTGT) $(SHRFLAGS) $(CFLAGS) $(OMPFLAGS) $(LDLIBS)
# Instead used make -B 

clean:
	 @rm -f $(SHRTGT) $(OBJ) histograms_otf.pyc
	 
.PHONY: clean 
#.PHONY: all clean force 



