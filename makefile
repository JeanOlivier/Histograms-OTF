# Toolchain, using mingw on windows
CC = $(OS:Windows_NT=x86_64-w64-mingw32-)gcc
PKG_CFG = $(OS:Windows_NT=x86_64-w64-mingw32-)pkg-config
RM = rm

# flags
CFLAGS = -Ofast -march=native -Wall
#CFLAGS = -O3 -march=native  -Wall
#CFLAGS = -Ofast -march=native -fno-math-errno  -Wall
#CFLAGS = -Ofast -march=native -ffast-math  -Wall
OMPFLAGS = -fopenmp -fopenmp-simd
SHRFLAGS = -fPIC -shared

# libraries
LDLIBS = -lmpfr

# filenames
SRC = histograms.c
SHREXT = $(if $(filter $(OS),Windows_NT),.dll,.so)
SHRTGT = $(SRC:.c=$(SHREXT))


all: $(SHRTGT) #$(TARGET) $(SHRTGT) 

$(SHRTGT): $(SRC)
	$(CC) $(SRC) -o $(SHRTGT) $(SHRFLAGS) $(CFLAGS) $(OMPFLAGS) $(LDLIBS)

force: 
	$(CC) $(SRC) -o $(SHRTGT) $(SHRFLAGS) $(CFLAGS) $(OMPFLAGS) $(LDLIBS)

clean:
	@[ -f $(SHRTGT) ] && $(RM) $(SHRTGT) || true

.PHONY: all clean force 
