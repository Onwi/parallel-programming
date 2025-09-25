CC = gcc
CFLAGS = -Wall -O2 -fopenmp
LDFLAGS = -lm -fopenmp

TARGET = pca
SRC = pca_openmp.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

