CC = mpicc
CFLAGS = -O2
LDFLAGS = -lm

SRC_DIR = src
SRCS = $(SRC_DIR)/main.c $(SRC_DIR)/parallel_sort.c $(SRC_DIR)/sorting.c $(SRC_DIR)/load_analysis.c $(SRC_DIR)/load_balance.c $(SRC_DIR)/output.c
TARGET = parallel_sort

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -I$(SRC_DIR) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)
