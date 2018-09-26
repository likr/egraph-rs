CCFLAGS=-L../target/release -legraph_c_api -lc++

ifeq ($(OS),Windows_NT)
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Darwin)
        CCFLAGS += -framework Security
    endif
endif

example: example.c
	gcc $(CCFLAGS) -o example example.c
