#-------------------------------------------------------------------------------
# DEPENDENCIES
#-------------------------------------------------------------------------------

# The location of the header file for the erlang runtime system.
#
# Example:
#
#   /usr/lib/erlang/erts-8.2/include
#
ERL_INCLUDE_PATH = $(shell erl -eval 'io:format("~s", [lists:concat([code:root_dir(), "/erts-", erlang:system_info(version), "/include"])])' -s init stop -noshell)


#-------------------------------------------------------------------------------
# FLAGS
#-------------------------------------------------------------------------------

# Switches between different BLAS implementations
# Can be blas, openblas, atlas, noblas
ifdef MATREX_BLAS
	BLAS = $(MATREX_BLAS)
else
	BLAS = blas
endif

# For compiling and linking the final NIF shared objects.

CFLAGS = -fPIC -I$(ERL_INCLUDE_PATH) -O3 -std=gnu11 -Wall -Wextra
LDFLAGS =

ifeq ($(BLAS), blas)
	LDFLAGS += -lblas
else ifeq ($(BLAS), noblas)
	CFLAGS += -D MATREX_NO_BLAS
endif

# Determine Platform, and check for override $ARCH var
COMPILE_ARCH=linux
ifeq ($(shell uname -s), Darwin)
	COMPILE_ARCH=darwin
endif

ifeq ($(findstring linux,$(CC)),linux)
	COMPILE_ARCH=linux
endif

# MacOS needs extra flags to link successfully
ifeq ($(COMPILE_ARCH), darwin)
	LDFLAGS +=  -flat_namespace -undefined suppress

## MacOS BLAS
ifeq ($(BLAS), openblas)
	CFLAGS += -I/usr/local/opt/openblas/include
	LDFLAGS += -L/usr/local/opt/openblas/lib
else ifeq ($(BLAS), blas)
	CFLAGS += -I/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Versions/Current/Headers
endif

else ifeq ($(COMPILE_ARCH), linux) # Linux
	CFLAGS += -shared
	LDFLAGS += -lm 

ifeq ($(BLAS), openblas)
	LDFLAGS += -lopenblas
else ifeq ($(BLAS), atlas)
	LDFLAGS += -latlas
endif

else
	$(error var was not specified at commandline!)
endif

# For compiling and linking the test runner.
TEST_CFLAGS  = -g -O0 -std=gnu11 -Wall -Wextra --coverage
TEST_LDFLAGS =# -lgcov


#-------------------------------------------------------------------------------
# DIRECTORIES
#-------------------------------------------------------------------------------

# Location of the root directory for the C code files.
SRC_DIRECTORY = ./native/src

# Location fo the root directory for the C header files.
INCLUDE_DIRECTORY = ./native/include

# Location of the root directory for the object files created from the C code
# files.
OBJ_DIRECTORY = ./_build/obj

# Location of the root directory for the NIF code files.
NIFS_DIRECTORY = ./native/nifs

# Location of the `priv` directory where the final shared objects are placed.
PRIV_DIRECTORY = ./priv

# A list of all the subdirectories containing C code files.
SOURCES_DIRECTORIES := $(shell find $(SRC_DIRECTORY) -type d)

# A list of all the subdirectories containing object files created from the C
# code files. The list is a mirror of the directory structure in which the C
# code files reside.
OBJECTS_DIRECTORIES := $(subst $(SRC_DIRECTORY),$(OBJ_DIRECTORY),$(SOURCES_DIRECTORIES))

# Location of the root directory for the test object files created from the
# C source files with the test specific flags.
TEST_OBJ_DIRECTORY = ./test/c/obj

# A list of all the subdirectories containing object files created from the C
# code files with the test specific flags. The list is a mirror of the directory
# structure in which the C code files reside.
TEST_OBJECTS_DIRECTORIES := $(subst $(SRC_DIRECTORY),$(TEST_OBJ_DIRECTORY),$(SOURCES_DIRECTORIES))


#-------------------------------------------------------------------------------
# FILES
#-------------------------------------------------------------------------------

# Lists of all the C source files.
SOURCES := $(shell find $(SRC_DIRECTORY) -name *.c)

# Lists of all the C header files.
HEADERS := $(shell find $(INCLUDE_DIRECTORY) -name *.h)

# List of all the object files created from the C code files. The list is a
# mirror of the directory structure in which the C code files reside.
OBJECTS := $(SOURCES:$(SRC_DIRECTORY)/%.c=$(OBJ_DIRECTORY)/%.o)

# A list of all the NIF code files.
NIFS_SOURCES := $(wildcard $(NIFS_DIRECTORY)/*.c)

# A list of all the helpers used by the NIF code files.
NIFS_HELPERS := $(shell find $(NIFS_DIRECTORY) -name *_helper.c)

# A list of all the shared objects created by compiling the NIF code files.
NIFS_OBJECTS := $(NIFS_SOURCES:$(NIFS_DIRECTORY)/%.c=$(PRIV_DIRECTORY)/%.so)

# A list of all the shared objects created from the C code files with the test
# specific flags.
TEST_OBJECTS := $(SOURCES:$(SRC_DIRECTORY)/%.c=$(TEST_OBJ_DIRECTORY)/%.o)


#-------------------------------------------------------------------------------
# TARGETS
#-------------------------------------------------------------------------------

# Use the `build` target when executing make without arguments.
#
# Example:
#
#   Executing:
#   ```bash
#   make
#   ```
#
#   Is equivalent to executing:
#   ```bash
#   make build
#   ```
#
.DEFAULT_GLOBAL := build

# Targets that do not depend on files.
.PHONY: build ci clean test

# Compiles and links the C nifs.
#
# Example
#
#   ```bash
#   make build
#   ```
#
build: $(OBJECTS_DIRECTORIES) $(OBJECTS) $(PRIV_DIRECTORY) $(NIFS_OBJECTS)

# Target for creating directories for the C object files.
$(OBJECTS_DIRECTORIES):
	@mkdir -p $(OBJECTS_DIRECTORIES)
	@echo 'Compile Arch: '$(COMPILE_ARCH)
	@echo 'Library BLAS: '$(BLAS)

# Target for creating object files from C source files.
# Each object file depends on it's corresponding C source file for compilation.
#
# Example:
#   native/obj/matrix.o
#
# Depends on:
#   native/src/matrix.c
#   native/include/matrix.h
#
# Output:
#   ```
#   Compiling: native/src/matrix.c
#   ```
#
$(OBJECTS): $(OBJ_DIRECTORY)/%.o : $(SRC_DIRECTORY)/%.c $(INCLUDE_DIRECTORY)/%.h
	@echo 'Compiling: '$<
	@$(CC) $(CFLAGS) -c $< -o $@

# Target for creating the `priv` directory for the NIF shared objects.
$(PRIV_DIRECTORY):
	@mkdir -p $(PRIV_DIRECTORY)

# Target for creating the NIF shared objects.
#
# Example:
#   priv/worker_nifs.so
#
# Depends on:
#   priv/
#   native/nifs/matrix_nifs.c
#   native/nifs/helpers/network_state_helper.c
#   _build/obj/matrix.o
#
# Output:
#   ```
#   Creating NIF: priv/worker_nifs.so
#   ```
#
$(NIFS_OBJECTS): $(PRIV_DIRECTORY)/%.so : $(NIFS_DIRECTORY)/%.c $(OBJECTS) $(NIFS_HELPERS)
	@echo 'Creating NIF: '$@
	@$(CC) $(CFLAGS) $(OBJECTS) -o $@ $< $(LDFLAGS)

# Builds the C code with debugging and testing flags and runs the tests.
#
# Example:
#
#   ```bash
#   make test
#   ```
#
test: $(TEST_OBJECTS_DIRECTORIES) $(TEST_OBJECTS)
	@find test/c/temp/ ! -name '.keep' -type f -exec rm -f {} +
	@lcov --directory . -z --rc lcov_branch_coverage=1
	$(CC) $(TEST_CFLAGS) $(TEST_OBJECTS) -o test/c/temp/test test/test_helper.c $(LDFLAGS) $(TEST_LDFLAGS)
	./test/c/temp/test
	@lcov --directory . -c -o cover/lcov.info-file --rc lcov_branch_coverage=1
	@lcov --list cover/lcov.info-file --rc lcov_branch_coverage=1
	@genhtml --branch-coverage -o cover cover/lcov.info-file > /dev/null
	@rm -rf *.gcda *.gcno

# Target for creating directories for the C object files compiled with test
# specific flags.
$(TEST_OBJECTS_DIRECTORIES):
	@mkdir -p $(TEST_OBJECTS_DIRECTORIES)

# Target for creating object files from C source files with test specific flags.
# Each object file depends on it's corresponding C source file for compilation.
#
# Example:
#   test/c/obj/matrix.o
#
# Depends on:
#   native/src/matrix.c
#   native/include/matrix.h
#
# Output:
#   ```
#   Compiling: native/src/matrix.c
#   ```
#
$(TEST_OBJECTS): $(TEST_OBJ_DIRECTORY)/%.o : $(SRC_DIRECTORY)/%.c $(INCLUDE_DIRECTORY)/%.h
	@echo 'Compiling: '$<
	@$(CC) $(TEST_CFLAGS) -c $< -o $@

# Remove build artifacts.
# Run this when you want to ensure you run a fresh build.
#
# Example:
#
#   To remove all artifacts:
#   ```bash
#   make clean
#   ```
#
#   To create a fresh build:
#   ```
#   make clean build
#   ```
#
clean:
	@$(RM) -rf $(OBJ_DIRECTORY) $(PRIV_DIRECTORY) $(TEST_OBJ_DIRECTORY)

# Run the complete suite of tests and checks.
#
# Example:
#
#   ```bash
#   make ci
#   ```
#
ci:
	@make
	@mix deps.get
	@mix dialyzer --plt
	@make test
	@mix coveralls.travis
	@mix dialyzer --halt-exit-status

