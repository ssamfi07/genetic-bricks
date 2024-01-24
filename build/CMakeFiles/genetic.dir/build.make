# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/home/sstef/Documents/Master/Nico's adventure/genetic-bricks"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/sstef/Documents/Master/Nico's adventure/genetic-bricks/build"

# Include any dependencies generated for this target.
include CMakeFiles/genetic.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/genetic.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/genetic.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/genetic.dir/flags.make

CMakeFiles/genetic.dir/src/genetic_algorithm.cpp.o: CMakeFiles/genetic.dir/flags.make
CMakeFiles/genetic.dir/src/genetic_algorithm.cpp.o: ../src/genetic_algorithm.cpp
CMakeFiles/genetic.dir/src/genetic_algorithm.cpp.o: CMakeFiles/genetic.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/sstef/Documents/Master/Nico's adventure/genetic-bricks/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/genetic.dir/src/genetic_algorithm.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/genetic.dir/src/genetic_algorithm.cpp.o -MF CMakeFiles/genetic.dir/src/genetic_algorithm.cpp.o.d -o CMakeFiles/genetic.dir/src/genetic_algorithm.cpp.o -c "/home/sstef/Documents/Master/Nico's adventure/genetic-bricks/src/genetic_algorithm.cpp"

CMakeFiles/genetic.dir/src/genetic_algorithm.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/genetic.dir/src/genetic_algorithm.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/sstef/Documents/Master/Nico's adventure/genetic-bricks/src/genetic_algorithm.cpp" > CMakeFiles/genetic.dir/src/genetic_algorithm.cpp.i

CMakeFiles/genetic.dir/src/genetic_algorithm.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/genetic.dir/src/genetic_algorithm.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/sstef/Documents/Master/Nico's adventure/genetic-bricks/src/genetic_algorithm.cpp" -o CMakeFiles/genetic.dir/src/genetic_algorithm.cpp.s

# Object files for target genetic
genetic_OBJECTS = \
"CMakeFiles/genetic.dir/src/genetic_algorithm.cpp.o"

# External object files for target genetic
genetic_EXTERNAL_OBJECTS =

genetic: CMakeFiles/genetic.dir/src/genetic_algorithm.cpp.o
genetic: CMakeFiles/genetic.dir/build.make
genetic: /usr/lib/x86_64-linux-gnu/libboost_log_setup.so.1.74.0
genetic: /usr/lib/x86_64-linux-gnu/libboost_log.so.1.74.0
genetic: /usr/lib/x86_64-linux-gnu/libboost_chrono.so.1.74.0
genetic: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so.1.74.0
genetic: /usr/lib/x86_64-linux-gnu/libboost_regex.so.1.74.0
genetic: /usr/lib/x86_64-linux-gnu/libboost_thread.so.1.74.0
genetic: /usr/lib/x86_64-linux-gnu/libboost_atomic.so.1.74.0
genetic: CMakeFiles/genetic.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/home/sstef/Documents/Master/Nico's adventure/genetic-bricks/build/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable genetic"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/genetic.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/genetic.dir/build: genetic
.PHONY : CMakeFiles/genetic.dir/build

CMakeFiles/genetic.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/genetic.dir/cmake_clean.cmake
.PHONY : CMakeFiles/genetic.dir/clean

CMakeFiles/genetic.dir/depend:
	cd "/home/sstef/Documents/Master/Nico's adventure/genetic-bricks/build" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/sstef/Documents/Master/Nico's adventure/genetic-bricks" "/home/sstef/Documents/Master/Nico's adventure/genetic-bricks" "/home/sstef/Documents/Master/Nico's adventure/genetic-bricks/build" "/home/sstef/Documents/Master/Nico's adventure/genetic-bricks/build" "/home/sstef/Documents/Master/Nico's adventure/genetic-bricks/build/CMakeFiles/genetic.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/genetic.dir/depend
