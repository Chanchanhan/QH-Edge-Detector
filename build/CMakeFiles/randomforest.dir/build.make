# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/qqh/projects/RandomForest

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/qqh/projects/RandomForest/build

# Include any dependencies generated for this target.
include CMakeFiles/randomforest.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/randomforest.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/randomforest.dir/flags.make

CMakeFiles/randomforest.dir/src/main.cpp.o: CMakeFiles/randomforest.dir/flags.make
CMakeFiles/randomforest.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/qqh/projects/RandomForest/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/randomforest.dir/src/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/randomforest.dir/src/main.cpp.o -c /home/qqh/projects/RandomForest/src/main.cpp

CMakeFiles/randomforest.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/randomforest.dir/src/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/qqh/projects/RandomForest/src/main.cpp > CMakeFiles/randomforest.dir/src/main.cpp.i

CMakeFiles/randomforest.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/randomforest.dir/src/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/qqh/projects/RandomForest/src/main.cpp -o CMakeFiles/randomforest.dir/src/main.cpp.s

CMakeFiles/randomforest.dir/src/main.cpp.o.requires:

.PHONY : CMakeFiles/randomforest.dir/src/main.cpp.o.requires

CMakeFiles/randomforest.dir/src/main.cpp.o.provides: CMakeFiles/randomforest.dir/src/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/randomforest.dir/build.make CMakeFiles/randomforest.dir/src/main.cpp.o.provides.build
.PHONY : CMakeFiles/randomforest.dir/src/main.cpp.o.provides

CMakeFiles/randomforest.dir/src/main.cpp.o.provides.build: CMakeFiles/randomforest.dir/src/main.cpp.o


# Object files for target randomforest
randomforest_OBJECTS = \
"CMakeFiles/randomforest.dir/src/main.cpp.o"

# External object files for target randomforest
randomforest_EXTERNAL_OBJECTS =

randomforest: CMakeFiles/randomforest.dir/src/main.cpp.o
randomforest: CMakeFiles/randomforest.dir/build.make
randomforest: CMakeFiles/randomforest.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/qqh/projects/RandomForest/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable randomforest"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/randomforest.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/randomforest.dir/build: randomforest

.PHONY : CMakeFiles/randomforest.dir/build

CMakeFiles/randomforest.dir/requires: CMakeFiles/randomforest.dir/src/main.cpp.o.requires

.PHONY : CMakeFiles/randomforest.dir/requires

CMakeFiles/randomforest.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/randomforest.dir/cmake_clean.cmake
.PHONY : CMakeFiles/randomforest.dir/clean

CMakeFiles/randomforest.dir/depend:
	cd /home/qqh/projects/RandomForest/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/qqh/projects/RandomForest /home/qqh/projects/RandomForest /home/qqh/projects/RandomForest/build /home/qqh/projects/RandomForest/build /home/qqh/projects/RandomForest/build/CMakeFiles/randomforest.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/randomforest.dir/depend

