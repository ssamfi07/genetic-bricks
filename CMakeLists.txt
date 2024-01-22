cmake_minimum_required(VERSION 3.10)

# set the project name
project(Nico_v1)

# add definitions
add_definitions(-lboost_log -DBOOST_ALL_DYN_LINK -lpthread -lboost_date_time)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -lboost_log -DBOOST_ALL_DYN_LINK -lpthread -lboost_date_time")

find_package(Boost COMPONENTS program_options log log_setup REQUIRED)
set(PROJECT_LIB ${PROJECT_LIB} ${Boost_LOG_LIBRARY} ${Boost_LOG_SETUP_LIBRARY})

# add the executable
add_executable(genetic
                src/genetic_algorithm.cpp)

include_directories(
    src
)

file(GLOB MY_FILES "*.csv")

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/csv_exports/wtf.csv
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(genetic pthread)
target_link_libraries(genetic ${PROJECT_LIB})