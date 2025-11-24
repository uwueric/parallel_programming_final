#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string>
#include <queue>
#include <utility> 
#include <omp.h>

class RingBuffer {
    public:
        void rb_push(std::pair<std::string, int> p);
        std::pair<std::string, int> rb_pop();
    private:
        std::queue<std::pair<std::string, int>> items;
        omp_lock_t l;
};







