#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <omp.h>
#include <unistd.h>
#include <string>
#include <utility> 
#include "rbuf2.h"




void RingBuffer::rb_push(std::pair<std::string, int> p) {
    omp_set_lock(&l);
    items.push(p);
    omp_unset_lock(&l);
}

std::pair<std::string, int> RingBuffer::rb_pop() {
    while (true) {
        omp_set_lock(&l);
        if (!items.empty()) {
            std::pair<std::string, int> out = items.front();
            items.pop();
            omp_unset_lock(&l);
            return out;
        }
        omp_unset_lock(&l);
        usleep(100);
    }
}