#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <dirent.h>
#include <stdbool.h>
#include <sys/stat.h>
#include <ctype.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include "rbuf2.h"

#define MAX_WORD 60
#define MAX_FILES 1024
#define MAX_PATH 256
#define MAX_UNIQUE 200000

using namespace std;

unsigned int djb_hash(const string &s, int num_reducers) {
    unsigned long hash = 5381;
    for (char c : s) hash = ((hash << 5) + hash) + c;
    return hash % num_reducers;
}

string trim_punct(const string &s) {
    if (s.empty()) return s;
    size_t start = 0;
    while (start < s.size() && ispunct(s[start])) start++;
    size_t end = s.size();
    while (end > start && ispunct(s[end - 1])) end--;
    return s.substr(start, end - start);
}

static void process_line(const string &line, RingBuffer &rb) {
    istringstream is(line);
    string token;
    while (getline(is, token, ' ')) {
        istringstream is2(token);
        string token2;
        while(getline(is2, token2, '\t')) {
            istringstream is3(token2);
            string token3;
            while(getline(is3, token3, '\r')) {
                istringstream is4(token3);
                string token4;
                while(getline(is4, token4, '\n')) rb.rb_push(make_pair(trim_punct(token4), 1));
            }
        }
    }
}

void reader(const char *path, RingBuffer &rb) {
    string line;
    ifstream f;
    f.open(path);
    if (!f.is_open()) {
        printf("invalid file\n");
        return;
    }
    while(getline(f, line)) process_line(line, rb);
    f.close();
}

int main(int argc, char **argv) {
    int num_readers, num_mappers, num_reducers;
    if (argc < 3) {
        cout << "usage: ./part1plus -path_to_directory -nthreads <-nreaders> <-nmappers> <-nreducers> (optional args in brackets)\n";
        return 1;
    }
    cout << argv[2] << endl;
    omp_set_num_threads(atoi(argv[2]));
    num_readers = (argc >= 4) ? atoi(argv[3]) : atoi(argv[2]);
    num_mappers = (argc >= 5) ? atoi(argv[4]) : atoi(argv[2]);
    num_reducers = (argc >= 6) ? atoi(argv[5]) : atoi(argv[2]);
    char path[MAX_PATH];
    char files[MAX_FILES][MAX_PATH];
    strcpy(path, argv[1]);
    int nfiles = 0;
    DIR *dp = opendir(path);
    struct dirent *ep = NULL;
    while ((ep = readdir(dp))) {
        if (strncmp(ep->d_name, ".", 1) == 0 || strncmp(ep->d_name, "..", 2) == 0) continue;
        sprintf(files[nfiles++], "%s/%s", path, ep->d_name);
    }
    closedir(dp);
    printf("nfiles: %d\n", nfiles);
    RingBuffer work_q;
    vector<RingBuffer> reducer_q(num_reducers);
    double ostart = omp_get_wtime();
    double reastart, mapstart, redstart, reaend, mapend, redend;
    #pragma omp parallel sections
    {
        #pragma omp section 
        {
            reastart = omp_get_wtime();
            #pragma omp parallel for num_threads(num_readers)
            for (size_t i = 0; i < nfiles; i++) reader(files[i], work_q);
            for (size_t i = 0; i < num_mappers; i++) work_q.rb_push(make_pair(("___EOF___"), -1));
            reaend = omp_get_wtime();
        }
        #pragma omp section
        {
            double lmstart = 0, lmend = 0;
            mapstart = omp_get_wtime();
            #pragma omp parallel num_threads(num_mappers)
            {
                #pragma omp single
                lmstart = omp_get_wtime();
                std::unordered_map<std::string, int> local_unique;
                std::pair<std::string, int> w;
                while (true) {
                    w = work_q.rb_pop();
                    if (w.second == -1) break;
                    auto it = local_unique.find(w.first);
                    if (it != local_unique.end()) it->second++;
                    else local_unique.insert(w);
                }
                for (const auto& p : local_unique) reducer_q[djb_hash(p.first, num_reducers)].rb_push(p);
                #pragma omp barrier
                #pragma omp single
                lmend = omp_get_wtime();
            }
            mapstart = lmstart;
            mapend = lmend;
        for (size_t i = 0; i < num_reducers; i++) reducer_q[i].rb_push(make_pair("___EOF___", -1));
        }
    }
    vector<std::unordered_map<std::string, int>> local_maps(num_reducers);
    int un = 0;
    redstart = omp_get_wtime();
    #pragma omp parallel num_threads(num_reducers) reduction(+:un)
    {
        int tid = omp_get_thread_num();
        while (true) {
            auto cur_word = reducer_q[tid].rb_pop();
            if (cur_word.second == -1) break;
            local_maps[tid][cur_word.first] += cur_word.second;
        }
        string outfile = "out_" + std::to_string(tid) + ".txt";
        ofstream f;
        f.open(outfile);
        for (const auto& [word, count] : local_maps[tid]) f << word << " , " << count << endl;
        f.close();
        un += local_maps[tid].size();
    }
    redend = omp_get_wtime();
    double oend = omp_get_wtime();
    cout << "Unique words: " << un << endl << "Overall time with " << atoi(argv[2]) << " threads: " << oend - ostart << endl;
    cout << "Reader section time with " << num_readers << " threads: " << reaend - reastart << endl;
    cout << "Mapper section time with " << num_mappers << " threads: " << mapend - mapstart << endl;
    cout << "Reducer section time with " << num_reducers << " threads: " << redend - redstart << endl;
    return 0;
}