#include "conway.h"

std::ostream &operator<<(std::ostream &os, bool* arr)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
            os << (arr[i + j * N] ? "o" : ".") << " ";
        os << std::endl;
    }
    return os;
}

std::ostream &operator<<(std::ostream &os, bool** arr)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
            os << (arr[i][j] ? "o" : ".") << " ";
        os << std::endl;
    }
    return os;
}


void init(bool* &arr)
{
    srandom(time(nullptr));

    size_t t = N * M;
    while (t--) {
        size_t i = random() % N;
        size_t j = random() % M;

        arr[i + j * N] = true;
    }
}

void init(bool** &arr)
{
    srandom(time(nullptr));

    size_t t = N * M;
    while (t--) {
        size_t i = random() % N;
        size_t j = random() % M;

        arr[i][j] = true;
    }
}