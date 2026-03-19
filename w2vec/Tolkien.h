#ifndef TOLKIEN_H
#define TOLKIEN_H

typedef struct Tolkien
{
    char* Word;
    int Id;
    int Frequency;
    float Embedding[128];

} Tolkien;

void PrintTolkien(Tolkien T);

#endif
