#include "Voorhees.h"
#include <stdio.h>
#include <stdlib.h>

int TolkienToJSON(Tolkien* TolkienArray, int tokenCount, const char* outputPath) {
    if (!TolkienArray || tokenCount <= 0 || !outputPath) {
        fprintf(stderr, "Error: Parámetros inválidos.\n");
        return -1;
    }

    FILE* file = fopen(outputPath, "w");
    if (!file) {
        perror("Error al abrir el archivo JSON");
        return -1;
    }

    fprintf(file, "[\n");
    for (int i = 0; i < tokenCount; i++) {
        fprintf(file, "  {\n");
        fprintf(file, "    \"Word\": \"%s\",\n", TolkienArray[i].Word ? TolkienArray[i].Word : "null");
        fprintf(file, "    \"Id\": %d,\n", TolkienArray[i].Id);
        fprintf(file, "    \"Frequency\": %d,\n", TolkienArray[i].Frequency);

        // Embeddings (128 valores)
        fprintf(file, "    \"Embedding\": [");
        for (int j = 0; j < 128; j++) {
            fprintf(file, "%.6f%s", TolkienArray[i].Embedding[j], j < 127 ? ", " : "");
        }
        fprintf(file, "]\n  }%s\n", i < tokenCount - 1 ? "," : "");
    }
    fprintf(file, "]\n");

    fclose(file);
    return 0;
}