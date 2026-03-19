#include <stdio.h>
#include "FileReader.h"
#include "Tolkien.h"
#include <stdlib.h>
#include "Voorhees.h"

//Usar gcc -I./CUSTOMLIBS/FileReader Tokenizer.c CUSTOMLIBS/FileReader/FileReader.c -o nombre_del_exe
//en la consola, para poder hacer un ejecutable.

int main() {

    const char* fileroute = "C:/Users/samue/Documents/DEV/KONOBOTLLM/DATASET/Final/Edited_Corpus.txt";

    // Tolkien *TolkienArray = malloc(WordCount(true, fileroute)*sizeof(Tolkien));

    //añadi esto 
    int TotalWords = WordCount(true, fileroute);
    Tolkien* TolkienArray = malloc(TotalWords * sizeof(Tolkien));

    if (TolkienArray == NULL) {
        fprintf(stderr, "Error al asignar memoria para TolkienArray.\n");
        return -1;
    }
    //
    int TID = 0;

    //int TotalWords = WordCount(true, fileroute);

    char* word = FileReader(fileroute);

    while (word != NULL) {
        // printf("%s\n", word);

        Tolkien tolkien = { word, TID++, 0, {0} };

        TolkienArray[tolkien.Id] = tolkien;

        free(word);

        word = FileReader(fileroute);
    }

    //Añadi esto 
    // Llamar a la función para escribir el array a un archivo JSON
    int result = TolkienToJSON(TolkienArray, TotalWords, "output.json");

    if (result == 0) {
        printf("El archivo JSON se ha creado exitosamente.\n");
    }
    else {
        printf("Hubo un error al crear el archivo JSON.\n");
    } // escrito de otra manera tmb pudiera ser "printf(result == 0 ? "JSON generado con éxito.\n" : "Error al generar JSON.\n");"


//for (int i = 0 ; i<TotalWords ; i++) {
//    PrintTolkien(TolkienArray[i]);
// }

    free(TolkienArray);
    return 0;
}

void PrintTolkien(Tolkien T) {
    printf("Word: %s,\nId: %d,\n", T.Word, T.Id);
}