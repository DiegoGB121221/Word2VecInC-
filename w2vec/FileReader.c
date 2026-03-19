#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

const char* FileReader(const char* fileroute) {
    static FILE* file = NULL;

    char char_buffer;
    char* string_buffer = malloc(256 * sizeof(char));
    int char_counter = 0;

    // if (file == NULL) {
    //     printf("ERROR : No se pudo Acceder al archivo");
    // }

    if (string_buffer == NULL) {
        printf("ERROR : No se pudo asignar memoria para el buffer\n");
        return NULL;
    }

    if (file == NULL) {
        file = fopen(fileroute, "r");
        if (file == NULL) {
            printf("ERROR : No se pudo Acceder al archivo");
            free(string_buffer);
            return NULL;
        }
    }

    while ((char_buffer = fgetc(file)) != EOF) {
        // printf("%c", char_buffer);
        //tengo que hacer que se guarde un string aqui, y luego regresarlo 

        if (char_buffer == '\n') {
            //termina la palabra

            string_buffer[char_counter] = '\0';
            return string_buffer;
        }

        string_buffer[char_counter++] = char_buffer;

        if (char_counter >= 256) {
            string_buffer = realloc(string_buffer, (char_counter + 1) * sizeof(char));
            if (string_buffer == NULL) {
                printf("ERROR : No se pudo reasignar memoria para el buffer\n");
                fclose(file);
                return NULL;
            }
        }

    }

    if (char_counter == 0 && char_buffer == EOF) {
        fclose(file);
        file = NULL;
        free(string_buffer);
        return NULL;
    }

    string_buffer[char_counter] = '\0';
    fclose(file);
    file = NULL;
    return string_buffer;

}

int WordCount(bool skip, const char* fileroute) {
    int Word_Count = 0;

    FILE* file;

    char char_buffer;

    file = fopen(fileroute, "r");

    if (file == NULL) {
        printf("ERROR : No se pudo Acceder al archivo");
    }

    while ((char_buffer = fgetc(file)) != EOF) {
        if (char_buffer == '\n') {
            Word_Count += 1;
        }
    }

    fclose(file);

    if (!skip) {
        printf("Im Count Word,\nThe Word Count,\nCounter of Words,\nYour Word Count is: %d", Word_Count);
    }

    return Word_Count;
}