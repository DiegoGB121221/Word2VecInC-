#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <jansson.h> // Biblioteca para manejar JSON

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

// Estructuras de datos
typedef float real;

struct vocab_word {
    long long cn;                     // frecuencia del token
    char word[MAX_STRING];            // texto del token (opcional)
    long long token_id;               // ID del token
};

// Variables globales
char train_file[MAX_STRING], output_file[MAX_STRING];
struct vocab_word* vocab;
int binary = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 1;
int* vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3;
real* syn0, * syn1neg, * expTable;
clock_t start;

// Tabla hash para mapear token_id a índices de vocabulario
#define VOCAB_HASH_SIZE 2000000
long long* token_id_to_index;

// Funciones
void InitUnigramTable() {
    // Implementar tabla unigram para negative sampling
    // Similar a la versión original
}

int GetWordHash(char* word) {
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
    hash = hash % vocab_max_size;
    return hash;
}

int AddWordToVocab(char* word, long long token_id) {
    unsigned int hash;
    if (vocab_size >= vocab_max_size) {
        vocab_max_size += 1000;
        vocab = (struct vocab_word*)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
        token_id_to_index = (long long*)realloc(token_id_to_index, vocab_max_size * sizeof(long long));
    }

    strcpy(vocab[vocab_size].word, word);
    vocab[vocab_size].token_id = token_id;
    vocab[vocab_size].cn = 0;

    // Mapear token_id a índice de vocabulario
    if (token_id >= VOCAB_HASH_SIZE) {
        printf("Token ID %lld excede el tamaño máximo de la tabla hash\n", token_id);
        exit(1);
    }
    token_id_to_index[token_id] = vocab_size;

    vocab_size++;
    return vocab_size - 1;
}

void SortVocab() {
    int a, b;
    struct vocab_word swap;
    for (a = 1; a < vocab_size; a++) {
        b = a;
        swap = vocab[b];
        while ((b > 0) && (swap.cn > vocab[b - 1].cn)) {
            vocab[b] = vocab[b - 1];
            b--;
        }
        vocab[b] = swap;
    }
}

void LearnVocabFromJSON() {
    json_t* root, * sentence, * token_array, * token_id_array;
    json_error_t error;
    size_t index;
    json_t* value, * id_value;
    long long token_id;
    char word[MAX_STRING];

    root = json_load_file(train_file, 0, &error);
    if (!root) {
        fprintf(stderr, "Error en JSON: línea %d: %s\n", error.line, error.text);
        exit(1);
    }

    if (!json_is_array(root)) {
        fprintf(stderr, "El JSON debe contener un array de oraciones\n");
        json_decref(root);
        exit(1);
    }

    // Inicializar tabla hash de token_id
    token_id_to_index = (long long*)calloc(VOCAB_HASH_SIZE, sizeof(long long));
    for (long long i = 0; i < VOCAB_HASH_SIZE; i++) {
        token_id_to_index[i] = -1;
    }

    json_array_foreach(root, index, sentence) {
        if (!json_is_object(sentence)) continue;

        token_array = json_object_get(sentence, "tokens");
        token_id_array = json_object_get(sentence, "token_ids");

        if (!json_is_array(token_array) || !json_is_array(token_id_array)) continue;

        size_t token_count = json_array_size(token_array);
        if (token_count != json_array_size(token_id_array)) continue;

        train_words += token_count;

        for (size_t i = 0; i < token_count; i++) {
            value = json_array_get(token_array, i);
            id_value = json_array_get(token_id_array, i);

            if (!json_is_string(value) || !json_is_integer(id_value)) continue;

            token_id = json_integer_value(id_value);
            strncpy(word, json_string_value(value), MAX_STRING - 1);
            word[MAX_STRING - 1] = '\0';

            if (token_id_to_index[token_id] == -1) {
                AddWordToVocab(word, token_id);
                vocab[vocab_size - 1].cn = 1;
            }
            else {
                vocab[token_id_to_index[token_id]].cn++;
            }
        }
    }

    json_decref(root);
    SortVocab();

    if (debug_mode > 0) {
        printf("Vocab size: %lld\n", vocab_size);
        printf("Words in train file: %lld\n", train_words);
    }
}

void InitNet() {
    long long a, b;
    unsigned long long next_random = 1;

    syn0 = (real*)malloc((long long)vocab_size * layer1_size * sizeof(real));
    if (syn0 == NULL) { printf("Memory allocation failed\n"); exit(1); }

    syn1neg = (real*)malloc((long long)vocab_size * layer1_size * sizeof(real));
    if (syn1neg == NULL) { printf("Memory allocation failed\n"); exit(1); }

    for (a = 0; a < vocab_size; a++) {
        for (b = 0; b < layer1_size; b++) {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
        }
    }

    for (a = 0; a < vocab_size; a++) {
        for (b = 0; b < layer1_size; b++) {
            syn1neg[a * layer1_size + b] = 0;
        }
    }
}

void TrainModel() {
    json_t* root, * sentence, * token_id_array;
    json_error_t error;
    size_t index;
    json_t* id_value;
    long long word, last_word, sentence_length = 0, sentence_position = 0;
    long long word_count = 0, last_word_count = 0;
    long long sen[MAX_SENTENCE_LENGTH];
    long long l1, l2, target, label;
    unsigned long long next_random = (long long)1;
    real f, g;
    clock_t now;

    root = json_load_file(train_file, 0, &error);
    if (!root) {
        fprintf(stderr, "Error en JSON: línea %d: %s\n", error.line, error.text);
        exit(1);
    }

    if (!json_is_array(root)) {
        fprintf(stderr, "El JSON debe contener un array de oraciones\n");
        json_decref(root);
        exit(1);
    }

    while (1) {
        if (word_count - last_word_count > 10000) {
            word_count_actual += word_count - last_word_count;
            last_word_count = word_count;

            if ((debug_mode > 1)) {
                now = clock();
                printf("%cAlpha: %f  Progress: %.2f%%  Words/sec: %.2fk  ", 13, alpha,
                    word_count_actual / (real)(train_words + 1) * 100,
                    word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
                fflush(stdout);
            }

            alpha = starting_alpha * (1 - word_count_actual / (real)(train_words + 1));
            if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
        }

        if (sentence_length == 0) {
            if (index >= json_array_size(root)) break;

            sentence = json_array_get(root, index);
            index++;

            token_id_array = json_object_get(sentence, "token_ids");
            if (!json_is_array(token_id_array)) continue;

            sentence_length = 0;
            json_array_foreach(token_id_array, index, id_value) {
                if (!json_is_integer(id_value)) continue;

                long long token_id = json_integer_value(id_value);
                word = token_id_to_index[token_id];

                if (word == -1) continue; // Token no está en el vocabulario

                word_count++;

                // Submuestreo de palabras frecuentes
                if (sample > 0) {
                    real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
                    next_random = next_random * (unsigned long long)25214903917 + 11;
                    if (ran < (next_random & 0xFFFF) / (real)65536) continue;
                }

                sen[sentence_length] = word;
                sentence_length++;
                if (sentence_length >= MAX_SENTENCE_LENGTH) break;
            }
            sentence_position = 0;
        }

        if (word_count > train_words) break;
        if (sentence_length == 0) continue;

        word = sen[sentence_position];
        if (word == -1) continue;

        for (int a = 0; a < window * 2 + 1; a++) {
            if (a == window) continue; // Saltar la palabra central

            int c = sentence_position - window + a;
            if (c < 0 || c >= sentence_length) continue;

            last_word = sen[c];
            if (last_word == -1) continue;

            // Entrenamiento Skip-Gram con Negative Sampling
            l1 = last_word * layer1_size;
            real* neu1e = (real*)calloc(layer1_size, sizeof(real));

            // NEGATIVE SAMPLING
            for (int d = 0; d < negative + 1; d++) {
                if (d == 0) {
                    target = word;
                    label = 1;
                }
                else {
                    next_random = next_random * (unsigned long long)25214903917 + 11;
                    target = table[(next_random >> 16) % table_size];
                    if (target == 0) target = next_random % (vocab_size - 1) + 1;
                    if (target == word) continue;
                    label = 0;
                }

                l2 = target * layer1_size;
                f = 0;
                for (int c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];

                if (f > MAX_EXP) g = (label - 1) * alpha;
                else if (f < -MAX_EXP) g = (label - 0) * alpha;
                else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;

                for (int c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
                for (int c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
            }

            // Aprender pesos entre la palabra de entrada y la salida
            for (int c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];

            free(neu1e);
        }

        sentence_position++;
        if (sentence_position >= sentence_length) {
            sentence_length = 0;
        }
    }

    json_decref(root);
}

int main(int argc, char** argv) {
    // Parseo de argumentos (similar al original)

    // Inicializar estructuras
    vocab = (struct vocab_word*)calloc(vocab_max_size, sizeof(struct vocab_word));
    vocab_hash = (int*)calloc(vocab_max_size, sizeof(int));
    expTable = (real*)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));

    for (int i = 0; i < EXP_TABLE_SIZE; i++) {
        expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);
        expTable[i] = expTable[i] / (expTable[i] + 1);
    }

    LearnVocabFromJSON();
    InitNet();
    InitUnigramTable();
    start = clock();
    TrainModel();

    // Guardar vectores
    FILE* fo = fopen(output_file, "wb");
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);

    for (long long a = 0; a < vocab_size; a++) {
        fprintf(fo, "%lld ", vocab[a].token_id); // Guardar token_id en lugar de la palabra
        if (binary) {
            for (int b = 0; b < layer1_size; b++) {
                fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
            }
        }
        else {
            for (int b = 0; b < layer1_size; b++) {
                fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
            }
        }
        fprintf(fo, "\n");
    }

    fclose(fo);

    // Liberar memoria
    free(vocab);
    free(vocab_hash);
    free(token_id_to_index);
    free(syn0);
    free(syn1neg);
    free(expTable);

    return 0;
}