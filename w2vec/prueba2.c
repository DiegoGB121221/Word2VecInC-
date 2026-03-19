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
#define MAX_WORD_LENGTH 200
#define TABLE_SIZE 10000000
#define EMBEDDING_SIZE 128


// Estructuras de datos
typedef float real;

struct vocab_word {
    char word[MAX_STRING];
    long long cn;
    long long token_id;
    real* embedding; 
};

// Variables globales
char train_file[MAX_STRING] = "Tokens2.json"; // Archivo de entrada
char output_file[MAX_STRING] = "embedings.json";
struct vocab_word* vocab;
int binary = 0, debug_mode = 2, window = 3, min_count = 5, num_threads = 1;
int* vocab_hash;
long long vocab_max_size = 10000000, vocab_size = 0, layer1_size = EMBEDDING_SIZE;
long long train_words = 0, word_count_actual = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3;
real* syn0, * syn1neg, * expTable;
clock_t start;
int* unigram_table;
long long table_size = 0;
int negative = 5; // Número de muestras negativas por palabra positiva
long long* token_id_to_index;


// Tabla hash para mapear token_id a índices de vocabulario
#define VOCAB_HASH_SIZE 20000000
long long* token_id_to_index;

// Funciones
void InitUnigramTable() {
    long long a, i;
    double train_words_pow = 0;
    double d1, power = 0.75; // Factor de suavizado (usualmente 0.75)

    // Calcular la suma de frecuencias elevadas a power
    for (a = 0; a < vocab_size; a++) {
        train_words_pow += pow(vocab[a].cn, power);
    }

    // Asignar memoria para la tabla unigram
    unigram_table = (int*)malloc(TABLE_SIZE * sizeof(int));
    if (unigram_table == NULL) {
        printf("Error: No se pudo asignar memoria para la tabla unigram\n");
        exit(1);
    }

    // Llenar la tabla con índices de palabras según su frecuencia
    i = 0;
    d1 = pow(vocab[i].cn, power) / train_words_pow;
    for (a = 0; a < TABLE_SIZE; a++) {
        unigram_table[a] = i;
        if (a / (double)TABLE_SIZE > d1) {
            i++;
            d1 += pow(vocab[i].cn, power) / train_words_pow;
        }
        if (i >= vocab_size) i = vocab_size - 1; // Prevenir desbordamiento
    }

    table_size = TABLE_SIZE;

    if (debug_mode > 0) {
        printf("Tabla unigram creada con tamaño %lld\n", table_size);
    }
}


int GetWordHash(char* word) {
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
    hash = hash % vocab_max_size;
    return hash;
}

void InitializeEmbedding(real* embedding, int size) {
    for (int i = 0; i < size; i++) {
        embedding[i] = 0.0f;
    }
}

int AddWordToVocab(char* word, long long token_id, long long frequency, real* embedding) {
    // Verificar parámetros de entrada
    if (word == NULL) {
        fprintf(stderr, "Error: palabra NULL pasada a AddWordToVocab\n");
        exit(EXIT_FAILURE);
    }

    if (vocab_size >= vocab_max_size) {
        vocab_max_size += 1000;

        // Reasignar memoria para vocab con verificación de error
        struct vocab_word* new_vocab = (struct vocab_word*)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
        if (new_vocab == NULL) {
            fprintf(stderr, "Error: no se pudo reasignar memoria para vocab\n");
            free(vocab);
            exit(EXIT_FAILURE);
        }
        vocab = new_vocab;

        // Reasignar memoria para token_id_to_index con verificación de error
        long long* new_token_id_to_index = (long long*)realloc(token_id_to_index, vocab_max_size * sizeof(long long));
        if (new_token_id_to_index == NULL) {
            fprintf(stderr, "Error: no se pudo reasignar memoria para token_id_to_index\n");
            free(vocab);
            free(token_id_to_index);
            exit(EXIT_FAILURE);
        }
        token_id_to_index = new_token_id_to_index;
    }

    // Inicializar el nuevo elemento del vocabulario trainfile
    vocab[vocab_size].embedding = NULL; // Inicializar a NULL

    // Copiar la palabra al vocabulario
    strncpy(vocab[vocab_size].word, word, MAX_STRING - 1);
    vocab[vocab_size].word[MAX_STRING - 1] = '\0';

    vocab[vocab_size].token_id = token_id;
    vocab[vocab_size].cn = frequency;

    // Asignar memoria para el embedding
    vocab[vocab_size].embedding = (real*)malloc(layer1_size * sizeof(real));
    if (vocab[vocab_size].embedding == NULL) {
        fprintf(stderr, "Error: no se pudo asignar memoria para embedding\n");
        exit(EXIT_FAILURE);
    }

    // Copiar el embedding si se proporciona, de lo contrario inicializar con ceros
    if (embedding != NULL) {
        memcpy(vocab[vocab_size].embedding, embedding, layer1_size * sizeof(real));
    }
    else {
        InitializeEmbedding(vocab[vocab_size].embedding, layer1_size);
    }

    // Mapear token_id a índice de vocabulario
    if (token_id >= VOCAB_HASH_SIZE) {
        fprintf(stderr, "Error: Token ID %lld excede el tamaño máximo de la tabla hash (%d)\n",
            token_id, VOCAB_HASH_SIZE);
        exit(EXIT_FAILURE);
    }
    token_id_to_index[token_id] = vocab_size;

    vocab_size++;
    return vocab_size - 1;
}


void SortVocab() {
    int a, b;
    struct vocab_word swap;
    real* temp_embedding;

    for (a = 1; a < vocab_size; a++) {
        b = a;
        swap = vocab[b];
        temp_embedding = vocab[b].embedding;

        while ((b > 0) && (swap.cn > vocab[b - 1].cn)) {
            vocab[b] = vocab[b - 1];
            vocab[b].embedding = vocab[b - 1].embedding;
            b--;
        }

        vocab[b] = swap;
        vocab[b].embedding = temp_embedding;
    }
}

void FreeVocab() {
    // Liberar memoria de los embeddings
    for (long long i = 0; i < vocab_size; i++) {
        if (vocab[i].embedding != NULL) {
            free(vocab[i].embedding);
            vocab[i].embedding = NULL;
        }
    }

    // Liberar las demás estructuras
    if (vocab != NULL) free(vocab);
    if (token_id_to_index != NULL) free(token_id_to_index);
    if (vocab_hash != NULL) free(vocab_hash);
}



int LearnVocabFromJSON() {
    json_t* root, * word_obj;
    json_error_t error;
    size_t index;
    json_t* word_value, * id_value, * freq_value, * embed_value;
    long long token_id, frequency;
    char word[MAX_STRING];

    root = json_load_file(train_file, 0, &error);
    if (!root) {
        fprintf(stderr, "Error en JSON: línea %d: %s\n", error.line, error.text);
        exit(1);
    }

    if (!json_is_array(root)) {
        fprintf(stderr, "El JSON debe contener un array de palabras\n");
        json_decref(root);
        exit(1);
    }

    // Inicialización segura de token_id_to_index
    token_id_to_index = (long long*)calloc(VOCAB_HASH_SIZE, sizeof(long long));
    if (token_id_to_index == NULL) {
        fprintf(stderr, "Error: No se pudo asignar memoria para token_id_to_index\n");
        json_decref(root);
        exit(EXIT_FAILURE);
    }

    for (long long i = 0; i < VOCAB_HASH_SIZE; i++) {
        token_id_to_index[i] = -1;
    }

    // Inicializar el vocabulario
    vocab = (struct vocab_word*)calloc(vocab_max_size, sizeof(struct vocab_word));
    if (vocab == NULL) {
        fprintf(stderr, "Error: No se pudo asignar memoria para vocab\n");
        free(token_id_to_index);
        json_decref(root);
        exit(EXIT_FAILURE);
    }

    json_array_foreach(root, index, word_obj) {
        if (!json_is_object(word_obj)) continue;

        // Obtener los campos del objeto palabra
        word_value = json_object_get(word_obj, "Word");
        id_value = json_object_get(word_obj, "Id");
        freq_value = json_object_get(word_obj, "Frequency");
        embed_value = json_object_get(word_obj, "Embedding");

        if (!json_is_string(word_value) || !json_is_integer(id_value) ||
            !json_is_integer(freq_value) || !json_is_array(embed_value)) {
            fprintf(stderr, "Advertencia: Estructura incorrecta en el elemento %zu\n", index);
            continue;
        }

        token_id = json_integer_value(id_value);
        frequency = json_integer_value(freq_value);

        // Validación del token_id
        if (token_id < 0 || token_id >= VOCAB_HASH_SIZE) {
            fprintf(stderr, "Error: Token ID %lld fuera de rango [0, %d) en elemento %zu\n",
                token_id, VOCAB_HASH_SIZE, index);
            continue;
        }

        strncpy(word, json_string_value(word_value), MAX_STRING - 1);
        word[MAX_STRING - 1] = '\0';

        // Procesar el embedding
        size_t embed_size = json_array_size(embed_value);
        if (embed_size != layer1_size) {
            fprintf(stderr, "Advertencia: Tamaño de embedding incorrecto (%zu) en elemento %zu. Esperado: %lld\n",
                embed_size, index, layer1_size);
            continue;
        }

        real* embedding = (real*)malloc(layer1_size * sizeof(real));
        if (embedding == NULL) {
            fprintf(stderr, "Error: No se pudo asignar memoria para embedding temporal\n");
            continue;
        }

        for (size_t i = 0; i < layer1_size; i++) {
            json_t* embed_element = json_array_get(embed_value, i);
            if (!json_is_real(embed_element) && !json_is_integer(embed_element)) {
                fprintf(stderr, "Advertencia: Valor de embedding no numérico en elemento %zu, posición %zu\n", index, i);
                embedding[i] = 0.0;
                continue;
            }
            embedding[i] = (real)json_real_value(embed_element);
        }

        if (token_id_to_index[token_id] == -1) {
            AddWordToVocab(word, token_id, frequency, embedding);
            train_words += frequency;
        }
        else {
            // Si el token_id ya existe, actualizamos la frecuencia
            long long vocab_index = token_id_to_index[token_id];
            vocab[vocab_index].cn += frequency;
            train_words += frequency;

            // Actualizar el embedding si es diferente de cero
            for (size_t i = 0; i < layer1_size; i++) {
                if (embedding[i] != 0.0) {
                    vocab[vocab_index].embedding[i] = embedding[i];
                }
            }
        }

        free(embedding);
    }
    //InitializeEmbedding
    json_decref(root);
    SortVocab();

    if (debug_mode > 0) {
        printf("Vocab size: %lld\n", vocab_size);
        printf("Words in train file: %lld\n", train_words);

        // Opcional: Imprimir algunos embeddings para verificación
        if (debug_mode > 1 && vocab_size > 0) {
            printf("\nEjemplo de embedding para la palabra '%s':\n", vocab[0].word);
            for (int i = 0; i < (layer1_size < 10 ? layer1_size : 10); i++) {
                printf("%.6f ", vocab[0].embedding[i]);
            }
            printf("\n");
        }
    }

    return 0;
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


//window

void TrainModel() {
    json_t* root, * word_obj;
    json_error_t error;
    size_t index;
    json_t* word_value, * id_value, * embed_value;
    long long word, last_word, sentence_length = 0, sentence_position = 0;
    long long word_count = 0, last_word_count = 0;
    long long sen[MAX_SENTENCE_LENGTH];
    long long l1, l2, target, label;
    unsigned long long next_random = (long long)time(NULL);
    real f, g;
    clock_t now;

    // Cargar el archivo JSON con palabras individuales
    root = json_load_file(train_file, 0, &error);
    if (!root) {
        fprintf(stderr, "Error en JSON: línea %d: %s\n", error.line, error.text);
        exit(1);
    }

    if (!json_is_array(root)) {
        fprintf(stderr, "El JSON debe contener un array de palabras\n");
        json_decref(root);
        exit(1);
    }

    printf("Iniciando entrenamiento con %zu palabras...\n", json_array_size(root));

    start = clock();

    // Primero construimos una lista de palabras para formar "oraciones" artificiales
    long long* word_indices = (long long*)malloc(json_array_size(root) * sizeof(long long));
    if (!word_indices) {
        fprintf(stderr, "Error al asignar memoria para word_indices\n");
        exit(1);
    }

    size_t total_words = 0;
    json_array_foreach(root, index, word_obj) {
        if (!json_is_object(word_obj)) continue;

        id_value = json_object_get(word_obj, "Id");
        if (!json_is_integer(id_value)) continue;

        long long token_id = json_integer_value(id_value);
        word = token_id_to_index[token_id];
        if (word == -1) continue; // Saltar palabras fuera del vocabulario

        word_indices[total_words++] = word;
    }

    // Ahora realizamos el entrenamiento con ventanas deslizantes sobre la secuencia de palabras
    while (1) {
        // Mostrar progreso
        if (word_count - last_word_count > 10000) {
            word_count_actual += word_count - last_word_count;
            last_word_count = word_count;

            if ((debug_mode > 1)) {
                now = clock();
                printf("\rAlpha: %f  Progreso: %.2f%%  Palabras/seg: %.2fk  ", alpha,
                    word_count_actual / (real)(train_words + 1) * 100,
                    word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
                fflush(stdout);
            }

            // Actualizar tasa de aprendizaje
            alpha = starting_alpha * (1 - word_count_actual / (real)(train_words + 1));
            if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
        }

        // Construir una "oración" artificial de longitud fija
        if (sentence_length == 0) {
            long long start_pos = next_random % (total_words - MAX_SENTENCE_LENGTH);

            sentence_length = 0;
            for (long long i = 0; i < MAX_SENTENCE_LENGTH && (start_pos + i) < total_words; i++) {
                word = word_indices[start_pos + i];

                // Aplicar submuestreo de palabras frecuentes
                if (sample > 0) {
                    real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) *
                        (sample * train_words) / vocab[word].cn;
                    next_random = next_random * (unsigned long long)25214903917 + 11;
                    if (ran < (next_random & 0xFFFF) / (real)65536) continue;
                }

                sen[sentence_length++] = word;
                word_count++;
            }
            sentence_position = 0;
        }

        if (word_count > train_words) break;
        if (sentence_length == 0) continue;

        word = sen[sentence_position];
        if (word == -1) {
            sentence_position++;
            if (sentence_position >= sentence_length) {
                sentence_length = 0;
            }
            continue;
        }

        // Entrenamiento Skip-Gram con Negative Sampling
        l1 = word * layer1_size;
        real* neu1e = (real*)calloc(layer1_size, sizeof(real));
        if (neu1e == NULL) {
            fprintf(stderr, "Error al asignar memoria para neu1e\n");
            exit(1);
        }

        // Recorrer ventana de contexto
        int window_actual = next_random % window + 1; // Ventana variable
        for (int a = 0; a < window_actual * 2; a++) {
            if (a == window_actual) continue; // Saltar la palabra central

            int c = sentence_position - window_actual + a;
            if (c < 0 || c >= sentence_length) continue;

            last_word = sen[c];
            if (last_word == -1) continue;

            // Negative Sampling
            for (int d = 0; d < negative + 1; d++) {
                if (d == 0) {
                    target = last_word;
                    label = 1;
                }
                else {
                    next_random = next_random * (unsigned long long)25214903917 + 11;
                    target = unigram_table[(next_random >> 16) % table_size];
                    if (target == 0) target = next_random % (vocab_size - 1) + 1;
                    if (target == word) continue;
                    label = 0;
                }

                l2 = target * layer1_size;
                f = 0;
                for (int c = 0; c < layer1_size; c++) {
                    f += syn0[l1 + c] * syn1neg[l2 + c];
                }

                // Calcular gradiente
                if (f > MAX_EXP) g = (label - 1) * alpha;
                else if (f < -MAX_EXP) g = (label - 0) * alpha;
                else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;

                // Actualizar vectores de error
                for (int c = 0; c < layer1_size; c++) {
                    neu1e[c] += g * syn1neg[l2 + c];
                }

                // Actualizar vectores negativos
                for (int c = 0; c < layer1_size; c++) {
                    syn1neg[l2 + c] += g * syn0[l1 + c];
                }
            }
        }

        // Actualizar vectores de palabra
        for (int c = 0; c < layer1_size; c++) {
            syn0[l1 + c] += neu1e[c];
        }

        free(neu1e);
        sentence_position++;

        // Verificar fin de "oración" artificial
        if (sentence_position >= sentence_length) {
            sentence_length = 0;
        }
    }

    free(word_indices);
    json_decref(root);
    printf("\nEntrenamiento completado. Palabras procesadas: %lld\n", word_count_actual);
}
void FreeMemory() {
    if (vocab) free(vocab);
    if (vocab_hash) free(vocab_hash);
    if (token_id_to_index) free(token_id_to_index);
    if (syn0) free(syn0);
    if (syn1neg) free(syn1neg);
    if (expTable) free(expTable);
    if (unigram_table) free(unigram_table);
}

void SaveVectorsToJSON() {
    FILE* fo = fopen(output_file, "w");
    if (fo == NULL) {
        fprintf(stderr, "Error: No se pudo abrir el archivo de salida %s\n", output_file);
        FreeMemory();
        exit(EXIT_FAILURE);
    }

    // Encabezado del JSON
    fprintf(fo, "{\n");
    fprintf(fo, "  \"metadata\": {\n");
    fprintf(fo, "    \"vocab_size\": %lld,\n", vocab_size);
    fprintf(fo, "    \"vector_dimension\": %lld,\n", layer1_size);
    fprintf(fo, "    \"format_version\": \"1.0\"\n");
    fprintf(fo, "  },\n");
    fprintf(fo, "  \"embeddings\": [\n");

    // Escribir cada embedding
    for (long long a = 0; a < vocab_size; a++) {
        fprintf(fo, "    {\n");
        fprintf(fo, "      \"token_id\": %lld,\n", vocab[a].token_id);
        fprintf(fo, "      \"word\": \"%s\",\n", vocab[a].word); // Asume que vocab_word tiene campo 'word'
        fprintf(fo, "      \"count\": %lld,\n", vocab[a].cn);   // Frecuencia de la palabra
        fprintf(fo, "      \"vector\": [");

        // Escribir componentes del vector
        for (int b = 0; b < layer1_size; b++) {
            fprintf(fo, "%.6f", syn0[a * layer1_size + b]);
            if (b < layer1_size - 1) fprintf(fo, ", ");
        }

        fprintf(fo, "]");

        // Cerrar objeto (sin coma después del último)
        if (a < vocab_size - 1) {
            fprintf(fo, "\n    },\n");
        }
        else {
            fprintf(fo, "\n    }\n");
        }
    }

    // Cerrar documento JSON
    fprintf(fo, "  ]\n");
    fprintf(fo, "}\n");

    fclose(fo);
}


int main(int argc, char** argv) {
    // Parseo de argumentos (similar al original)

    // Inicializar estructuras
    vocab = (struct vocab_word*)calloc(vocab_max_size, sizeof(struct vocab_word));
    if (vocab == NULL) {
        fprintf(stderr, "Error: No se pudo asignar memoria para vocab\n");
        exit(EXIT_FAILURE);
    }

    vocab_hash = (int*)calloc(vocab_max_size, sizeof(int));
    if (vocab_hash == NULL) {
        fprintf(stderr, "Error: No se pudo asignar memoria para vocab_hash\n");
        free(vocab);
        exit(EXIT_FAILURE);
    }

    // Asignar memoria exacta para la tabla exp (sin +1)
    expTable = (real*)malloc(EXP_TABLE_SIZE * sizeof(real));
    if (expTable == NULL) {
        fprintf(stderr, "Error: No se pudo asignar memoria para expTable\n");
        free(vocab);
        free(vocab_hash);
        exit(EXIT_FAILURE);
    }

    // Inicializar tabla exp (solo los elementos necesarios)
    for (int i = 0; i < EXP_TABLE_SIZE; i++) {
        // Precomputar valores de sigmoide
        expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);
        expTable[i] = expTable[i] / (expTable[i] + 1);  // Función sigmoide
    }

    // Verificar errores en las funciones de inicialización
    if (LearnVocabFromJSON() != 0) {
        // Manejar error
        FreeMemory();
        exit(EXIT_FAILURE);
    }

    printf("Paso 1 completado");
    InitNet();
    printf("Paso 2 completado");
    InitUnigramTable();
    printf("Paso 3 completado");

    start = clock();
    TrainModel();
    printf("Paso 4 completado");

    // Guardar vectores en formato JSON
    SaveVectorsToJSON();
    printf("Paso 5 completado");

    FreeMemory();
    return 0;
}

