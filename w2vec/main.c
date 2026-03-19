#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

// Estructuras de datos
typedef float real;                    // Precision del vector (float o double)

struct vocab_word {
    long long cn;                     // frecuencia de la palabra
    char word[MAX_STRING];            // texto de la palabra
};

// Variables globales
char train_file[MAX_STRING], output_file[MAX_STRING];
struct vocab_word* vocab;
int binary = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 1;
int* vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, file_size = 0;
real alpha = 0.025, starting_alpha, sample = 1e-3;
real* syn0, * syn1neg, * expTable;
clock_t start;

// Funciones
/*/void InitUnigramTable() {
    // Implementar tabla unigram para negative sampling
}*/

/*void ReadWord(char* word, FILE* fin) {
    // Leer una palabra del archivo
    int a = 0, ch;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == 13) continue;
        if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
            if (a > 0) {
                if (ch == '\n') ungetc(ch, fin);
                break;
            }
            continue;
        }
        word[a] = ch;
        a++;
        if (a >= MAX_STRING - 1) a--; // Truncar palabras muy largas
    }
    word[a] = 0;
}*/

/*int GetWordHash(char* word) {
    // Hash para buscar palabras en el vocabulario
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
    hash = hash % vocab_max_size;
    return hash;
}*/

/*int SearchVocab(char* word) {
    // Buscar una palabra en el vocabulario
    unsigned int hash = GetWordHash(word);
    while (1) {
        if (vocab_hash[hash] == -1) return -1;
        if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
        hash = (hash + 1) % vocab_max_size;
    }
    return -1;
}*/

/*int AddWordToVocab(char* word) {
    // Añadir palabra al vocabulario
    unsigned int hash;
    if (vocab_size >= vocab_max_size) {
        vocab_max_size += 1000;
        vocab = (struct vocab_word*)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
    }
    strcpy(vocab[vocab_size].word, word);
    vocab[vocab_size].cn = 0;
    vocab_size++;
    // Rehash (necesario cuando cambia el tamaño de la tabla)
    hash = GetWordHash(word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_max_size;
    vocab_hash[hash] = vocab_size - 1;
    return vocab_size - 1;
}*/

/*void SortVocab() {
    // Ordenar vocabulario por frecuencia
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
}*/

/*void ReduceVocab() {
    // Reducir vocabulario eliminando palabras raras
    int a, b = 0;
    for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_count) {
        vocab[b] = vocab[a];
        b++;
    }
    else free(vocab[a].word);
    vocab_size = b;
}*/

void LearnVocabFromTrainFile() {
    // Aprender vocabulario desde archivo de entrenamiento
    char word[MAX_STRING];
    FILE* fin;
    long long a, i;
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    fin = fopen(train_file, "rb");
    if (fin == NULL) {
        printf("ERROR: archivo de entrenamiento no encontrado\n");
        exit(1);
    }
    vocab_size = 0;
    AddWordToVocab("</s>");
    while (1) {
        ReadWord(word, fin);
        if (feof(fin)) break;
        train_words++;
        if ((debug_mode > 1) && (train_words % 100000 == 0)) {
            printf("%lldK%c", train_words / 1000, 13);
            fflush(stdout);
        }
        i = SearchVocab(word);
        if (i == -1) {
            a = AddWordToVocab(word);
            vocab[a].cn = 1;
        }
        else vocab[i].cn++;
    }
    SortVocab();
    if (debug_mode > 0) {
        printf("Vocab size: %lld\n", vocab_size);
        printf("Words in train file: %lld\n", train_words);
    }
    file_size = ftell(fin);
    fclose(fin);
}

void InitNet() {
    // Inicializar red neuronal
    long long a, b;
    unsigned long long next_random = 1;
    a = posix_memalign((void**)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn0 == NULL) { printf("Memory allocation failed\n"); exit(1); }
    a = posix_memalign((void**)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1neg == NULL) { printf("Memory allocation failed\n"); exit(1); }
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++) {
        next_random = next_random * (unsigned long long)25214903917 + 11;
        syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
    }
    for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
        syn1neg[a * layer1_size + b] = 0;
}

void TrainModel() {
    // Entrenar el modelo Skip-Gram
    long long a, b, c, d, word, last_word, sentence_length = 0, sentence_position = 0;
    long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH];
    long long l1, l2, target, label;
    unsigned long long next_random = (long long)1;
    real f, g;
    clock_t now;
    FILE* fi = fopen(train_file, "rb");
    if (fi == NULL) { printf("ERROR: training data file not found\n"); exit(1); }

    fseek(fi, 0, SEEK_END);
    file_size = ftell(fi);
    fseek(fi, 0, SEEK_SET);

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
            while (1) {
                word = ReadWordIndex(fi);
                if (feof(fi)) break;
                if (word == -1) continue;
                word_count++;
                if (word == 0) break; // Fin de oración
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

        if (feof(fi)) break;
        if (word_count > train_words) break;

        word = sen[sentence_position];
        if (word == -1) continue;

        for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
            c = sentence_position - window + a;
            if (c < 0 || c >= sentence_length) continue;
            last_word = sen[c];
            if (last_word == -1) continue;

            // Entrenamiento Skip-Gram con Negative Sampling
            l1 = last_word * layer1_size;
            for (d = 0; d < layer1_size; d++) neu1e[d] = 0;

            // NEGATIVE SAMPLING
            for (d = 0; d < negative + 1; d++) {
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
                }\
                l2 = target * layer1_size;
                f = 0;
                for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
                if (f > MAX_EXP) g = (label - 1) * alpha;
                else if (f < -MAX_EXP) g = (label - 0) * alpha;
                else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
                for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
                for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
            }

            // Aprender pesos entre la palabra de entrada y la salida
            for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
        }

        sentence_position++;
        if (sentence_position >= sentence_length) {
            sentence_length = 0;
            continue;
        }
    }
    fclose(fi);
}

int ArgPos(char* str, int argc, char** argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}

int main(int argc, char** argv) {
    int i;
    if (argc == 1) {
        printf("WORD VECTOR estimation toolkit v 0.1b\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-train <file>\n");
        printf("\t\tUse text data from <file> to train the model\n");
        printf("\t-output <file>\n");
        printf("\t\tUse <file> to save the resulting word vectors\n");
        printf("\t-size <int>\n");
        printf("\t\tSet size of word vectors; default is 100\n");
        printf("\t-window <int>\n");
        printf("\t\tSet max skip length between words; default is 5\n");
        printf("\t-sample <float>\n");
        printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency");
        printf("\t\tin the training data will be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
        printf("\t-binary <int>\n");
        printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
        printf("\t-min-count <int>\n");
        printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
        printf("\t-threads <int>\n");
        printf("\t\tUse <int> threads (default 1)\n");
        printf("\t-alpha <float>\n");
        printf("\t\tSet the starting learning rate; default is 0.025\n");
        printf("\nExamples:\n");
        printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -binary 0 -min-count 5\n\n");
        return 0;
    }

    output_file[0] = 0;
    save_vocab_file[0] = 0;
    read_vocab_file[0] = 0;
    if ((i = ArgPos((char*)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char*)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
    if ((i = ArgPos((char*)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char*)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char*)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
    if ((i = ArgPos((char*)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
    if ((i = ArgPos((char*)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
    if ((i = ArgPos((char*)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
    if ((i = ArgPos((char*)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
    if ((i = ArgPos((char*)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
    if ((i = ArgPos((char*)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char*)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);

    vocab = (struct vocab_word*)calloc(vocab_max_size, sizeof(struct vocab_word));
    vocab_hash = (int*)calloc(vocab_hash_size, sizeof(int));
    expTable = (real*)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
    for (i = 0; i < EXP_TABLE_SIZE; i++) {
        expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute exp(x)
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }

    LearnVocabFromTrainFile();
    InitNet();
    InitUnigramTable();
    start = clock();
    TrainModel();

    // Guardar vectores
    FILE* fo = fopen(output_file, "wb");
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
        fprintf(fo, "%s ", vocab[a].word);
        if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
        else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
        fprintf(fo, "\n");
    }
    fclose(fo);

    free(vocab);
    free(vocab_hash);
    free(syn0);
    free(syn1neg);
    free(expTable);

    return 0;
}


