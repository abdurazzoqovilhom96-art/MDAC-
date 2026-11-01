// Evaluate the official CEC2017 C reference on points read from a binary file.
//   usage: cec17_eval <func> <dim> <n> <in.bin> <out.bin>
// in.bin holds n*dim little-endian doubles (row-major); out.bin gets n doubles.
// Run with the working directory containing input_data/ (the reference code
// opens "input_data/..." relative to the working directory).
#include "cec17_test_func.cpp"
#include <cstdlib>

double *OShift, *M, *y, *z, *x_bound;
int ini_flag = 0, n_flag, func_flag, *SS;

int main(int argc, char **argv)
{
    if (argc != 6) {
        fprintf(stderr, "usage: cec17_eval func dim n in.bin out.bin\n");
        return 2;
    }
    int func = atoi(argv[1]), dim = atoi(argv[2]), n = atoi(argv[3]);
    size_t cnt = (size_t)n * (size_t)dim;
    double *x = (double *)malloc(sizeof(double) * cnt);
    double *f = (double *)malloc(sizeof(double) * n);
    FILE *fi = fopen(argv[4], "rb");
    if (!fi) return 3;
    if (fread(x, sizeof(double), cnt, fi) != cnt) return 4;
    fclose(fi);
    FILE *probe = fopen("input_data/shift_data_1.txt", "r");
    if (!probe) { fprintf(stderr, "input_data/ not found in working dir\n"); return 5; }
    fclose(probe);
    cec17_test_func(x, f, dim, n, func);
    FILE *fo = fopen(argv[5], "wb");
    if (!fo) return 6;
    fwrite(f, sizeof(double), n, fo);
    fclose(fo);
    return 0;
}
