// System includes
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

#define BOARD_WIDTH 7
#define BOARD_HEIGHT 7

#define N_PIECES 8

#define OK 0
#define OUT_OF_MEMORY 1

#define NUM_THREADS_PER_BLOCK 128
#define NUM_BLOCKS   (2560 / NUM_THREADS_PER_BLOCK)
#define NUM_THREADS  (NUM_BLOCKS * NUM_THREADS_PER_BLOCK)
#define N_SOLUTIONS_PER_THREAD 128

typedef char cell;

typedef struct _orientation {
  int height;
  int width;
  bool flipped;
  cell value[9];
} orientation;

typedef struct _definition {
  bool mirror;
  int rotations;
  int height;
  int width;
  cell * value;
} definition_t;

typedef struct _piece {
  int n_orientations;
  orientation orientations[8];
} piece_t;

typedef struct _board {
  cell cells[BOARD_WIDTH * BOARD_HEIGHT];
  bool flips[N_PIECES];
} board_t;

typedef struct _rotation {
  int x_x;
  int x_y;
  int y_x;
  int y_y;
} rotation;

__device__ static inline void board_set(board_t * board, int i, int j, cell value) {
  board->cells[j * BOARD_WIDTH + i] = value;
}

__device__ static inline cell board_get(board_t * board, int i, int j) {
  return board->cells[j * BOARD_WIDTH + i];
}

__device__ static inline int thread_id() {
  return threadIdx.x + blockDim.x * blockIdx.x;
}

void print_orientation(char * prefix, orientation * o) {
  for (int j = 0; j < o->height; j++) {
    printf(prefix);
    for (int i = 0; i < o->width; i++) {
      int idx = j * o->width + i;
      if (o->value[idx]) {
        printf("%c ", o->value[idx]);
      } else {
        printf("  ");
      }
    }
    printf("\n");
  }
}

void print_board(FILE * stream, board_t * board) {
  fprintf(stream, "flips: ");
  for (int p = 0; p < N_PIECES; p++) {
    if (board->flips[p]) {
      fprintf(stream, "F");
    } else {
      fprintf(stream, "O");
    }
  }
  fprintf(stream, "\n");
  for (int j = 0; j < BOARD_HEIGHT; j++) {
    for (int i = 0; i < BOARD_WIDTH; i++) {
      int idx = j * BOARD_WIDTH + i;
      if (board->cells[idx]) {
        fprintf(stream, "%c ", board->cells[idx]);
      } else {
        fprintf(stream, "  ");
      }
    }
    fprintf(stream, "\n");
  }
}

__device__ void initialize_pieces(piece_t * pieces) {
    // printf("Initializing piece\n");
    // print_orientation("", &def->orientation);

  const definition_t definitions[N_PIECES] = {
    { false, 2, 2, 3, (cell[]) {
        'O', 'O', 'O',
        'O', 'O', 'O' } },
    { true, 4, 2, 3, (cell[]) {
        'P', 'P',  0,
        'P', 'P', 'P' } },
    { false, 4, 2, 3, (cell[]) {
        'C', 'C', 'C',
        'C',   0, 'C' } },
    { false, 4, 3, 3, (cell[]) {
        'L', 'L', 'L',
        'L',   0,   0,
        'L',   0,   0, } },
    { true, 2, 3, 3, (cell[]) {
        'S', 'S',   0,
          0, 'S',   0,
          0, 'S', 'S', } },
    { true, 4, 2, 4, (cell[]) {
        'j', 'j', 'j', 'j',
        'j',   0,   0,   0 } },
    { true, 4, 2, 4, (cell[]) {
        'T', 'T', 'T', 'T',
          0,   0, 'T',   0 } },
    { true, 4, 2, 4, (cell[]) {
          0, 'Z', 'Z', 'Z',
        'Z', 'Z',   0,   0 } },
  };

  const rotation rotations[4] = {
    {  1,  0,  0,  1 },
    {  0,  1, -1,  0 },
    { -1,  0,  0, -1 },
    {  0, -1,  1,  0 },
  };

  for (int p = 0; p < N_PIECES; p++) {
    const definition_t * def = &definitions[p];
    piece_t * piece = &pieces[p];

    int n_flips = def->mirror + 1;
    int n_orientations = n_flips * def->rotations;
    piece->n_orientations = n_orientations;

    int value_size = def->width * def->height * sizeof(cell);
    int n_values = value_size * n_orientations;
    for (int o = 0; o < n_flips; o++) {
      int x_dir = (o == 0 ? 1 : -1);
      for (int r = 0; r < def->rotations; r++) {
        bool shift_i = x_dir * rotations[r].x_x < 0 || x_dir * rotations[r].x_y < 0;
        bool shift_j = rotations[r].y_x < 0 || rotations[r].y_y < 0;

        int idx = o * def->rotations + r;
        bool rotated = r % 2 == 0;
        orientation * orientation = &piece->orientations[idx];
        orientation->width = (rotated ? def->width : def->height);
        orientation->height = (rotated ? def->height : def->width);
        orientation->flipped = o > 0;
        cell * value = orientation->value;
        for (int j = 0; j < orientation->height; j++) {
          for (int i = 0; i < orientation->width; i++) {
            int i_orig = shift_i * (def->width - 1) + x_dir * (rotations[r].x_x * i + rotations[r].x_y * j);
            int j_orig = shift_j * (def->height - 1) + rotations[r].y_x * i + rotations[r].y_y * j;
            value[j * orientation->width + i] = def->value[j_orig * def->width + i_orig];
          }
        }
        // print_orientation("   ", orientation);
        // printf("\n");
      }
    }
    // printf("\n");
  }
}

__device__ void initialize_board(board_t * board) {
  for (int j = 0; j < BOARD_HEIGHT; j++) {
    for (int i = 0; i < BOARD_WIDTH; i++) {
      board->cells[j * BOARD_WIDTH + i] = 0;
    }
  }
  for (int p = 0; p < N_PIECES; p++) {
    board->flips[p] = false;
  }
  board_set(board, 6, 0, 'B');
  board_set(board, 6, 1, 'B');
  board_set(board, 3, 6, 'B');
  board_set(board, 4, 6, 'B');
  board_set(board, 5, 6, 'B');
  board_set(board, 6, 6, 'B');
}

__device__ bool place_piece(board_t * board, const orientation * o, int x, int y) {
  for (int j = 0; j < o->height; j++) {
    for (int i = 0; i < o->width; i++) {
      if (o->value[j * o->width + i] && board_get(board, x + i, y + j)) {
        return false;
      }
    }
  }
  for (int j = 0; j < o->height; j++) {
    for (int i = 0; i < o->width; i++) {
      int idx = j * o->width + i;
      if (o->value[idx]) {
        board_set(board, x + i, y + j, o->value[idx]);
      }
    }
  }
  return true;
}

__device__ void unplace_piece(board_t * board, const orientation * o, int x, int y) {
  for (int j = 0; j < o->height; j++) {
    for (int i = 0; i < o->width; i++) {
      if (o->value[j * o->width + i]) {
        board_set(board, x + i, y + j, 0);
      }
    }
  }
}

typedef struct _context {
  const piece_t * pieces;
  board_t * solutions;
  board_t board;
  int n_solutions;
  int task;
} context_t;

__device__ int add_solution(context_t * ctx) {
    if (ctx->n_solutions == N_SOLUTIONS_PER_THREAD) {
      return OUT_OF_MEMORY;
    }
    board_t * result = &ctx->solutions[ctx->n_solutions];
    memcpy(result, &ctx->board, sizeof(board_t));
    ctx->n_solutions = ctx->n_solutions + 1;
    return OK;
}

__device__ int solve(context_t * ctx, int p) {
  if (p == N_PIECES) {
    return add_solution(ctx);
  }

  for (int o = 0; o < ctx->pieces[p].n_orientations; o++) {
    const orientation * orientation = &ctx->pieces[p].orientations[o];
    for (int y = 0; y <= BOARD_HEIGHT - orientation->height; y++) {
      for (int x = 0; x <= BOARD_WIDTH - orientation->width; x++) {
        if (p == 2) {
          ctx->task = ctx->task + 1;
          if ((ctx->task % NUM_THREADS) != thread_id()) {
            continue;
          }
        }
        bool placed = place_piece(&ctx->board, orientation, x, y);
        if (placed) {
          ctx->board.flips[p] = orientation->flipped;
          int subresult = solve(ctx, p + 1);
          if (subresult) {
            return subresult;
          }
          unplace_piece(&ctx->board, orientation, x, y);
          ctx->board.flips[p] = false;
        }
      }
    }
  }
  return OK;
}

const char * months[] = {
  "jan", "feb", "mar", "apr", "mai", "jun", "jul", "aug", "sep", "okt", "nov", "des"
};
__device__ const int days[] = {
  31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
};

__global__ static void run_thread(const int month, board_t * solutions, int * n_thread_solutions) {
  int tid = thread_id();

  extern __shared__ piece_t pieces[N_PIECES];
  if (threadIdx.x == 0) {
    initialize_pieces(pieces);
  }

  __syncthreads();

  context_t ctx = {pieces, &solutions[tid * N_SOLUTIONS_PER_THREAD]};
  initialize_board(&ctx.board);
  ctx.n_solutions = 0;
  ctx.task = 0;

  int month_i = month % 6;
  int month_j = month / 6;
  board_set(&ctx.board, month_i, month_j, 'D');

  for (int day = 0; day < days[month]; day++) {
  // for (int day = 0; day < 1; day++) {
    int day_i = day % 7;
    int day_j = day / 7 + 2;
    board_set(&ctx.board, day_i, day_j, 'D');

    int result = solve(&ctx, 0);
    if (result) {
      n_thread_solutions[tid] = result;
      return;
    }

    board_set(&ctx.board, day_i, day_j, 0);
  }

  n_thread_solutions[tid] = ctx.n_solutions;
}

bool select_gpu()
{
    int devicesCount;
    cudaGetDeviceCount(&devicesCount);
    for(int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
    {
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties, deviceIndex);
        if (deviceProperties.major >= 2
            && deviceProperties.minor >= 0)
        {
            cudaSetDevice(deviceIndex);
            return true;
        }
    }

    return false;
}

int main(int argc, char * argv[]) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <month> [--noflip]\n", argv[0]);
    return -1;
  }
  int month = atoi(argv[1]);

  bool gpu_selected = select_gpu();
  if (gpu_selected) {
    printf("Selected GPU\n");
  }

  int setFlags = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
  if (setFlags) {
   printf("Setting flags: %d\n", setFlags);
  }

  board_t * dsolutions = NULL;
  int *dn_solutions = NULL;
  cudaMalloc((void **) &dsolutions, sizeof(board_t) * N_SOLUTIONS_PER_THREAD * NUM_THREADS);
  cudaMalloc((void **) &dn_solutions, sizeof(int) * NUM_THREADS);

  run_thread<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(month, dsolutions, dn_solutions);
  cudaError_t err = cudaGetLastError();
  printf("Err: %s\n", cudaGetErrorString(err));

  board_t * solutions = (board_t *) malloc(sizeof(board_t) * N_SOLUTIONS_PER_THREAD * NUM_THREADS);
  int n_solutions[NUM_THREADS];
  cudaMemcpy(solutions, dsolutions, sizeof(board_t) * N_SOLUTIONS_PER_THREAD * NUM_THREADS, cudaMemcpyDeviceToHost);
  cudaMemcpy(n_solutions, dn_solutions, sizeof(int) * NUM_THREADS, cudaMemcpyDeviceToHost);

  cudaFree(dsolutions);
  cudaFree(dn_solutions);

  int total = 0;
  for (int tid = 0; tid < NUM_THREADS; tid++) {
    total += n_solutions[tid];
    if (n_solutions[tid] < 0) {
      printf("%d: %d\n", tid, n_solutions[tid]);
    }
  }
  printf("total: %d\n", total);

/*
  // for (int month = 0; month < 12; month++) {
    for (int day = 0; day < days[month]; day++) {
      char filename[11];
      sprintf(filename, "out/%s-%d", months[month], day + 1);
      printf("%s: ", filename);
      fflush(stdout);

      solution * solutions = solve(board, 0);
      int n_solutions = 0;
      FILE * out = fopen(filename, "w");
      while (solutions) {
        n_solutions++;
        print_board(out, &solutions->board);
        fprintf(out, "\n");
        solutions = solutions->next;
      }
      fclose(out);
      printf("%d\n", n_solutions);

      board_set(board, day_i, day_j, 0);
      board_set(board, month_i, month_j, 0);
    }
  // }
  */

  return EXIT_SUCCESS;
}
