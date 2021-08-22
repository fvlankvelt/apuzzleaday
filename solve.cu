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

#define CUDA_STACK_SIZE 1536

#ifndef DEBUG
#define NUM_THREADS_PER_BLOCK 64
#define NUM_BLOCKS   (2560 / NUM_THREADS_PER_BLOCK)
#define NUM_THREADS  (NUM_BLOCKS * NUM_THREADS_PER_BLOCK)
#define N_SOLUTIONS_PER_THREAD 2048
#else
#define NUM_THREADS_PER_BLOCK 1
#define NUM_BLOCKS   1
#define NUM_THREADS  (NUM_BLOCKS * NUM_THREADS_PER_BLOCK)
#define N_SOLUTIONS_PER_THREAD (4096 * 128)
#endif

#ifndef DEBUG
#define DEVICE __device__
#define GLOBAL __global__
#define SHARED extern __shared__
#else
#define DEVICE
#define GLOBAL
#define SHARED
#endif

typedef char cell;
typedef unsigned long long int occupied_t;

typedef struct _definition {
  bool mirror;
  int rotations;
  int height;
  int width;
  bool * set;
} definition_t;

typedef struct _orientations {
  bool flipped[8];
  int width[8];
  int height[8];
  occupied_t occupied[8];
} orientations_t;

typedef struct _pieces {
  cell value[N_PIECES];
  int n_orientations[N_PIECES];
  orientations_t orientations[N_PIECES];
} pieces_t;

typedef struct _board {
  occupied_t overlay;
  occupied_t pieces[N_PIECES];
  bool flipped[N_PIECES];
} board_t;

typedef struct _rotation {
  int x_x;
  int x_y;
  int y_x;
  int y_y;
} rotation;

inline bool isset(occupied_t o, int i, int j) {
  return o & ((unsigned long long int) 1) << (j * BOARD_WIDTH + i);
}

DEVICE static inline void board_set(board_t * board, int i, int j) {
  board->overlay |= ((unsigned long long int) 1) << (j * BOARD_WIDTH + i);
}

DEVICE static inline void board_unset(board_t * board, int i, int j) {
  board->overlay &= 0xffffffffffffffff ^ (1 << (j * BOARD_WIDTH + i));
}

#ifdef DEBUG
static inline int thread_id() { return 0; }
#else
__device__ static inline int thread_id() {
  return threadIdx.x + blockDim.x * blockIdx.x;
}
#endif

void print_orientation(char * prefix, const orientations_t * o, int idx) {
  for (int j = 0; j < o->height[idx]; j++) {
    printf(prefix);
    for (int i = 0; i < o->width[idx]; i++) {
      if (isset(o->occupied[idx], i, j)) {
        printf("x ");
      } else {
        printf("  ");
      }
    }
    printf("\n");
  }
}

void print_occupation(char * prefix, const occupied_t o) {
  printf("%d => \n", o);
  for (int y = 0; y < BOARD_HEIGHT; y++) {
    printf(prefix);
    for (int x = 0; x < BOARD_WIDTH; x++) {
      if (isset(o, x, y)) {
        printf("x ");
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
    if (board->flipped[p]) {
      fprintf(stream, "F");
    } else {
      fprintf(stream, "O");
    }
  }
  fprintf(stream, "\n");
  for (int p = 0; p < N_PIECES; p++) {
    fprintf(stream, "%d:\n", p);
    for (int j = 0; j < BOARD_HEIGHT; j++) {
      fprintf(stream, "  ");
      for (int i = 0; i < BOARD_WIDTH; i++) {
        if (isset(board->pieces[p], i, j)) {
          fprintf(stream, "x ");
        } else {
          fprintf(stream, ". ");
        }
      }
      fprintf(stream, "\n");
    }
    fprintf(stream, "\n");
  }
}

#ifdef DEBUG
const definition_t definitions[N_PIECES] = {
    { false, 2, 2, 3, (bool[]) {
        1, 1, 1,
        1, 1, 1 }, },
    { true, 4, 2, 3, (bool[]) {
        1, 1, 0,
        1, 1, 1 } },
    { false, 4, 2, 3, (bool[]) {
        1, 1, 1,
        1, 0, 1 } },
    { false, 4, 3, 3, (bool[]) {
        1, 1, 1,
        1, 0, 0,
        1, 0, 0 } },
    { true, 2, 3, 3, (bool[]) {
        1, 1, 0,
        0, 1, 0,
        0, 1, 1 } },
    { true, 4, 2, 4, (bool[]) {
        1, 1, 1, 1,
        1, 0, 0, 0 } },
    { true, 4, 2, 4, (bool[]) {
        1, 1, 1, 1,
        0, 0, 1, 0 } },
    { true, 4, 2, 4, (bool[]) {
        0, 1, 1, 1,
        1, 1, 0, 0 } },
  };
#endif

DEVICE void initialize_pieces(pieces_t * pieces) {
    // printf("Initializing pieces\n");
    // print_orientation("", &def->orientation);

#ifndef DEBUG
const definition_t definitions[N_PIECES] = {
    { false, 2, 2, 3, (bool[]) {
        1, 1, 1,
        1, 1, 1 }, },
    { true, 4, 2, 3, (bool[]) {
        1, 1, 0,
        1, 1, 1 } },
    { false, 4, 2, 3, (bool[]) {
        1, 1, 1,
        1, 0, 1 } },
    { false, 4, 3, 3, (bool[]) {
        1, 1, 1,
        1, 0, 0,
        1, 0, 0 } },
    { true, 2, 3, 3, (bool[]) {
        1, 1, 0,
        0, 1, 0,
        0, 1, 1 } },
    { true, 4, 2, 4, (bool[]) {
        1, 1, 1, 1,
        1, 0, 0, 0 } },
    { true, 4, 2, 4, (bool[]) {
        1, 1, 1, 1,
        0, 0, 1, 0 } },
    { true, 4, 2, 4, (bool[]) {
        0, 1, 1, 1,
        1, 1, 0, 0 } },
  };
#endif

  const rotation rotations[4] = {
    {  1,  0,  0,  1 },
    {  0,  1, -1,  0 },
    { -1,  0,  0, -1 },
    {  0, -1,  1,  0 },
  };

  for (int p = 0; p < N_PIECES; p++) {
    // printf("PIECE %d\n", p);
    const definition_t * def = &definitions[p];

    int n_flips = def->mirror + 1;
    int n_orientations = n_flips * def->rotations;
    pieces->n_orientations[p] = n_orientations;

    orientations_t * orientations = &pieces->orientations[p];
    for (int o = 0; o < n_flips; o++) {
      int x_dir = (o == 0 ? 1 : -1);
      for (int r = 0; r < def->rotations; r++) {
        bool shift_i = x_dir * rotations[r].x_x < 0 || x_dir * rotations[r].x_y < 0;
        bool shift_j = rotations[r].y_x < 0 || rotations[r].y_y < 0;

        int idx = o * def->rotations + r;
        bool rotated = r % 2 == 0;
        orientations->width[idx] = (rotated ? def->width : def->height);
        orientations->height[idx] = (rotated ? def->height : def->width);
        orientations->flipped[idx] = o > 0;
        orientations->occupied[idx] = 0;
        for (int j = 0; j < orientations->height[idx]; j++) {
          for (int i = 0; i < orientations->width[idx]; i++) {
            int i_orig = shift_i * (def->width - 1) + x_dir * (rotations[r].x_x * i + rotations[r].x_y * j);
            int j_orig = shift_j * (def->height - 1) + rotations[r].y_x * i + rotations[r].y_y * j;
            int target_idx = j * BOARD_WIDTH + i;
            if (def->set[j_orig * def->width + i_orig]) {
              orientations->occupied[idx] |= 1 << target_idx;
            }
          }
        }
        // print_orientation("   ", orientations, idx);
        // printf("\n");
      }
    }
    // printf("\n");
  }
}

DEVICE void initialize_board(board_t * board) {
  board->overlay = 0xffffffffffffffff << (BOARD_WIDTH * BOARD_HEIGHT);
  for (int p = 0; p < N_PIECES; p++) {
    board->pieces[p] = 0;
    board->flipped[p] = false;
  }
  board_set(board, 6, 0);
  board_set(board, 6, 1);
  board_set(board, 3, 6);
  board_set(board, 4, 6);
  board_set(board, 5, 6);
  board_set(board, 6, 6);
}

DEVICE bool place_piece(board_t * board, const occupied_t o, int x, int y, int p) {
  occupied_t placed = (o << (y * BOARD_WIDTH + x));
  if (board->overlay & placed) {
    return false;
  }
  board->overlay |= placed;
  board->pieces[p] = placed;
  return true;
}

DEVICE void unplace_piece(board_t * board, int p) {
  board->overlay &= (0xffffffffffffffff ^ board->pieces[p]);
}

typedef struct _context {
  const pieces_t * pieces;
  board_t * solutions;
  board_t board;
  int n_solutions;
  int task;
} context_t;

DEVICE int add_solution(context_t * ctx) {
    // printf("adding solution (%d)\n", ctx->n_solutions);
    // print_board(stdout, &ctx->board);
    if (ctx->n_solutions == N_SOLUTIONS_PER_THREAD) {
      printf("OOUT OF MEMORY\n");
      return OUT_OF_MEMORY;
    }
    // board_t * result = &ctx->solutions[ctx->n_solutions];
    // memcpy(result, &ctx->board, sizeof(board_t));
    ctx->n_solutions = ctx->n_solutions + 1;
    return OK;
}

DEVICE int solve(context_t * ctx, int p) {
  if (p == N_PIECES) {
  // if (p == 7) {
    return add_solution(ctx);
  }

  const orientations_t * orientations = &ctx->pieces->orientations[p];
  for (int o = 0; o < ctx->pieces->n_orientations[p]; o++) {
    for (int y = 0; y <= BOARD_HEIGHT - orientations->height[o]; y++) {
      for (int x = 0; x <= BOARD_WIDTH - orientations->width[o]; x++) {
        if (p == 2) {
          ctx->task = ctx->task + 1;
          if ((ctx->task % NUM_THREADS) != thread_id()) {
            continue;
          }
        }
        bool placed = place_piece(&ctx->board, orientations->occupied[o], x, y, p);
        if (placed) {
          // printf("adding %d\n", p);
          // print_occupation(" d ", orientations->occupied[o]);
          // print_occupation(" + ", ctx->board.pieces[p]);
          ctx->board.flipped[p] = orientations->flipped[o];
          int subresult = solve(ctx, p + 1);
          if (subresult) {
            return subresult;
          }
          unplace_piece(&ctx->board, p);
        }
      }
    }
  }
  return OK;
}

const char * months[] = {
  "jan", "feb", "mar", "apr", "mai", "jun", "jul", "aug", "sep", "okt", "nov", "des"
};
DEVICE const int days[] = {
  31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31
};

GLOBAL static void run_thread(const int month, board_t * solutions, int * n_thread_solutions) {
  int tid = thread_id();

  SHARED pieces_t pieces;
  if ((thread_id() % NUM_THREADS_PER_BLOCK) == 0) {
    initialize_pieces(&pieces);
  }

#ifndef DEBUG
  __syncthreads();
#endif

  context_t ctx = {&pieces, &solutions[tid * N_SOLUTIONS_PER_THREAD]};
  initialize_board(&ctx.board);
  ctx.n_solutions = 0;
  ctx.task = 0;

  int month_i = month % 6;
  int month_j = month / 6;
  board_set(&ctx.board, month_i, month_j);

  for (int day = 0; day < days[month]; day++) {
  // for (int day = 0; day < 1; day++) {
    int day_i = day % 7;
    int day_j = day / 7 + 2;
    board_set(&ctx.board, day_i, day_j);

    int result = solve(&ctx, 0);
    if (result) {
      n_thread_solutions[tid] = -result;
      return;
    }

    board_unset(&ctx.board, day_i, day_j);
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

  int setLimit = cudaDeviceSetLimit(cudaLimitStackSize, CUDA_STACK_SIZE);
  if (setLimit) {
   printf("Setting limit: %d\n", setLimit);
  }

  // device memory
  board_t * dsolutions = NULL;
  int *dn_solutions = NULL;
  cudaMalloc((void **) &dsolutions, sizeof(board_t) * N_SOLUTIONS_PER_THREAD * NUM_THREADS);
  cudaMalloc((void **) &dn_solutions, sizeof(int) * NUM_THREADS);

  // host memory
  board_t * solutions = (board_t *) malloc(sizeof(board_t) * N_SOLUTIONS_PER_THREAD * NUM_THREADS);
  int n_solutions[NUM_THREADS];

#ifdef DEBUG
  run_thread(month, solutions, n_solutions);
#else
  run_thread<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(month, dsolutions, dn_solutions);
  cudaError_t err = cudaGetLastError();
  printf("Err: %s\n", cudaGetErrorString(err));

  cudaMemcpy(solutions, dsolutions, sizeof(board_t) * N_SOLUTIONS_PER_THREAD * NUM_THREADS, cudaMemcpyDeviceToHost);
  cudaMemcpy(n_solutions, dn_solutions, sizeof(int) * NUM_THREADS, cudaMemcpyDeviceToHost);
#endif

  cudaFree(dsolutions);
  cudaFree(dn_solutions);

  int total = 0;
  for (int tid = 0; tid < NUM_THREADS; tid++) {
    total += n_solutions[tid];
    // if (n_solutions[tid] < 0) {
      printf("%d: %d\n", tid, n_solutions[tid]);
    // }
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
